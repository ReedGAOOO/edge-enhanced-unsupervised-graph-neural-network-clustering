#!/usr/bin/env python3
"""Top-level runner for all Urban 3-factor experiment packages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from urban_factor_experiment_utils import (
    ensure_dir,
    parse_cities,
    parse_csv,
    run_command,
    write_readme,
)


PYTHON_BIN = "/home/aitx/miniconda3/envs/gnn/bin/python"
R_CMD = ["conda", "run", "-n", "r", "Rscript"]


def _package_cmds(args: argparse.Namespace, results_root: Path, derived_root: Path) -> list[dict[str, Any]]:
    common = [
        "--cities",
        args.cities,
        "--seeds",
        args.seeds,
        "--max_edges",
        str(args.max_edges),
        "--base_graph_root",
        args.base_graph_root,
    ]
    return [
        {
            "package": "01_label_data",
            "cmd": [
                PYTHON_BIN,
                "tools/run_urban_label_data_experiments.py",
                *common,
                "--out_dir",
                str(results_root / "01_label_data"),
            ],
        },
        {
            "package": "02_graph_projection",
            "cmd": [
                PYTHON_BIN,
                "tools/run_urban_projection_experiments.py",
                *common,
                "--urban_root",
                args.urban_root,
                "--derived_graph_root",
                str(derived_root),
                "--out_dir",
                str(results_root / "02_graph_projection"),
                "--repo_root",
                ".",
                "--dry_run",
                str(args.dry_run),
            ]
            + (["--force"] if args.force else []),
        },
        {
            "package": "03_urban_form",
            "cmd": [
                PYTHON_BIN,
                "tools/run_urban_form_experiments.py",
                *common,
                "--urban_root",
                args.urban_root,
                "--out_dir",
                str(results_root / "03_urban_form"),
            ],
        },
    ]


def _collect_failures(results_root: Path) -> pd.DataFrame:
    frames = []
    for package in ["01_label_data", "02_graph_projection", "03_urban_form"]:
        path = results_root / package / "failures.csv"
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df.insert(0, "package", package)
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["package", "city", "dataset", "error", "traceback"])
    return pd.concat(frames, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(".").resolve()
    results_root = Path(args.results_root)
    derived_root = Path(args.derived_graph_root)
    ensure_dir(results_root)
    ensure_dir(results_root / "figures")
    ensure_dir(results_root / "report")
    ensure_dir(results_root / "01_label_data")
    ensure_dir(results_root / "02_graph_projection")
    ensure_dir(results_root / "03_urban_form")
    ensure_dir(derived_root)

    selected_packages = set(parse_csv(args.packages.lower()))
    if "all" in selected_packages:
        selected_packages = {"01_label_data", "02_graph_projection", "03_urban_form", "visualization", "report"}

    cities = parse_cities(args.cities, graph_root=Path(args.base_graph_root))
    manifest = {
        "created_at_unix": time.time(),
        "repo_root": str(repo_root),
        "results_root": str(results_root),
        "derived_graph_root": str(derived_root),
        "urban_root": args.urban_root,
        "base_graph_root": args.base_graph_root,
        "cities_arg": args.cities,
        "cities": cities,
        "seeds": args.seeds,
        "max_edges": int(args.max_edges),
        "cpu_policy": {
            "CUDA_VISIBLE_DEVICES": "",
            "python": PYTHON_BIN,
            "r": "conda run -n r Rscript",
        },
        "dry_run": str(args.dry_run).strip().lower() in {"1", "true", "yes", "y"},
        "force": bool(args.force),
    }
    (results_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    run_rows: list[dict[str, Any]] = []
    for item in _package_cmds(args, results_root, derived_root):
        if item["package"] not in selected_packages:
            continue
        started = time.time()
        res = run_command(item["cmd"], cwd=repo_root, dry_run=False)
        run_rows.append(
            {
                "stage": item["package"],
                "rc": int(res["rc"]),
                "elapsed_sec": float(time.time() - started),
                "cmd": res["cmd"],
                "stdout_tail": res["stdout"],
                "stderr_tail": res["stderr"],
            }
        )

    if "visualization" in selected_packages:
        cmd = [
            *R_CMD,
            "tools/plot_urban_factor_experiment_packages.R",
            "--results_root",
            str(results_root),
        ]
        started = time.time()
        res = run_command(cmd, cwd=repo_root, dry_run=False)
        run_rows.append(
            {
                "stage": "visualization",
                "rc": int(res["rc"]),
                "elapsed_sec": float(time.time() - started),
                "cmd": res["cmd"],
                "stdout_tail": res["stdout"],
                "stderr_tail": res["stderr"],
            }
        )

    if "report" in selected_packages:
        cmd = [
            *R_CMD,
            "tools/render_urban_factor_experiment_packages.R",
            "--results_root",
            str(results_root),
        ]
        started = time.time()
        res = run_command(cmd, cwd=repo_root, dry_run=False)
        run_rows.append(
            {
                "stage": "report",
                "rc": int(res["rc"]),
                "elapsed_sec": float(time.time() - started),
                "cmd": res["cmd"],
                "stdout_tail": res["stdout"],
                "stderr_tail": res["stderr"],
            }
        )

    run_log = pd.DataFrame(run_rows)
    run_log.to_csv(results_root / "run_log.csv", index=False)

    failures = _collect_failures(results_root)
    if not run_log.empty:
        failed_stage = run_log[run_log["rc"] != 0].copy()
        if not failed_stage.empty:
            stage_fail = pd.DataFrame(
                {
                    "package": failed_stage["stage"],
                    "city": "",
                    "dataset": "",
                    "error": failed_stage["stderr_tail"].fillna("").astype(str),
                    "traceback": failed_stage["stdout_tail"].fillna("").astype(str),
                }
            )
            failures = pd.concat([failures, stage_fail], ignore_index=True)
    failures.to_csv(results_root / "failure_manifest.csv", index=False)

    write_readme(
        results_root / "README.md",
        "Urban 3-Factor Experiment Packages",
        [
            "This directory contains CPU-only diagnostics for all selected Urban cities.",
            "",
            "Package layout:",
            "- `01_label_data/`: label coverage, class balance, and granularity controls.",
            "- `02_graph_projection/`: V3b relation interventions, edge-attribute permutation, and V4 construction sweep.",
            "- `03_urban_form/`: urban-form strata, matched-edge controls, and tile-level summaries.",
            "- `figures/`: R-generated figure set.",
            "- `report/`: rendered report side outputs.",
            "",
            "Top-level files:",
            "- `manifest.json`: run arguments and path policy.",
            "- `run_log.csv`: package command results.",
            "- `failure_manifest.csv`: aggregate package/city failures.",
        ],
    )

    return {
        "results_root": str(results_root),
        "derived_graph_root": str(derived_root),
        "n_failed_records": int(len(failures)),
        "run_log": str(results_root / "run_log.csv"),
        "failure_manifest": str(results_root / "failure_manifest.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Urban 3-factor experiment packages.")
    parser.add_argument("--cities", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_edges", type=int, default=30000)
    parser.add_argument("--urban_root", type=str, default="/mnt/e/urban network datasets")
    parser.add_argument(
        "--base_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data",
    )
    parser.add_argument(
        "--derived_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data/urban_factor_experiments",
    )
    parser.add_argument("--results_root", type=str, default="results/urban_factor_experiments")
    parser.add_argument("--packages", type=str, default="all")
    parser.add_argument("--dry_run", type=str, default="false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
