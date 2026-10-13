#!/usr/bin/env python3
"""
Run a CPU-only Urban edge-validity matrix and aggregate metrics.

This script is meant for data validation, not final leaderboard runs. It forces
CPU execution by setting CUDA_VISIBLE_DEVICES="" and passing --no_gpu to main.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "b31_dualscalar_assign": "configs/presets/b31_dualscalar_assign.json",
    "g20_se_consistent_main": "archive/configs/historical_presets/g20_se_consistent_main.json",
    "b40_v31_msgcond": "archive/configs/historical_presets/b40_v31_msgcond.json",
    "b45_v31_msgcond_gs050": "configs/presets/b45_v31_msgcond_gs050.json",
    "b47_v31_msgcond_gs050_matchonly": "archive/configs/historical_presets/b47_v31_msgcond_gs050_matchonly.json",
    "b48_v31_msgcond_gs050_confgate": "archive/configs/historical_presets/b48_v31_msgcond_gs050_confgate.json",
}


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _load_preset(repo_root: Path, name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if p.exists():
        path = p
    elif name_or_path in PRESET_MAP:
        path = repo_root / PRESET_MAP[name_or_path]
    else:
        raise FileNotFoundError(f"Unknown preset: {name_or_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(metrics: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def _maybe_add_bool(cmd: list[str], preset: dict[str, Any], key: str, flag: str) -> None:
    if bool(preset.get(key, False)):
        cmd.append(flag)


def _build_cmd(
    dataset: str,
    preset_name: str,
    preset: dict[str, Any],
    seed: int,
    version: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "python",
        "main.py",
        "--dataset",
        dataset,
        "--root_path",
        args.root_path,
        "--task",
        "Clustering",
        "--version",
        version,
        "--save_path",
        f"{version}.pt",
        "--seed",
        str(seed),
        "--exp_iters",
        "1",
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--train_log_interval",
        str(args.train_log_interval),
        "--n_cluster_trials",
        str(args.n_cluster_trials),
        "--hid_dim",
        str(args.hid_dim),
        "--max_nums",
        str(args.max_nums),
        "--patience",
        "0",
        "--no_gpu",
        "--edge_variant",
        str(preset.get("edge_variant", "V1")),
        "--edge_hybrid_alpha",
        str(preset.get("edge_hybrid_alpha", 0.5)),
        "--edge_feat_temp",
        str(preset.get("edge_feat_temp", 1.0)),
        "--edge_input_prior_alpha",
        str(preset.get("edge_input_prior_alpha", 0.0)),
        "--edge_fusion_gamma",
        str(preset.get("edge_fusion_gamma", 1.0)),
        "--edge_fusion_gamma_sched_epochs",
        str(preset.get("edge_fusion_gamma_sched_epochs", 0)),
        "--edge_confidence_quantile",
        str(preset.get("edge_confidence_quantile", 0.0)),
        "--edge_adaptive_alpha_strength",
        str(preset.get("edge_adaptive_alpha_strength", 2.0)),
        "--edge_adaptive_alpha_bias",
        str(preset.get("edge_adaptive_alpha_bias", 0.0)),
        "--edge_reliability_temp",
        str(preset.get("edge_reliability_temp", 1.0)),
        "--edge_attr_hidden_dim",
        str(preset.get("edge_attr_hidden_dim", 64)),
        "--edge_attr_fusion_scale",
        str(preset.get("edge_attr_fusion_scale", 1.0)),
        "--edge_attr_pool_topk",
        str(preset.get("edge_attr_pool_topk", 1)),
        "--edge_msg_gate_scale",
        str(preset.get("edge_msg_gate_scale", 0.35)),
        "--edge_msg_confidence_temp",
        str(preset.get("edge_msg_confidence_temp", 1.0)),
        "--edge_attr_weight_blend",
        str(preset.get("edge_attr_weight_blend", 0.0)),
        "--edge_attr_weight_temp",
        str(preset.get("edge_attr_weight_temp", 1.0)),
        "--edge_attr_weight_apply_to",
        str(preset.get("edge_attr_weight_apply_to", "si_only")),
        "--edge_weight_learn_reg_lambda",
        str(preset.get("edge_weight_learn_reg_lambda", 0.02)),
        "--edge_weight_learn_logclip",
        str(preset.get("edge_weight_learn_logclip", 0.8)),
        "--edge_weight_learn_temp",
        str(preset.get("edge_weight_learn_temp", 1.0)),
        "--edge_weight_learn_apply_to",
        str(preset.get("edge_weight_learn_apply_to", "both")),
        "--edge_aug_prior_scale",
        str(preset.get("edge_aug_prior_scale", 0.0)),
        "--edge_aug_prior_mode",
        str(preset.get("edge_aug_prior_mode", "raw")),
        "--edge_state_temp",
        str(preset.get("edge_state_temp", 1.0)),
        "--edge_state_lambda_boundary",
        str(preset.get("edge_state_lambda_boundary", 0.1)),
        "--edge_state_lambda_support",
        str(preset.get("edge_state_lambda_support", 0.1)),
    ]

    if preset.get("edge_fusion_gamma_start", None) is not None:
        cmd += ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])]
    if preset.get("edge_fusion_gamma_end", None) is not None:
        cmd += ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])]

    _maybe_add_bool(cmd, preset, "edge_adaptive_alpha", "--edge_adaptive_alpha")
    _maybe_add_bool(cmd, preset, "edge_attr_hierarchical", "--edge_attr_hierarchical")
    _maybe_add_bool(cmd, preset, "edge_msg_conditioned", "--edge_msg_conditioned")
    _maybe_add_bool(cmd, preset, "edge_msg_matched_only", "--edge_msg_matched_only")
    _maybe_add_bool(cmd, preset, "edge_msg_confidence_gate", "--edge_msg_confidence_gate")
    _maybe_add_bool(cmd, preset, "edge_attr_pool_confidence", "--edge_attr_pool_confidence")
    _maybe_add_bool(cmd, preset, "append_generic_edge_attr", "--append_generic_edge_attr")
    _maybe_add_bool(cmd, preset, "edge_state_use_context", "--edge_state_use_context")

    if bool(args.known_only_eval):
        cmd.append("--known_only_eval")
    return cmd


def _read_metrics(repo_root: Path, version: str, dataset: str) -> dict[str, Any]:
    path = repo_root / "results" / version / f"{dataset}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(out_dir: Path, rows: list[dict[str, Any]], baseline: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)
    if df.empty:
        return

    ok = df[df["rc"] == 0].copy()
    if ok.empty:
        return

    by_condition = (
        ok.groupby("condition", as_index=False)
        .agg(
            runs=("dataset", "count"),
            nmi_mean=("nmi_mean", "mean"),
            nmi_std=("nmi_mean", "std"),
            ari_mean=("ari_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            modularity_mean=("modularity_mean", "mean"),
            conductance_weighted_mean=("conductance_weighted_mean", "mean"),
            pred_n_clusters_mean=("pred_n_clusters_mean", "mean"),
            msg_gate_mean=("final_msg_gate_factor_mean", "mean"),
            edge_factor_mean=("final_edge_factor_mean_mean", "mean"),
            diag_all_live_mean=("diag_all_required_live_mean", "mean"),
        )
        .sort_values(["nmi_mean", "ari_mean"], ascending=False)
    )
    by_condition.to_csv(out_dir / "summary_by_condition.csv", index=False)

    by_dataset = (
        ok.groupby(["dataset", "condition"], as_index=False)
        .agg(
            runs=("seed", "count"),
            nmi_mean=("nmi_mean", "mean"),
            ari_mean=("ari_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            modularity_mean=("modularity_mean", "mean"),
            conductance_weighted_mean=("conductance_weighted_mean", "mean"),
            pred_n_clusters_mean=("pred_n_clusters_mean", "mean"),
        )
        .sort_values(["dataset", "condition"])
    )
    by_dataset.to_csv(out_dir / "summary_by_dataset_condition.csv", index=False)

    if baseline in set(ok["condition"]):
        base = ok[ok["condition"] == baseline][["dataset", "seed", "nmi_mean", "ari_mean", "si_loss_mean"]].rename(
            columns={
                "nmi_mean": "baseline_nmi",
                "ari_mean": "baseline_ari",
                "si_loss_mean": "baseline_si_loss",
            }
        )
        delta = ok.merge(base, on=["dataset", "seed"], how="left")
        delta["delta_nmi_vs_baseline"] = delta["nmi_mean"] - delta["baseline_nmi"]
        delta["delta_ari_vs_baseline"] = delta["ari_mean"] - delta["baseline_ari"]
        delta["delta_si_loss_vs_baseline"] = delta["si_loss_mean"] - delta["baseline_si_loss"]
        delta.to_csv(out_dir / "delta_vs_baseline_runs.csv", index=False)

    readme = [
        "# Urban Edge Validity CPU Run",
        "",
        "Purpose: validate whether Urban edge semantics are useful before treating Urban as a main model result.",
        "",
        "Key files:",
        "- `runs.csv`: run-level metrics and command status.",
        "- `summary_by_condition.csv`: aggregate by preset/intervention condition.",
        "- `summary_by_dataset_condition.csv`: dataset-level comparison.",
        "- `delta_vs_baseline_runs.csv`: per-run delta if the baseline condition exists.",
        "",
        "Interpretation:",
        "- B45 should beat or at least change meaningfully versus baseline/G20 on native Urban graphs.",
        "- B45 on permuted edge_attr should drop versus original B45 if edge semantics are being used.",
        "- Relation drop/zero ablations should move metrics or structure metrics in directionally sensible ways.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Urban edge-validity matrix on CPU.")
    parser.add_argument("--datasets", type=str, default="urban_paris_plot_v3bsjg,urban_boston_plot_v3bsjg")
    parser.add_argument("--conditions", type=str, default="baseline_v1,b45_v31_msgcond_gs050")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--tag", type=str, default="urban_edge_validity_cpu_v1")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--train_log_interval", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--max_nums", type=int, default=12)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--baseline_condition", type=str, default="baseline_v1")
    parser.add_argument("--conda_env", type=str, default="gnn")
    parser.add_argument("--omp_threads", type=int, default=8)
    parser.add_argument("--known_only_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    datasets = _parse_csv(args.datasets)
    conditions = _parse_csv(args.conditions)
    seeds = _parse_int_csv(args.seeds)
    if not datasets or not conditions or not seeds:
        raise ValueError("datasets, conditions, and seeds must be non-empty.")

    out_dir = repo_root / "results" / "mainline_evidence" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(max(1, int(args.omp_threads)))
    env["MKL_NUM_THREADS"] = str(max(1, int(args.omp_threads)))

    for dataset in datasets:
        for condition in conditions:
            preset = _load_preset(repo_root, condition)
            for seed in seeds:
                version = f"mainline_evidence/raw_runs/{args.tag}/{condition}_{dataset}_s{seed}"
                metrics_path = repo_root / "results" / version / f"{dataset}_metrics.json"
                rc = 0
                seconds = 0.0
                if args.resume and metrics_path.exists():
                    print(f"[skip] {condition} {dataset} seed={seed}", flush=True)
                else:
                    inner = _build_cmd(
                        dataset=dataset,
                        preset_name=condition,
                        preset=preset,
                        seed=seed,
                        version=version,
                        args=args,
                    )
                    cmd = ["conda", "run", "-n", args.conda_env] + inner
                    log_path = out_dir / f"{condition}_{dataset}_s{seed}.log"
                    print("[run] " + " ".join(cmd), flush=True)
                    if args.dry_run:
                        rc = 0
                    else:
                        start = time.time()
                        with log_path.open("w", encoding="utf-8") as f:
                            f.write("CMD: " + " ".join(cmd) + "\n\n")
                            f.flush()
                            proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)
                            rc = int(proc.returncode)
                        seconds = time.time() - start
                        print(f"[done] rc={rc} seconds={seconds:.1f}", flush=True)

                metrics = _read_metrics(repo_root, version, dataset)
                row = {
                    "dataset": dataset,
                    "condition": condition,
                    "seed": int(seed),
                    "rc": int(rc),
                    "seconds": float(seconds),
                    "version": version,
                    "metrics_path": str(metrics_path),
                    "nmi_mean": _metric_value(metrics, "nmi_mean"),
                    "ari_mean": _metric_value(metrics, "ari_mean"),
                    "acc_mean": _metric_value(metrics, "acc_mean"),
                    "si_loss_mean": _metric_value(metrics, "si_loss_mean"),
                    "modularity_mean": _metric_value(metrics, "modularity_mean"),
                    "conductance_weighted_mean": _metric_value(metrics, "conductance_weighted_mean"),
                    "pred_n_clusters_mean": _metric_value(metrics, "pred_n_clusters_mean"),
                    "final_msg_gate_factor_mean": _metric_value(metrics, "final_msg_gate_factor_mean"),
                    "final_edge_factor_mean_mean": _metric_value(metrics, "final_edge_factor_mean_mean"),
                    "diag_all_required_live_mean": _metric_value(metrics, "diag_all_required_live_mean"),
                }
                rows.append(row)
                _aggregate(out_dir=out_dir, rows=rows, baseline=args.baseline_condition)

    _aggregate(out_dir=out_dir, rows=rows, baseline=args.baseline_condition)
    print(f"[ok] summary={out_dir}", flush=True)


if __name__ == "__main__":
    main()
