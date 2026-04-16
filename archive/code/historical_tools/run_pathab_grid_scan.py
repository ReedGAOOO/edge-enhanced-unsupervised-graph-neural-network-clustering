#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import shutil

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "computers", "photo"]
DATASET_MAX_NUMS = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
}


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_condition_filter(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def condition_space() -> List[Dict[str, object]]:
    base = {
        "edge_hybrid_alpha": 0.7,
        "edge_feat_temp": 1.0,
        "edge_input_prior_alpha": 0.35,
        "edge_fusion_gamma": 1.0,
        "edge_fusion_gamma_start": None,
        "edge_fusion_gamma_end": None,
        "edge_fusion_gamma_sched_epochs": 0,
        "edge_adaptive_alpha": False,
        "edge_adaptive_alpha_strength": 1.0,
        "edge_adaptive_alpha_bias": 2.0,
        "edge_reliability_temp": 1.0,
        "edge_confidence_quantile": 0.0,
        "edge_attr_hidden_dim": 64,
        "edge_attr_fusion_scale": 1.0,
        "edge_attr_weight_blend": 0.0,
        "edge_attr_weight_temp": 1.0,
        "edge_attr_weight_apply_to": "si_only",
        "edge_attr_hierarchical": False,
    }

    def m(name: str, edge_variant: str, **kwargs):
        c = dict(base)
        c.update({"condition": name, "edge_variant": edge_variant})
        c.update(kwargs)
        return c

    return [
        m("BASE_B0_V1", "V1"),
        # Path-B family (V12 + hierarchical edge-attr propagation).
        m(
            "G15_default",
            "V12",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=True,
            edge_adaptive_alpha_strength=2.0,
            edge_adaptive_alpha_bias=0.0,
            edge_attr_fusion_scale=0.7,
            edge_attr_hierarchical=True,
        ),
        m(
            "G15_fusion1p0",
            "V12",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=True,
            edge_adaptive_alpha_strength=2.0,
            edge_adaptive_alpha_bias=0.0,
            edge_attr_fusion_scale=1.0,
            edge_attr_hierarchical=True,
        ),
        m(
            "G15_quant0p1",
            "V12",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=True,
            edge_adaptive_alpha_strength=2.0,
            edge_adaptive_alpha_bias=0.0,
            edge_confidence_quantile=0.1,
            edge_attr_fusion_scale=0.7,
            edge_attr_hierarchical=True,
        ),
        m(
            "G15_noadapt",
            "V12",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=False,
            edge_attr_fusion_scale=0.7,
            edge_attr_hierarchical=True,
        ),
        m(
            "G15_sched_soft",
            "V12",
            edge_fusion_gamma_start=0.1,
            edge_fusion_gamma_end=1.0,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=True,
            edge_adaptive_alpha_strength=2.0,
            edge_adaptive_alpha_bias=0.0,
            edge_attr_fusion_scale=0.7,
            edge_attr_hierarchical=True,
        ),
        m(
            "G15_sched_strong",
            "V12",
            edge_fusion_gamma_start=0.3,
            edge_fusion_gamma_end=1.5,
            edge_fusion_gamma_sched_epochs=100,
            edge_adaptive_alpha=True,
            edge_adaptive_alpha_strength=2.0,
            edge_adaptive_alpha_bias=0.0,
            edge_attr_fusion_scale=0.7,
            edge_attr_hierarchical=True,
        ),
        # Path-A family on a stable V5 trunk.
        m(
            "G17_default",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.5,
            edge_attr_weight_temp=1.0,
            edge_attr_weight_apply_to="si_only",
        ),
        m(
            "G17_blend0p3",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.3,
            edge_attr_weight_temp=1.0,
            edge_attr_weight_apply_to="si_only",
        ),
        m(
            "G17_blend0p7",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.7,
            edge_attr_weight_temp=1.0,
            edge_attr_weight_apply_to="si_only",
        ),
        m(
            "G17_temp0p7",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.5,
            edge_attr_weight_temp=0.7,
            edge_attr_weight_apply_to="si_only",
        ),
        m(
            "G17_temp1p5",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.5,
            edge_attr_weight_temp=1.5,
            edge_attr_weight_apply_to="si_only",
        ),
        m(
            "G17_apply_both",
            "V5",
            edge_fusion_gamma_start=0.2,
            edge_fusion_gamma_end=1.2,
            edge_fusion_gamma_sched_epochs=100,
            edge_attr_weight_blend=0.5,
            edge_attr_weight_temp=1.0,
            edge_attr_weight_apply_to="both",
        ),
    ]


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def dataset_max_nums(name: str, default_max: int) -> int:
    return int(DATASET_MAX_NUMS.get(name.lower(), default_max))


def build_cmd(dataset: str, seed: int, cond: Dict[str, object], args, run_dir: Path) -> List[str]:
    max_nums = dataset_max_nums(dataset, args.default_max_nums)
    run_ckpt = f"{args.tag}_{cond['condition']}_{dataset}_s{seed}.pt"
    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset,
        "--task",
        "Clustering",
        "--version",
        "pathab_grid",
        "--log_path",
        str(run_dir / f"{dataset}.log"),
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
        "--save_path",
        run_ckpt,
        "--gpu",
        str(args.gpu),
        "--hid_dim",
        str(args.hid_dim),
        "--n_cluster_trials",
        str(args.n_cluster_trials),
        "--max_nums",
        str(max_nums),
        "--knn",
        str(args.knn),
        "--knn_mode",
        str(args.knn_mode),
        "--knn_auto_threshold",
        str(args.knn_auto_threshold),
        "--edge_variant",
        str(cond["edge_variant"]),
        "--edge_hybrid_alpha",
        str(cond["edge_hybrid_alpha"]),
        "--edge_feat_temp",
        str(cond["edge_feat_temp"]),
        "--edge_input_prior_alpha",
        str(cond["edge_input_prior_alpha"]),
        "--edge_fusion_gamma",
        str(cond["edge_fusion_gamma"]),
        "--edge_fusion_gamma_sched_epochs",
        str(cond["edge_fusion_gamma_sched_epochs"]),
        "--edge_confidence_quantile",
        str(cond["edge_confidence_quantile"]),
        "--edge_adaptive_alpha_strength",
        str(cond["edge_adaptive_alpha_strength"]),
        "--edge_adaptive_alpha_bias",
        str(cond["edge_adaptive_alpha_bias"]),
        "--edge_reliability_temp",
        str(cond["edge_reliability_temp"]),
        "--edge_attr_hidden_dim",
        str(cond["edge_attr_hidden_dim"]),
        "--edge_attr_fusion_scale",
        str(cond["edge_attr_fusion_scale"]),
        "--edge_attr_weight_blend",
        str(cond["edge_attr_weight_blend"]),
        "--edge_attr_weight_temp",
        str(cond["edge_attr_weight_temp"]),
        "--edge_attr_weight_apply_to",
        str(cond["edge_attr_weight_apply_to"]),
        "--patience",
        str(args.patience),
    ]
    if cond["edge_fusion_gamma_start"] is not None:
        cmd += ["--edge_fusion_gamma_start", str(cond["edge_fusion_gamma_start"])]
    if cond["edge_fusion_gamma_end"] is not None:
        cmd += ["--edge_fusion_gamma_end", str(cond["edge_fusion_gamma_end"])]
    if bool(cond["edge_adaptive_alpha"]):
        cmd += ["--edge_adaptive_alpha"]
    if bool(cond["edge_attr_hierarchical"]):
        cmd += ["--edge_attr_hierarchical"]
    if args.amp_bf16:
        cmd += ["--amp_bf16"]
    return cmd


def aggregate(out_dir: Path, run_rows: List[Dict[str, object]], baseline_condition: str) -> None:
    runs = pd.DataFrame(run_rows)
    runs.to_csv(out_dir / "runs.csv", index=False)

    ok = runs[runs["status"] == "ok"].copy()
    if ok.empty:
        return

    by_cd = (
        ok.groupby(["condition", "dataset"], as_index=False)
        .agg(
            n_runs_ok=("status", "count"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            seconds_mean=("seconds", "mean"),
        )
        .fillna(0.0)
    )
    by_cd.to_csv(out_dir / "summary_by_condition_dataset.csv", index=False)

    base = by_cd[by_cd["condition"] == baseline_condition][["dataset", "nmi_mean", "ari_mean"]].rename(
        columns={"nmi_mean": "base_nmi", "ari_mean": "base_ari"}
    )
    delta = by_cd.merge(base, on="dataset", how="left")
    delta["delta_nmi_vs_baseline"] = delta["nmi_mean"] - delta["base_nmi"]
    delta["delta_ari_vs_baseline"] = delta["ari_mean"] - delta["base_ari"]
    delta.to_csv(out_dir / "delta_vs_baseline.csv", index=False)

    by_c = (
        delta.groupby("condition", as_index=False)
        .agg(
            datasets_covered=("dataset", "count"),
            mean_nmi=("nmi_mean", "mean"),
            mean_ari=("ari_mean", "mean"),
            mean_delta_nmi_vs_baseline=("delta_nmi_vs_baseline", "mean"),
            mean_delta_ari_vs_baseline=("delta_ari_vs_baseline", "mean"),
            win_rate_nmi_vs_baseline=("delta_nmi_vs_baseline", lambda s: float((s > 0).mean())),
            win_rate_ari_vs_baseline=("delta_ari_vs_baseline", lambda s: float((s > 0).mean())),
            mean_seconds=("seconds_mean", "mean"),
            avg_nmi_std=("nmi_std", "mean"),
            avg_ari_std=("ari_std", "mean"),
            tail_p10_delta_nmi=("delta_nmi_vs_baseline", lambda s: float(np.percentile(s, 10))),
            tail_p10_delta_ari=("delta_ari_vs_baseline", lambda s: float(np.percentile(s, 10))),
            min_delta_nmi=("delta_nmi_vs_baseline", "min"),
            min_delta_ari=("delta_ari_vs_baseline", "min"),
            worse_datasets_nmi=("delta_nmi_vs_baseline", lambda s: int((s <= 0).sum())),
            worse_datasets_ari=("delta_ari_vs_baseline", lambda s: int((s <= 0).sum())),
        )
        .fillna(0.0)
    )
    by_c["rank_score"] = (
        by_c["mean_delta_nmi_vs_baseline"]
        + by_c["mean_delta_ari_vs_baseline"]
        + 0.1 * by_c["win_rate_nmi_vs_baseline"]
        + 0.1 * by_c["win_rate_ari_vs_baseline"]
        + 0.2 * by_c["tail_p10_delta_nmi"]
    )
    by_c.sort_values(["rank_score", "mean_delta_nmi_vs_baseline"], ascending=False, inplace=True)
    by_c.to_csv(out_dir / "summary_by_condition.csv", index=False)

    best_ds = delta.sort_values(["dataset", "nmi_mean"], ascending=[True, False]).groupby("dataset", as_index=False).head(1)
    best_ds = best_ds[
        ["dataset", "condition", "nmi_mean", "ari_mean", "delta_nmi_vs_baseline", "delta_ari_vs_baseline"]
    ].rename(columns={"condition": "best_condition_by_nmi"})
    best_ds.to_csv(out_dir / "best_condition_by_dataset.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Path-A/B grid scan with resume support")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_freq", type=int, default=180)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=3)
    parser.add_argument("--default_max_nums", type=int, default=10)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--knn_mode", type=str, default="auto", choices=["auto", "dense", "edge"])
    parser.add_argument("--knn_auto_threshold", type=int, default=20000)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--tag", type=str, default="benchmark_pathAB_grid_v1")
    parser.add_argument("--baseline_condition", type=str, default="BASE_B0_V1")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    conds = condition_space()
    cond_filter = parse_condition_filter(args.conditions)
    if cond_filter:
        conds = [c for c in conds if c["condition"] in cond_filter]

    if not any(c["condition"] == args.baseline_condition for c in conds):
        print(f"[warn] baseline condition '{args.baseline_condition}' not in selected conditions.")

    run_rows: List[Dict[str, object]] = []
    total = len(conds) * len(datasets) * len(seeds)
    idx = 0
    n_skip = 0
    n_fail = 0
    n_ok = 0

    for cond in conds:
        for ds in datasets:
            for seed in seeds:
                idx += 1
                cond_name = cond["condition"]
                run_id = f"{args.tag}_{cond_name}_{ds}_s{seed}"
                run_dir = repo_root / "results" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "runner.log"
                metrics_path = run_dir / f"{ds}_metrics.json"
                shared_metrics_path = repo_root / "results" / "pathab_grid" / f"{ds}_metrics.json"

                if metrics_path.exists() and not args.force_rerun:
                    try:
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        run_rows.append(
                            {
                                "condition": cond_name,
                                "dataset": ds,
                                "seed": seed,
                                "status": "skip_exists",
                                "nmi": safe_float(m.get("nmi_mean")),
                                "ari": safe_float(m.get("ari_mean")),
                                "seconds": safe_float(m.get("seconds", np.nan)),
                                "run_dir": str(run_dir),
                            }
                        )
                        n_skip += 1
                        print(f"[{idx}/{total}] skip {run_id}")
                        continue
                    except Exception:
                        pass

                cmd = build_cmd(ds, seed, cond, args, run_dir=run_dir)
                t0 = time.time()
                print(f"[{idx}/{total}] run {run_id}")
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write("CMD: " + " ".join(cmd) + "\n\n")
                    lf.flush()
                    proc = subprocess.run(cmd, cwd=repo_root, stdout=lf, stderr=subprocess.STDOUT)
                sec = time.time() - t0

                if proc.returncode == 0 and shared_metrics_path.exists():
                    try:
                        m = json.loads(shared_metrics_path.read_text(encoding="utf-8"))
                        shutil.copyfile(shared_metrics_path, metrics_path)
                        run_rows.append(
                            {
                                "condition": cond_name,
                                "dataset": ds,
                                "seed": seed,
                                "status": "ok",
                                "nmi": safe_float(m.get("nmi_mean")),
                                "ari": safe_float(m.get("ari_mean")),
                                "seconds": float(sec),
                                "run_dir": str(run_dir),
                                "edge_variant": m.get("edge_variant", cond["edge_variant"]),
                                "edge_attr_hierarchical": m.get(
                                    "edge_attr_hierarchical", cond["edge_attr_hierarchical"]
                                ),
                                "edge_attr_weight_blend": m.get(
                                    "edge_attr_weight_blend", cond["edge_attr_weight_blend"]
                                ),
                                "edge_attr_weight_temp": m.get(
                                    "edge_attr_weight_temp", cond["edge_attr_weight_temp"]
                                ),
                                "edge_attr_weight_apply_to": m.get(
                                    "edge_attr_weight_apply_to", cond["edge_attr_weight_apply_to"]
                                ),
                                "edge_confidence_quantile": m.get(
                                    "edge_confidence_quantile", cond["edge_confidence_quantile"]
                                ),
                                "edge_attr_fusion_scale": m.get(
                                    "edge_attr_fusion_scale", cond["edge_attr_fusion_scale"]
                                ),
                            }
                        )
                        n_ok += 1
                    except Exception:
                        run_rows.append(
                            {
                                "condition": cond_name,
                                "dataset": ds,
                                "seed": seed,
                                "status": "fail_parse",
                                "nmi": np.nan,
                                "ari": np.nan,
                                "seconds": float(sec),
                                "run_dir": str(run_dir),
                            }
                        )
                        n_fail += 1
                else:
                    run_rows.append(
                        {
                            "condition": cond_name,
                            "dataset": ds,
                            "seed": seed,
                            "status": "fail",
                            "nmi": np.nan,
                            "ari": np.nan,
                            "seconds": float(sec),
                            "run_dir": str(run_dir),
                        }
                    )
                    n_fail += 1
                    print(f"[warn] failed run {run_id}, check {log_path}")

                aggregate(out_dir, run_rows, baseline_condition=args.baseline_condition)

    aggregate(out_dir, run_rows, baseline_condition=args.baseline_condition)
    summary = {
        "tag": args.tag,
        "datasets": datasets,
        "seeds": seeds,
        "n_conditions": len(conds),
        "total_runs": total,
        "ok_runs": n_ok,
        "skip_exists_runs": n_skip,
        "fail_runs": n_fail,
        "runs_csv": str(out_dir / "runs.csv"),
        "summary_by_condition_dataset_csv": str(out_dir / "summary_by_condition_dataset.csv"),
        "delta_vs_baseline_csv": str(out_dir / "delta_vs_baseline.csv"),
        "summary_by_condition_csv": str(out_dir / "summary_by_condition.csv"),
        "best_condition_by_dataset_csv": str(out_dir / "best_condition_by_dataset.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
