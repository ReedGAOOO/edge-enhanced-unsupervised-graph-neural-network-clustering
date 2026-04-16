#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

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

# Backward-compatible condition aliases.
CONDITION_ALIASES = {
    "B15_PATHB_v12_hier_sched60": "B15_ECHF_s60",
    "B15_PATHB_v12_hier": "B15_ECHF_main",
}


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_condition_filter(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def canonical_condition_name(name: str) -> str:
    return CONDITION_ALIASES.get(name, name)


def condition_space() -> List[Dict[str, object]]:
    return [
        {
            "condition": "B0_V1_baseline",
            "edge_variant": "V1",
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
        },
        {
            "condition": "B1_V2_struct_pre",
            "edge_variant": "V2",
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
        },
        {
            "condition": "B2_V3_feat_pre",
            "edge_variant": "V3",
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
        },
        {
            "condition": "B3_V4_hybrid_pre",
            "edge_variant": "V4",
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
        },
        {
            "condition": "B4_V5_mid_adapt",
            "edge_variant": "V5",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
        },
        {
            "condition": "B5_V5_mid_no_adapt",
            "edge_variant": "V5",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": False,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
        },
        {
            "condition": "B6_V6_attr_gate",
            "edge_variant": "V6",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": False,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 1.0,
        },
        {
            "condition": "B7_V7_attr_align",
            "edge_variant": "V7",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 1.0,
        },
        {
            "condition": "B8_V6_attr_gate_strong",
            "edge_variant": "V6",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.3,
            "edge_fusion_gamma_end": 1.5,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 96,
            "edge_attr_fusion_scale": 1.5,
        },
        {
            "condition": "B9_V7_attr_align_conservative",
            "edge_variant": "V7",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.1,
            "edge_fusion_gamma_end": 0.8,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": False,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.1,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.8,
        },
        {
            "condition": "B10_V8_attr_calibrated",
            "edge_variant": "V8",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 2.0,
            "edge_adaptive_alpha_bias": 0.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 1.0,
        },
        {
            "condition": "B11_V8_attr_calibrated_conservative",
            "edge_variant": "V8",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.1,
            "edge_fusion_gamma_end": 0.9,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 1.5,
            "edge_adaptive_alpha_bias": -0.5,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.1,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.8,
        },
        {
            "condition": "B12_V12_residual_calibrated",
            "edge_variant": "V12",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 2.0,
            "edge_adaptive_alpha_bias": 0.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.7,
        },
        {
            "condition": "B13_V12_residual_calibrated_conservative",
            "edge_variant": "V12",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.1,
            "edge_fusion_gamma_end": 0.9,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 1.5,
            "edge_adaptive_alpha_bias": -0.5,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.1,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.5,
        },
        {
            "condition": "B14_PATHA_only_v1",
            "edge_variant": "V1",
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
            "edge_attr_weight_blend": 0.5,
            "edge_attr_weight_temp": 1.0,
            "edge_attr_weight_apply_to": "si_only",
            "edge_attr_hierarchical": False,
        },
        {
            "condition": "B15_ECHF_s60",
            "edge_variant": "V12",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 60,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 2.0,
            "edge_adaptive_alpha_bias": 0.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.7,
            "edge_attr_weight_blend": 0.0,
            "edge_attr_weight_temp": 1.0,
            "edge_attr_weight_apply_to": "si_only",
            "edge_attr_hierarchical": True,
        },
        {
            "condition": "B15_ECHF_main",
            "edge_variant": "V12",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 2.0,
            "edge_adaptive_alpha_bias": 0.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.7,
            "edge_attr_weight_blend": 0.0,
            "edge_attr_weight_temp": 1.0,
            "edge_attr_weight_apply_to": "si_only",
            "edge_attr_hierarchical": True,
        },
        {
            "condition": "B16_PATHA_PATHB_v12_hier",
            "edge_variant": "V12",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": True,
            "edge_adaptive_alpha_strength": 2.0,
            "edge_adaptive_alpha_bias": 0.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_hidden_dim": 64,
            "edge_attr_fusion_scale": 0.7,
            "edge_attr_weight_blend": 0.5,
            "edge_attr_weight_temp": 1.0,
            "edge_attr_weight_apply_to": "si_only",
            "edge_attr_hierarchical": True,
        },
        {
            "condition": "B17_PATHA_on_V5",
            "edge_variant": "V5",
            "edge_hybrid_alpha": 0.7,
            "edge_feat_temp": 1.0,
            "edge_input_prior_alpha": 0.35,
            "edge_fusion_gamma": 1.0,
            "edge_fusion_gamma_start": 0.2,
            "edge_fusion_gamma_end": 1.2,
            "edge_fusion_gamma_sched_epochs": 100,
            "edge_adaptive_alpha": False,
            "edge_adaptive_alpha_strength": 1.0,
            "edge_adaptive_alpha_bias": 2.0,
            "edge_reliability_temp": 1.0,
            "edge_confidence_quantile": 0.0,
            "edge_attr_weight_blend": 0.5,
            "edge_attr_weight_temp": 1.0,
            "edge_attr_weight_apply_to": "si_only",
            "edge_attr_hierarchical": False,
        },
    ]


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def dataset_max_nums(name: str, default_max: int) -> int:
    return int(DATASET_MAX_NUMS.get(name.lower(), default_max))


def build_cmd(dataset: str, seed: int, version: str, cond: Dict[str, object], args) -> List[str]:
    max_nums = dataset_max_nums(dataset, args.default_max_nums)
    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset,
        "--root_path",
        args.root_path,
        "--version",
        version,
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--train_log_interval",
        str(args.train_log_interval),
        "--exp_iters",
        "1",
        "--n_cluster_trials",
        str(args.n_cluster_trials),
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--hid_dim",
        str(args.hid_dim),
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
        str(cond.get("edge_attr_hidden_dim", args.edge_attr_hidden_dim)),
        "--edge_attr_fusion_scale",
        str(cond.get("edge_attr_fusion_scale", args.edge_attr_fusion_scale)),
        "--edge_attr_weight_blend",
        str(cond.get("edge_attr_weight_blend", args.edge_attr_weight_blend)),
        "--edge_attr_weight_temp",
        str(cond.get("edge_attr_weight_temp", args.edge_attr_weight_temp)),
        "--edge_attr_weight_apply_to",
        str(cond.get("edge_attr_weight_apply_to", args.edge_attr_weight_apply_to)),
        "--patience",
        str(args.patience),
    ]
    if cond["edge_fusion_gamma_start"] is not None:
        cmd += ["--edge_fusion_gamma_start", str(cond["edge_fusion_gamma_start"])]
    if cond["edge_fusion_gamma_end"] is not None:
        cmd += ["--edge_fusion_gamma_end", str(cond["edge_fusion_gamma_end"])]
    if bool(cond["edge_adaptive_alpha"]):
        cmd += ["--edge_adaptive_alpha"]
    if bool(cond.get("edge_attr_hierarchical", args.edge_attr_hierarchical)):
        cmd += ["--edge_attr_hierarchical"]
    if args.amp_bf16:
        cmd += ["--amp_bf16"]
    return cmd


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(out_dir: Path, run_rows: List[Dict[str, object]]) -> None:
    runs = pd.DataFrame(run_rows)
    ok = runs[runs["status"].isin(["ok", "skip_exists"])].copy()
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

    base = by_cd[by_cd["condition"] == "B0_V1_baseline"][["dataset", "nmi_mean", "ari_mean"]].rename(
        columns={"nmi_mean": "base_nmi", "ari_mean": "base_ari"}
    )
    delta = by_cd.merge(base, on="dataset", how="left")
    delta["delta_nmi_vs_baseline"] = delta["nmi_mean"] - delta["base_nmi"]
    delta["delta_ari_vs_baseline"] = delta["ari_mean"] - delta["base_ari"]
    delta.to_csv(out_dir / "delta_vs_baseline.csv", index=False)

    by_cond = (
        delta.groupby("condition", as_index=False)
        .agg(
            datasets_covered=("dataset", "count"),
            mean_nmi=("nmi_mean", "mean"),
            mean_ari=("ari_mean", "mean"),
            mean_delta_nmi_vs_baseline=("delta_nmi_vs_baseline", "mean"),
            mean_delta_ari_vs_baseline=("delta_ari_vs_baseline", "mean"),
            win_rate_nmi_vs_baseline=("delta_nmi_vs_baseline", lambda s: float(np.mean(s > 0))),
            win_rate_ari_vs_baseline=("delta_ari_vs_baseline", lambda s: float(np.mean(s > 0))),
            mean_seconds=("seconds_mean", "mean"),
        )
        .sort_values(["mean_delta_nmi_vs_baseline", "mean_delta_ari_vs_baseline"], ascending=False)
    )
    by_cond["rank_score"] = (
        by_cond["mean_delta_nmi_vs_baseline"]
        + 0.25 * by_cond["win_rate_nmi_vs_baseline"]
        + 0.15 * by_cond["mean_delta_ari_vs_baseline"]
    )
    by_cond = by_cond.sort_values("rank_score", ascending=False).reset_index(drop=True)
    by_cond.to_csv(out_dir / "summary_by_condition.csv", index=False)

    best = (
        delta.sort_values(["dataset", "nmi_mean"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
    )
    best = best[
        [
            "dataset",
            "condition",
            "nmi_mean",
            "ari_mean",
            "delta_nmi_vs_baseline",
            "delta_ari_vs_baseline",
        ]
    ].rename(columns={"condition": "best_condition_by_nmi"})
    best.to_csv(out_dir / "best_condition_by_dataset.csv", index=False)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Branch comparison on classic non-urban datasets.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--conditions", type=str, default="",
                        help="Optional comma-separated condition names to run.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=180,
        help="Set equal to epochs to avoid label-based multi-epoch model selection bias.",
    )
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=3)
    parser.add_argument("--default_max_nums", type=int, default=10)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--knn_mode", type=str, default="auto", choices=["auto", "dense", "edge"])
    parser.add_argument("--knn_auto_threshold", type=int, default=20000)
    parser.add_argument("--edge_attr_hidden_dim", type=int, default=64)
    parser.add_argument("--edge_attr_fusion_scale", type=float, default=1.0)
    parser.add_argument("--edge_attr_weight_blend", type=float, default=0.0)
    parser.add_argument("--edge_attr_weight_temp", type=float, default=1.0)
    parser.add_argument("--edge_attr_weight_apply_to", type=str, default="si_only", choices=["si_only", "both"])
    parser.add_argument("--edge_attr_hierarchical", action="store_true")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience passed to main.py. Use 0 to disable early stopping for fair branch comparison.",
    )
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--tag", type=str, default="benchmark_branch_compare_v1")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    conds = condition_space()
    cond_filter = parse_condition_filter(args.conditions)
    if cond_filter:
        selected = {canonical_condition_name(x) for x in cond_filter}
        conds = [c for c in conds if str(c["condition"]) in selected]
        if not conds:
            raise ValueError(f"No valid conditions selected from: {cond_filter}")

    out_dir = repo / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    total = len(datasets) * len(conds) * len(seeds)
    i = 0
    for d in datasets:
        for cond in conds:
            c_name = str(cond["condition"])
            for s in seeds:
                i += 1
                version = f"{args.tag}_{c_name}_{d}_s{s}"
                run_dir = repo / "results" / version
                metrics_path = run_dir / f"{d}_metrics.json"
                if metrics_path.exists() and not args.force_rerun:
                    nmi = np.nan
                    ari = np.nan
                    try:
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        nmi = safe_float(m.get("nmi_mean"))
                        ari = safe_float(m.get("ari_mean"))
                    except Exception:
                        pass
                    row = {
                        "dataset": d,
                        "seed": s,
                        "condition": c_name,
                        "version": version,
                        "status": "skip_exists",
                        "seconds": 0.0,
                        "nmi": nmi,
                        "ari": ari,
                        "metrics_path": str(metrics_path),
                        "log_path": str(run_dir / f"{d}.log"),
                        "stderr_tail": "",
                    }
                    run_rows.append(row)
                    print(f"[{i}/{total}] {d} {c_name} s{s} -> skip_exists")
                    continue

                cmd = build_cmd(dataset=d, seed=s, version=version, cond=cond, args=args)
                t0 = time.time()
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                sec = time.time() - t0
                status = "ok" if proc.returncode == 0 else f"fail_{proc.returncode}"
                nmi = np.nan
                ari = np.nan
                stderr_tail = ""
                if proc.stderr:
                    lines = [x for x in proc.stderr.strip().splitlines() if x.strip()]
                    if lines:
                        stderr_tail = lines[-1]
                if metrics_path.exists():
                    try:
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        nmi = safe_float(m.get("nmi_mean"))
                        ari = safe_float(m.get("ari_mean"))
                    except Exception:
                        pass
                row = {
                    "dataset": d,
                    "seed": s,
                    "condition": c_name,
                    "version": version,
                    "status": status,
                    "seconds": sec,
                    "nmi": nmi,
                    "ari": ari,
                    "metrics_path": str(metrics_path),
                    "log_path": str(run_dir / f"{d}.log"),
                    "stderr_tail": stderr_tail,
                }
                run_rows.append(row)
                print(
                    f"[{i}/{total}] {d} {c_name} s{s} -> {status} "
                    f"sec={sec:.2f} nmi={nmi:.6g} ari={ari:.6g}"
                )

    runs_csv = out_dir / "runs.csv"
    write_csv(
        runs_csv,
        run_rows,
        [
            "dataset",
            "seed",
            "condition",
            "version",
            "status",
            "seconds",
            "nmi",
            "ari",
            "metrics_path",
            "log_path",
            "stderr_tail",
        ],
    )
    aggregate(out_dir=out_dir, run_rows=run_rows)

    runs_df = pd.DataFrame(run_rows)
    summary = {
        "tag": args.tag,
        "datasets": datasets,
        "seeds": seeds,
        "n_conditions": len(conds),
        "total_runs": int(len(run_rows)),
        "ok_runs": int((runs_df["status"] == "ok").sum()),
        "skip_exists_runs": int((runs_df["status"] == "skip_exists").sum()),
        "fail_runs": int((~runs_df["status"].isin(["ok", "skip_exists"])).sum()),
        "runs_csv": str(runs_csv),
        "summary_by_condition_dataset_csv": str(out_dir / "summary_by_condition_dataset.csv"),
        "delta_vs_baseline_csv": str(out_dir / "delta_vs_baseline.csv"),
        "summary_by_condition_csv": str(out_dir / "summary_by_condition.csv"),
        "best_condition_by_dataset_csv": str(out_dir / "best_condition_by_dataset.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {runs_csv}")
    print(f"[ok] wrote {out_dir / 'summary_by_condition_dataset.csv'}")
    print(f"[ok] wrote {out_dir / 'delta_vs_baseline.csv'}")
    print(f"[ok] wrote {out_dir / 'summary_by_condition.csv'}")
    print(f"[ok] wrote {out_dir / 'best_condition_by_dataset.csv'}")
    print(f"[ok] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
