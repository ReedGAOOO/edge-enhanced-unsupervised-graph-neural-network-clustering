#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASET_MAX_NUMS = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
}

PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "g15_echf_main": "configs/presets/g15_echf_main.json",
    "g20_se_consistent_main": "configs/presets/g20_se_consistent_main.json",
    "b30_dualscalar": "configs/presets/b30_dualscalar.json",
    "b31_dualscalar_assign": "configs/presets/b31_dualscalar_assign.json",
    "b32_dualscalar_assign_hier": "configs/presets/b32_dualscalar_assign_hier.json",
    "b33_dualscalar_assign_hier_aug": "configs/presets/b33_dualscalar_assign_hier_aug.json",
}


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_preset(repo_root: Path, name_or_path: str) -> Dict[str, object]:
    p = Path(name_or_path)
    if p.exists():
        target = p
    else:
        if name_or_path not in PRESET_MAP:
            raise FileNotFoundError(f"Unknown preset '{name_or_path}'.")
        target = repo_root / PRESET_MAP[name_or_path]
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cmd(repo_root: Path, dataset: str, seed: int, version: str, preset: Dict[str, object], args) -> List[str]:
    return [
        sys.executable,
        "main.py",
        "--dataset", dataset,
        "--root_path", args.root_path,
        "--version", version,
        "--seed", str(seed),
        "--gpu", str(args.gpu),
        "--epochs", str(args.epochs),
        "--eval_freq", str(args.eval_freq),
        "--train_log_interval", str(args.train_log_interval),
        "--hid_dim", str(args.hid_dim),
        "--n_cluster_trials", str(args.n_cluster_trials),
        "--exp_iters", "1",
        "--patience", "0",
        "--max_nums", str(DATASET_MAX_NUMS[dataset.lower()]),
        "--edge_variant", str(preset.get("edge_variant", "V1")),
        "--edge_hybrid_alpha", str(preset.get("edge_hybrid_alpha", 0.5)),
        "--edge_feat_temp", str(preset.get("edge_feat_temp", 1.0)),
        "--edge_input_prior_alpha", str(preset.get("edge_input_prior_alpha", 0.0)),
        "--edge_fusion_gamma", str(preset.get("edge_fusion_gamma", 1.0)),
        "--edge_fusion_gamma_sched_epochs", str(preset.get("edge_fusion_gamma_sched_epochs", 0)),
        "--edge_confidence_quantile", str(preset.get("edge_confidence_quantile", 0.0)),
        "--edge_adaptive_alpha_strength", str(preset.get("edge_adaptive_alpha_strength", 2.0)),
        "--edge_adaptive_alpha_bias", str(preset.get("edge_adaptive_alpha_bias", 0.0)),
        "--edge_reliability_temp", str(preset.get("edge_reliability_temp", 1.0)),
        "--edge_attr_hidden_dim", str(preset.get("edge_attr_hidden_dim", 64)),
        "--edge_attr_fusion_scale", str(preset.get("edge_attr_fusion_scale", 1.0)),
        "--edge_attr_pool_topk", str(preset.get("edge_attr_pool_topk", 1)),
        "--edge_weight_learn_reg_lambda", str(preset.get("edge_weight_learn_reg_lambda", 0.02)),
        "--edge_weight_learn_logclip", str(preset.get("edge_weight_learn_logclip", 0.8)),
        "--edge_weight_learn_temp", str(preset.get("edge_weight_learn_temp", 1.0)),
        "--edge_weight_learn_apply_to", str(preset.get("edge_weight_learn_apply_to", "both")),
        "--edge_aug_prior_scale", str(preset.get("edge_aug_prior_scale", 0.0)),
        "--edge_attr_weight_blend", str(preset.get("edge_attr_weight_blend", 0.0)),
        "--edge_attr_weight_temp", str(preset.get("edge_attr_weight_temp", 1.0)),
        "--edge_attr_weight_apply_to", str(preset.get("edge_attr_weight_apply_to", "si_only")),
    ] + (
        ["--edge_adaptive_alpha"] if bool(preset.get("edge_adaptive_alpha", False)) else []
    ) + (
        ["--edge_attr_hierarchical"] if bool(preset.get("edge_attr_hierarchical", False)) else []
    ) + (
        ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])] if preset.get("edge_fusion_gamma_start", None) is not None else []
    ) + (
        ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])] if preset.get("edge_fusion_gamma_end", None) is not None else []
    )


def run_one(cmd: List[str], cwd: Path, log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact public benchmark screen for B30 family")
    parser.add_argument("--tag", type=str, default="benchmark_b30_public_screen_v1")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed,computers,photo")
    parser.add_argument("--conditions", type=str, default="baseline_v1,g15_echf_main,g20_se_consistent_main,b30_dualscalar,b31_dualscalar_assign,b32_dualscalar_assign_hier,b33_dualscalar_assign_hier_aug")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_freq", type=int, default=120)
    parser.add_argument("--train_log_interval", type=int, default=30)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / args.tag
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    conditions = parse_list(args.conditions)
    presets = {cond: load_preset(repo_root, cond) for cond in conditions}

    rows = []
    total = len(datasets) * len(seeds) * len(conditions)
    done = 0
    for dataset in datasets:
        for seed in seeds:
            for cond in conditions:
                done += 1
                version = f"{args.tag}_{cond}_{dataset}_s{seed}"
                log_path = logs_dir / f"{version}.log"
                metrics_path = repo_root / "results" / version / f"{dataset}_metrics.json"
                cmd = build_cmd(repo_root, dataset, seed, version, presets[cond], args)
                print(f"[{done}/{total}] {dataset} | {cond} | seed={seed}")
                t0 = time.time()
                if args.dry_run:
                    print("CMD:", " ".join(cmd))
                    rc = 0
                elif args.resume and metrics_path.exists():
                    rc = 0
                else:
                    rc = run_one(cmd, repo_root, log_path)
                sec = time.time() - t0
                metrics = {}
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    except Exception:
                        metrics = {}
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "condition": cond,
                    "rc": rc,
                    "runtime_sec": sec,
                    "nmi_mean": float(metrics.get("nmi_mean", float("nan"))),
                    "ari_mean": float(metrics.get("ari_mean", float("nan"))),
                    "acc_mean": float(metrics.get("acc_mean", float("nan"))),
                    "si_loss_mean": float(metrics.get("si_loss_mean", float("nan"))),
                    "modularity_mean": float(metrics.get("modularity_mean", float("nan"))),
                    "conductance_weighted_mean": float(metrics.get("conductance_weighted_mean", float("nan"))),
                    "final_edge_factor_msg_mean_mean": float(metrics.get("final_edge_factor_msg_mean", float("nan"))),
                    "final_edge_factor_si_mean_mean": float(metrics.get("final_edge_factor_si_mean", float("nan"))),
                    "final_edge_aug_bias_mean_mean": float(metrics.get("final_edge_aug_bias_mean", float("nan"))),
                    "metrics_path": str(metrics_path),
                    "log_path": str(log_path),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)
    ok = df[df["rc"] == 0].copy()
    if ok.empty:
        return
    summary = (
        ok.groupby("condition", as_index=False)
        .agg(
            runs=("dataset", "count"),
            nmi_mean=("nmi_mean", "mean"),
            ari_mean=("ari_mean", "mean"),
            acc_mean=("acc_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            modularity_mean=("modularity_mean", "mean"),
            conductance_w_mean=("conductance_weighted_mean", "mean"),
            edge_factor_msg_mean=("final_edge_factor_msg_mean_mean", "mean"),
            edge_factor_si_mean=("final_edge_factor_si_mean_mean", "mean"),
            edge_aug_bias_mean=("final_edge_aug_bias_mean_mean", "mean"),
        )
        .sort_values(["nmi_mean", "ari_mean"], ascending=False)
    )
    summary.to_csv(out_dir / "summary_by_condition.csv", index=False)

    by_cd = (
        ok.groupby(["condition", "dataset"], as_index=False)
        .agg(
            nmi_mean=("nmi_mean", "mean"),
            ari_mean=("ari_mean", "mean"),
            acc_mean=("acc_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            modularity_mean=("modularity_mean", "mean"),
            conductance_w_mean=("conductance_weighted_mean", "mean"),
        )
        .sort_values(["dataset", "nmi_mean"], ascending=[True, False])
    )
    by_cd.to_csv(out_dir / "summary_by_condition_dataset.csv", index=False)

    baseline = ok[ok["condition"] == "baseline_v1"][["dataset", "seed", "nmi_mean", "ari_mean", "acc_mean", "si_loss_mean"]]
    if not baseline.empty:
        merged = ok.merge(
            baseline,
            on=["dataset", "seed"],
            how="left",
            suffixes=("", "_baseline"),
        )
        merged["delta_nmi"] = merged["nmi_mean"] - merged["nmi_mean_baseline"]
        merged["delta_ari"] = merged["ari_mean"] - merged["ari_mean_baseline"]
        merged["delta_acc"] = merged["acc_mean"] - merged["acc_mean_baseline"]
        merged["delta_si_loss"] = merged["si_loss_mean"] - merged["si_loss_mean_baseline"]
        merged.to_csv(out_dir / "delta_vs_baseline.csv", index=False)

    print(f"[ok] wrote {out_dir / 'summary_by_condition.csv'}")


if __name__ == "__main__":
    main()
