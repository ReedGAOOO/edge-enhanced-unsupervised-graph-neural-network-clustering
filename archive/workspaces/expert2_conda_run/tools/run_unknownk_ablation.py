#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATASET_DEFAULT_MAX = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
}

PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "g15_echf_main": "configs/presets/g15_echf_main.json",
    "g17_v5_temp15": "configs/presets/g17_v5_temp15.json",
    "b15_echf_branch": "configs/presets/b15_echf_branch.json",
}

METRIC_COLS = [
    "acc_mean",
    "nmi_mean",
    "ari_mean",
    "si_loss_mean",
    "modularity_mean",
    "conductance_mean",
    "conductance_weighted_mean",
    "pred_n_clusters_mean",
    "pred_cluster_size_cv_mean",
    "best_epoch_mean",
    "best_train_loss_mean",
]


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_preset(repo: Path, name_or_path: str) -> Dict[str, object]:
    p = Path(name_or_path)
    if p.exists():
        target = p
    else:
        if name_or_path not in PRESET_MAP:
            raise FileNotFoundError(f"Unknown preset: {name_or_path}")
        target = repo / PRESET_MAP[name_or_path]
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cmd(
    python_bin: str,
    dataset: str,
    seed: int,
    version: str,
    preset: Dict[str, object],
    max_nums: int,
    eps_int: int,
    args,
) -> List[str]:
    cmd = [
        python_bin,
        "main.py",
        "--dataset",
        dataset,
        "--root_path",
        args.root_path,
        "--task",
        "Clustering",
        "--version",
        version,
        "--seed",
        str(seed),
        "--exp_iters",
        "1",
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.epochs),
        "--train_log_interval",
        str(args.train_log_interval),
        "--save_path",
        f"{version}.pt",
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
        "--epsInt",
        str(eps_int),
        "--patience",
        "0",
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
        "--edge_attr_weight_blend",
        str(preset.get("edge_attr_weight_blend", 0.0)),
        "--edge_attr_weight_temp",
        str(preset.get("edge_attr_weight_temp", 1.0)),
        "--edge_attr_weight_apply_to",
        str(preset.get("edge_attr_weight_apply_to", "si_only")),
    ]
    if bool(preset.get("edge_adaptive_alpha", False)):
        cmd.append("--edge_adaptive_alpha")
    if bool(preset.get("edge_attr_hierarchical", False)):
        cmd.append("--edge_attr_hierarchical")
    if bool(preset.get("append_generic_edge_attr", False)):
        cmd.append("--append_generic_edge_attr")
    if preset.get("edge_fusion_gamma_start", None) is not None:
        cmd += ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])]
    if preset.get("edge_fusion_gamma_end", None) is not None:
        cmd += ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])]
    if bool(args.amp_bf16):
        cmd.append("--amp_bf16")
    return cmd


def run_one(cmd: List[str], cwd: Path, log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(p.returncode)


def summarize(out_dir: Path, runs: pd.DataFrame) -> None:
    runs.to_csv(out_dir / "runs.csv", index=False)

    max_df = runs[runs["part"] == "max_sweep"].copy()
    eps_df = runs[runs["part"] == "eps_sweep"].copy()

    if not max_df.empty:
        g = (
            max_df.groupby(["condition", "dataset", "max_nums"], as_index=False)[METRIC_COLS + ["seconds"]]
            .mean(numeric_only=True)
        )
        g.to_csv(out_dir / "max_sweep_summary.csv", index=False)

        robust_rows = []
        for (cond, ds, seed), sub in max_df.groupby(["condition", "dataset", "seed"]):
            sub = sub.sort_values("max_nums")
            robust_rows.append(
                {
                    "condition": cond,
                    "dataset": ds,
                    "seed": seed,
                    "n_max_values": int(sub["max_nums"].nunique()),
                    "nmi_span": float(sub["nmi_mean"].max() - sub["nmi_mean"].min()),
                    "ari_span": float(sub["ari_mean"].max() - sub["ari_mean"].min()),
                    "pred_k_span": float(sub["pred_n_clusters_mean"].max() - sub["pred_n_clusters_mean"].min()),
                    "nmi_std_over_max": float(sub["nmi_mean"].std(ddof=0)),
                    "ari_std_over_max": float(sub["ari_mean"].std(ddof=0)),
                }
            )
        robust = pd.DataFrame(robust_rows)
        robust.to_csv(out_dir / "max_sweep_robustness_by_seed.csv", index=False)
        if not robust.empty:
            (
                robust.groupby(["condition", "dataset"], as_index=False)[
                    ["nmi_span", "ari_span", "pred_k_span", "nmi_std_over_max", "ari_std_over_max"]
                ]
                .mean()
                .to_csv(out_dir / "max_sweep_robustness_by_dataset.csv", index=False)
            )
            (
                robust.groupby(["condition"], as_index=False)[
                    ["nmi_span", "ari_span", "pred_k_span", "nmi_std_over_max", "ari_std_over_max"]
                ]
                .mean()
                .to_csv(out_dir / "max_sweep_robustness_overall.csv", index=False)
            )

        best = (
            g.sort_values(["condition", "dataset", "nmi_mean"], ascending=[True, True, False])
            .groupby(["condition", "dataset"], as_index=False)
            .first()
        )
        best.to_csv(out_dir / "max_sweep_best_by_dataset.csv", index=False)

    if not eps_df.empty:
        g = (
            eps_df.groupby(["condition", "dataset", "epsInt"], as_index=False)[METRIC_COLS + ["seconds"]]
            .mean(numeric_only=True)
        )
        g.to_csv(out_dir / "eps_sweep_summary.csv", index=False)

        ref = (
            eps_df[eps_df["epsInt"] == -1][["condition", "dataset", "seed", "nmi_mean", "ari_mean", "acc_mean", "pred_n_clusters_mean"]]
            .rename(
                columns={
                    "nmi_mean": "nmi_ref_nofix",
                    "ari_mean": "ari_ref_nofix",
                    "acc_mean": "acc_ref_nofix",
                    "pred_n_clusters_mean": "pred_k_ref_nofix",
                }
            )
        )
        delta = eps_df.merge(ref, on=["condition", "dataset", "seed"], how="left")
        delta["delta_nmi_vs_nofix"] = delta["nmi_mean"] - delta["nmi_ref_nofix"]
        delta["delta_ari_vs_nofix"] = delta["ari_mean"] - delta["ari_ref_nofix"]
        delta["delta_acc_vs_nofix"] = delta["acc_mean"] - delta["acc_ref_nofix"]
        delta["delta_pred_k_vs_nofix"] = delta["pred_n_clusters_mean"] - delta["pred_k_ref_nofix"]
        delta.to_csv(out_dir / "eps_sweep_delta_vs_nofix.csv", index=False)

        (
            delta.groupby(["condition", "dataset", "epsInt"], as_index=False)[
                ["delta_nmi_vs_nofix", "delta_ari_vs_nofix", "delta_acc_vs_nofix", "delta_pred_k_vs_nofix"]
            ]
            .mean(numeric_only=True)
            .to_csv(out_dir / "eps_sweep_delta_by_dataset.csv", index=False)
        )
        (
            delta.groupby(["condition", "epsInt"], as_index=False)[
                ["delta_nmi_vs_nofix", "delta_ari_vs_nofix", "delta_acc_vs_nofix", "delta_pred_k_vs_nofix"]
            ]
            .mean(numeric_only=True)
            .to_csv(out_dir / "eps_sweep_delta_overall.csv", index=False)
        )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Unknown-K ablation for max_nums and epsInt.")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed,computers,photo")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--conditions", type=str, default="baseline_v1,g15_echf_main")
    parser.add_argument("--max_grid", type=str, default="6,10,16,24")
    parser.add_argument("--eps_grid", type=str, default="-1,0,2,5,8,12")
    parser.add_argument("--fixed_eps_for_max_sweep", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=3)
    parser.add_argument("--train_log_interval", type=int, default=100)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--knn_mode", type=str, default="auto", choices=["auto", "dense", "edge"])
    parser.add_argument("--knn_auto_threshold", type=int, default=20000)
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--tag", type=str, default="benchmark_unknownk_ablation_v1")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    conditions = parse_list(args.conditions)
    max_grid = parse_int_list(args.max_grid)
    eps_grid = parse_int_list(args.eps_grid)

    for d in datasets:
        if d.lower() not in DATASET_DEFAULT_MAX:
            raise ValueError(f"Dataset '{d}' missing from DATASET_DEFAULT_MAX.")

    out_dir = repo / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    total = (
        len(conditions) * len(datasets) * len(seeds) * len(max_grid)
        + len(conditions) * len(datasets) * len(seeds) * len(eps_grid)
    )
    idx = 0

    # Part A: max_nums robustness sweep.
    for cond in conditions:
        preset = load_preset(repo, cond)
        for dataset in datasets:
            for seed in seeds:
                for max_nums in max_grid:
                    idx += 1
                    eps_int = int(args.fixed_eps_for_max_sweep)
                    version = f"{args.tag}_max_{cond}_{dataset}_s{seed}_k{max_nums}_eps{eps_int}"
                    run_dir = repo / "results" / version
                    metrics_path = run_dir / f"{dataset}_metrics.json"
                    runner_log = run_dir / "runner.log"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    row = {
                        "part": "max_sweep",
                        "condition": cond,
                        "dataset": dataset,
                        "seed": int(seed),
                        "max_nums": int(max_nums),
                        "epsInt": int(eps_int),
                        "version": version,
                        "metrics_path": str(metrics_path),
                    }
                    if metrics_path.exists() and not args.force_rerun:
                        row["status"] = "skip_exists"
                        row["seconds"] = 0.0
                    else:
                        cmd = build_cmd(
                            python_bin=args.python_bin,
                            dataset=dataset,
                            seed=seed,
                            version=version,
                            preset=preset,
                            max_nums=max_nums,
                            eps_int=eps_int,
                            args=args,
                        )
                        t0 = time.time()
                        code = run_one(cmd, cwd=repo, log_path=runner_log)
                        row["seconds"] = time.time() - t0
                        row["status"] = "ok" if code == 0 else "fail"
                        row["returncode"] = int(code)
                    if metrics_path.exists():
                        try:
                            m = json.loads(metrics_path.read_text(encoding="utf-8"))
                            for k in METRIC_COLS:
                                row[k] = safe_float(m.get(k))
                            row["selection_rule"] = str(m.get("selection_rule", ""))
                        except Exception as e:
                            row["metrics_error"] = str(e)
                    print(
                        f"[{idx}/{total}] max_sweep {cond} {dataset} s{seed} k{max_nums} eps{eps_int} -> "
                        f"{row.get('status', 'na')}"
                    )
                    run_rows.append(row)

    # Part B: epsInt bias sweep.
    for cond in conditions:
        preset = load_preset(repo, cond)
        for dataset in datasets:
            default_k = int(DATASET_DEFAULT_MAX[dataset.lower()])
            for seed in seeds:
                for eps_int in eps_grid:
                    idx += 1
                    eps_tag = f"m{abs(eps_int)}" if eps_int < 0 else str(eps_int)
                    version = f"{args.tag}_eps_{cond}_{dataset}_s{seed}_k{default_k}_eps{eps_tag}"
                    run_dir = repo / "results" / version
                    metrics_path = run_dir / f"{dataset}_metrics.json"
                    runner_log = run_dir / "runner.log"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    row = {
                        "part": "eps_sweep",
                        "condition": cond,
                        "dataset": dataset,
                        "seed": int(seed),
                        "max_nums": int(default_k),
                        "epsInt": int(eps_int),
                        "version": version,
                        "metrics_path": str(metrics_path),
                    }
                    if metrics_path.exists() and not args.force_rerun:
                        row["status"] = "skip_exists"
                        row["seconds"] = 0.0
                    else:
                        cmd = build_cmd(
                            python_bin=args.python_bin,
                            dataset=dataset,
                            seed=seed,
                            version=version,
                            preset=preset,
                            max_nums=default_k,
                            eps_int=eps_int,
                            args=args,
                        )
                        t0 = time.time()
                        code = run_one(cmd, cwd=repo, log_path=runner_log)
                        row["seconds"] = time.time() - t0
                        row["status"] = "ok" if code == 0 else "fail"
                        row["returncode"] = int(code)
                    if metrics_path.exists():
                        try:
                            m = json.loads(metrics_path.read_text(encoding="utf-8"))
                            for k in METRIC_COLS:
                                row[k] = safe_float(m.get(k))
                            row["selection_rule"] = str(m.get("selection_rule", ""))
                        except Exception as e:
                            row["metrics_error"] = str(e)
                    print(
                        f"[{idx}/{total}] eps_sweep {cond} {dataset} s{seed} k{default_k} eps{eps_int} -> "
                        f"{row.get('status', 'na')}"
                    )
                    run_rows.append(row)

    runs = pd.DataFrame(run_rows)
    summarize(out_dir=out_dir, runs=runs)

    ok = int((runs["status"] == "ok").sum()) if "status" in runs.columns else 0
    skip = int((runs["status"] == "skip_exists").sum()) if "status" in runs.columns else 0
    fail = int((runs["status"] == "fail").sum()) if "status" in runs.columns else 0
    decision = {
        "tag": args.tag,
        "total_runs": int(len(runs)),
        "ok": ok,
        "skip_exists": skip,
        "fail": fail,
        "datasets": datasets,
        "conditions": conditions,
        "max_grid": max_grid,
        "eps_grid": eps_grid,
        "fixed_eps_for_max_sweep": int(args.fixed_eps_for_max_sweep),
        "epochs": int(args.epochs),
        "notes": [
            "epsInt=-1 can be treated as no-fix baseline because all clusters pass corr_idx in fix_cluster_results.",
            "selection_rule is min_train_loss (from metrics.json).",
        ],
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

