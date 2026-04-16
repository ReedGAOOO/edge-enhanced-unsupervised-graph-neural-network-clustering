#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "g15_echf_main": "configs/presets/g15_echf_main.json",
    "g17_v5_temp15": "configs/presets/g17_v5_temp15.json",
}

DATASET_MAX_NUMS = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
    "urban_bangkok_plot": 64,
    "urban_beijing_plot": 64,
    "urban_boston_plot": 64,
    "urban_chicago_plot": 64,
    "urban_johannesburg_plot": 64,
    "urban_madrid_plot": 64,
    "urban_melbourne_plot": 64,
    "urban_paris_plot": 64,
    "urban_shanghai_plot": 64,
    "urban_singapore_plot": 64,
    "urban_sydney_plot": 64,
    "urban_tokyo_plot": 64,
    "urban_washingtondc_plot": 64,
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
    knn: int,
    knn_mode: str,
    knn_auto_threshold: int,
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
        str(knn),
        "--knn_mode",
        str(knn_mode),
        "--knn_auto_threshold",
        str(knn_auto_threshold),
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
    if bool(args.known_only_eval):
        cmd.append("--known_only_eval")
    return cmd


def run_one(cmd: List[str], cwd: Path, log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(p.returncode)


def make_figures(out_dir: Path, runs_ok: pd.DataFrame, delta: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: runtime efficiency curve by k.
    eff = (
        runs_ok.groupby(["knn_mode", "knn"], as_index=False)[["seconds", "nmi_mean", "ari_mean"]]
        .mean(numeric_only=True)
        .sort_values(["knn_mode", "knn"])
    )
    plt.figure(figsize=(8, 5))
    for mode, g in eff.groupby("knn_mode"):
        plt.plot(g["knn"], g["seconds"], marker="o", label=mode)
    plt.xlabel("k (KNN neighbors)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Efficiency Curve: Runtime vs k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_efficiency_runtime_vs_k.png", dpi=180)
    plt.close()

    # Figure 2: error curve relative to dense mode.
    if not delta.empty:
        err = (
            delta.groupby(["knn_mode", "knn"], as_index=False)[["delta_nmi_vs_dense", "delta_ari_vs_dense"]]
            .mean(numeric_only=True)
            .sort_values(["knn_mode", "knn"])
        )
        plt.figure(figsize=(8, 5))
        for mode, g in err.groupby("knn_mode"):
            if mode == "dense":
                continue
            plt.plot(g["knn"], g["delta_nmi_vs_dense"], marker="o", label=f"{mode} ΔNMI")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("k (KNN neighbors)")
        plt.ylabel("ΔNMI vs dense")
        plt.title("Error Curve: Accuracy Gap vs Dense KNN")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig2_error_delta_nmi_vs_k.png", dpi=180)
        plt.close()

        # Figure 3: speedup-accuracy tradeoff scatter.
        plt.figure(figsize=(8, 5))
        for mode in sorted(delta["knn_mode"].unique()):
            if mode == "dense":
                continue
            g = delta[delta["knn_mode"] == mode]
            plt.scatter(g["speedup_x"], g["delta_nmi_vs_dense"], s=30, alpha=0.8, label=mode)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.axvline(1.0, color="gray", linewidth=1, linestyle="--")
        plt.xscale("log")
        plt.xlabel("Speedup x (dense seconds / mode seconds)")
        plt.ylabel("ΔNMI vs dense")
        plt.title("Tradeoff Scatter: Speedup vs Accuracy Gap")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig3_tradeoff_speedup_vs_delta_nmi.png", dpi=180)
        plt.close()


def summarize(out_dir: Path, runs: pd.DataFrame) -> None:
    runs.to_csv(out_dir / "runs.csv", index=False)
    runs_ok = runs[runs["status"].isin(["ok", "skip_exists"])].copy()
    runs_ok.to_csv(out_dir / "runs_ok.csv", index=False)

    if runs_ok.empty:
        return

    summary = (
        runs_ok.groupby(["dataset", "knn_mode", "knn"], as_index=False)[METRIC_COLS + ["seconds"]]
        .mean(numeric_only=True)
    )
    summary.to_csv(out_dir / "summary_by_dataset_mode_k.csv", index=False)

    mode_summary = (
        runs_ok.groupby(["knn_mode", "knn"], as_index=False)[METRIC_COLS + ["seconds"]]
        .mean(numeric_only=True)
        .sort_values(["knn_mode", "knn"])
    )
    mode_summary.to_csv(out_dir / "summary_by_mode_k.csv", index=False)

    # Delta vs dense for overlapping successful runs.
    dense = runs_ok[runs_ok["knn_mode"] == "dense"][
        ["dataset", "seed", "knn", "nmi_mean", "ari_mean", "acc_mean", "seconds"]
    ].rename(
        columns={
            "nmi_mean": "nmi_dense",
            "ari_mean": "ari_dense",
            "acc_mean": "acc_dense",
            "seconds": "seconds_dense",
        }
    )
    delta = runs_ok.merge(dense, on=["dataset", "seed", "knn"], how="left")
    delta["delta_nmi_vs_dense"] = delta["nmi_mean"] - delta["nmi_dense"]
    delta["delta_ari_vs_dense"] = delta["ari_mean"] - delta["ari_dense"]
    delta["delta_acc_vs_dense"] = delta["acc_mean"] - delta["acc_dense"]
    delta["speedup_x"] = delta["seconds_dense"] / delta["seconds"]
    delta.to_csv(out_dir / "delta_vs_dense_runs.csv", index=False)

    delta_mode = (
        delta.groupby(["knn_mode", "knn"], as_index=False)[
            ["delta_nmi_vs_dense", "delta_ari_vs_dense", "delta_acc_vs_dense", "speedup_x"]
        ]
        .mean(numeric_only=True)
        .sort_values(["knn_mode", "knn"])
    )
    delta_mode.to_csv(out_dir / "delta_vs_dense_by_mode_k.csv", index=False)

    delta_ds = (
        delta.groupby(["dataset", "knn_mode", "knn"], as_index=False)[
            ["delta_nmi_vs_dense", "delta_ari_vs_dense", "delta_acc_vs_dense", "speedup_x"]
        ]
        .mean(numeric_only=True)
    )
    delta_ds.to_csv(out_dir / "delta_vs_dense_by_dataset_mode_k.csv", index=False)

    make_figures(out_dir=out_dir, runs_ok=runs_ok, delta=delta)

    lines = []
    lines.append("# KNN Mode Efficiency-Accuracy Tradeoff")
    lines.append("")
    lines.append("## Run Scope")
    lines.append(f"- total runs: {len(runs)}")
    lines.append(f"- successful/loaded runs: {len(runs_ok)}")
    lines.append("")
    lines.append("## Overall (mean by mode-k)")
    for _, r in mode_summary.iterrows():
        lines.append(
            f"- mode={r['knn_mode']}, k={int(r['knn'])}: "
            f"NMI={r['nmi_mean']:.4f}, ARI={r['ari_mean']:.4f}, seconds={r['seconds']:.2f}"
        )
    lines.append("")
    lines.append("## Relative to dense (mean by mode-k)")
    for _, r in delta_mode.iterrows():
        lines.append(
            f"- mode={r['knn_mode']}, k={int(r['knn'])}: "
            f"ΔNMI={safe_float(r['delta_nmi_vs_dense']):+.4f}, "
            f"ΔARI={safe_float(r['delta_ari_vs_dense']):+.4f}, "
            f"speedup={safe_float(r['speedup_x']):.2f}x"
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Dense vs edge KNN efficiency-accuracy tradeoff benchmark.")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed,computers,photo")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--knn_list", type=str, default="4,8,16")
    parser.add_argument("--knn_modes", type=str, default="dense,edge,auto")
    parser.add_argument("--preset", type=str, default="g15_echf_main")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--train_log_interval", type=int, default=40)
    parser.add_argument("--knn_auto_threshold", type=int, default=15000)
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--known_only_eval", action="store_true")
    parser.add_argument("--tag", type=str, default="benchmark_knn_mode_tradeoff_v1")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    knn_list = parse_int_list(args.knn_list)
    knn_modes = parse_list(args.knn_modes)
    for d in datasets:
        if d.lower() not in DATASET_MAX_NUMS:
            raise ValueError(f"Dataset '{d}' missing in DATASET_MAX_NUMS.")

    preset = load_preset(repo, args.preset)

    out_dir = repo / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    total = len(datasets) * len(seeds) * len(knn_list) * len(knn_modes)
    idx = 0
    for dataset in datasets:
        max_nums = int(DATASET_MAX_NUMS[dataset.lower()])
        for seed in seeds:
            for knn in knn_list:
                for mode in knn_modes:
                    idx += 1
                    version = f"{args.tag}_{dataset}_s{seed}_{mode}_k{knn}"
                    run_dir = repo / "results" / version
                    metrics_path = run_dir / f"{dataset}_metrics.json"
                    log_path = run_dir / "runner.log"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    row: Dict[str, object] = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "knn": int(knn),
                        "knn_mode": mode,
                        "max_nums": max_nums,
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
                            knn=knn,
                            knn_mode=mode,
                            knn_auto_threshold=int(args.knn_auto_threshold),
                            args=args,
                        )
                        t0 = time.time()
                        rc = run_one(cmd=cmd, cwd=repo, log_path=log_path)
                        row["seconds"] = float(time.time() - t0)
                        row["status"] = "ok" if rc == 0 else "fail"
                        row["returncode"] = int(rc)
                    if metrics_path.exists():
                        try:
                            m = json.loads(metrics_path.read_text(encoding="utf-8"))
                            for k in METRIC_COLS:
                                row[k] = safe_float(m.get(k))
                            row["selection_rule"] = str(m.get("selection_rule", ""))
                            row["reported_knn_mode"] = str(m.get("knn_mode", ""))
                        except Exception as e:
                            row["metrics_error"] = str(e)

                    print(
                        f"[{idx}/{total}] {dataset} s{seed} mode={mode} k={knn} -> {row.get('status', 'na')}"
                    )
                    runs.append(row)

    runs_df = pd.DataFrame(runs)
    summarize(out_dir=out_dir, runs=runs_df)

    decision = {
        "tag": args.tag,
        "total_runs": int(len(runs_df)),
        "ok": int((runs_df["status"] == "ok").sum()) if "status" in runs_df.columns else 0,
        "skip_exists": int((runs_df["status"] == "skip_exists").sum()) if "status" in runs_df.columns else 0,
        "fail": int((runs_df["status"] == "fail").sum()) if "status" in runs_df.columns else 0,
        "datasets": datasets,
        "seeds": seeds,
        "knn_list": knn_list,
        "knn_modes": knn_modes,
        "preset": args.preset,
        "epochs": int(args.epochs),
        "hid_dim": int(args.hid_dim),
        "n_cluster_trials": int(args.n_cluster_trials),
        "knn_auto_threshold": int(args.knn_auto_threshold),
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

