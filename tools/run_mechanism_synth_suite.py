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
    "b31_dualscalar_assign": "configs/presets/b31_dualscalar_assign.json",
    "g20_se_consistent_main": "configs/presets/g20_se_consistent_main.json",
    "b40_v31_msgcond": "configs/presets/b40_v31_msgcond.json",
    "b45_v31_msgcond_gs050": "configs/presets/b45_v31_msgcond_gs050.json",
    "b47_v31_msgcond_gs050_matchonly": "configs/presets/b47_v31_msgcond_gs050_matchonly.json",
    "b48_v31_msgcond_gs050_confgate": "configs/presets/b48_v31_msgcond_gs050_confgate.json",
}


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


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


def discover_datasets(root_path: Path, prefix: str) -> List[str]:
    out = []
    if not root_path.exists():
        return out
    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        meta = d / f"{d.name}_meta.json"
        feat = d / f"{d.name}_feat.npy"
        edge_idx = d / f"{d.name}_edge_index.npy"
        label = d / f"{d.name}_label.npy"
        if meta.exists() and feat.exists() and edge_idx.exists() and label.exists():
            out.append(d.name)
    return out


def read_dataset_meta(root_path: Path, dataset: str) -> Dict[str, object]:
    meta_path = root_path / dataset / f"{dataset}_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_cmd(
    python_bin: str,
    dataset: str,
    seed: int,
    version: str,
    preset: Dict[str, object],
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
        str(args.eval_freq),
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
        str(args.max_nums),
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
        "--edge_attr_pool_topk",
        str(preset.get("edge_attr_pool_topk", 1)),
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


def aggregate(out_dir: Path, rows: List[Dict[str, object]], baseline_condition: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)
    if df.empty:
        return

    ok = df[df["rc"] == 0].copy()
    if ok.empty:
        return

    summary = (
        ok.groupby("condition", as_index=False)
        .agg(
            runs=("dataset", "count"),
            nmi_mean=("nmi_mean", "mean"),
            nmi_std=("nmi_mean", "std"),
            ari_mean=("ari_mean", "mean"),
            ari_std=("ari_mean", "std"),
            acc_mean=("acc_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            modularity_mean=("modularity_mean", "mean"),
            conductance_w_mean=("conductance_weighted_mean", "mean"),
            graph_alpha_mean=("final_graph_alpha_mean", "mean"),
            edge_rel_mean=("final_edge_reliability_mean", "mean"),
            edge_mix_mean=("final_edge_mix_beta_mean", "mean"),
            edge_factor_mean=("final_edge_factor_mean_mean", "mean"),
            edge_reg_mean=("final_edge_reg_mean", "mean"),
        )
        .sort_values(["nmi_mean", "ari_mean"], ascending=False)
    )
    summary.to_csv(out_dir / "summary_by_condition.csv", index=False)

    if baseline_condition in set(ok["condition"].unique()):
        base = ok[ok["condition"] == baseline_condition][
            ["dataset", "seed", "nmi_mean", "ari_mean", "acc_mean", "si_loss_mean"]
        ].rename(
            columns={
                "nmi_mean": "base_nmi",
                "ari_mean": "base_ari",
                "acc_mean": "base_acc",
                "si_loss_mean": "base_si_loss",
            }
        )
        delta = ok.merge(base, on=["dataset", "seed"], how="left")
        delta["delta_nmi_vs_baseline"] = delta["nmi_mean"] - delta["base_nmi"]
        delta["delta_ari_vs_baseline"] = delta["ari_mean"] - delta["base_ari"]
        delta["delta_acc_vs_baseline"] = delta["acc_mean"] - delta["base_acc"]
        delta["delta_si_loss_vs_baseline"] = delta["si_loss_mean"] - delta["base_si_loss"]
    else:
        delta = ok.copy()
        delta["delta_nmi_vs_baseline"] = np.nan
        delta["delta_ari_vs_baseline"] = np.nan
        delta["delta_acc_vs_baseline"] = np.nan
        delta["delta_si_loss_vs_baseline"] = np.nan

    delta.to_csv(out_dir / "delta_vs_baseline_runs.csv", index=False)

    by_ds_cond = (
        delta.groupby(["dataset", "condition"], as_index=False)
        .agg(
            homophily_target=("homophily_target", "mean"),
            edge_signal_target=("edge_signal_target", "mean"),
            homophily_observed=("homophily_observed", "mean"),
            edge_attr_signal_corr=("edge_attr_signal_corr", "mean"),
            nmi_mean=("nmi_mean", "mean"),
            ari_mean=("ari_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            delta_nmi_vs_baseline=("delta_nmi_vs_baseline", "mean"),
            delta_ari_vs_baseline=("delta_ari_vs_baseline", "mean"),
            delta_si_loss_vs_baseline=("delta_si_loss_vs_baseline", "mean"),
        )
        .sort_values(["dataset", "condition"])
    )
    by_ds_cond.to_csv(out_dir / "delta_vs_baseline_by_dataset_condition.csv", index=False)

    by_regime = (
        delta.groupby(["condition", "homophily_target", "edge_signal_target"], as_index=False)
        .agg(
            nmi_mean=("nmi_mean", "mean"),
            ari_mean=("ari_mean", "mean"),
            si_loss_mean=("si_loss_mean", "mean"),
            delta_nmi_vs_baseline=("delta_nmi_vs_baseline", "mean"),
            delta_ari_vs_baseline=("delta_ari_vs_baseline", "mean"),
            graph_alpha_mean=("final_graph_alpha_mean", "mean"),
            edge_factor_mean=("final_edge_factor_mean_mean", "mean"),
            edge_reg_mean=("final_edge_reg_mean", "mean"),
        )
        .sort_values(["condition", "homophily_target", "edge_signal_target"])
    )
    by_regime.to_csv(out_dir / "regime_summary.csv", index=False)

    phase = by_ds_cond.copy()
    phase["heterophily_observed"] = 1.0 - phase["homophily_observed"]
    phase.to_csv(out_dir / "phase_diagram_points.csv", index=False)

    make_figures(out_dir=out_dir, phase=phase, by_regime=by_regime, baseline_condition=baseline_condition)
    write_report(out_dir=out_dir, summary=summary, by_regime=by_regime, baseline_condition=baseline_condition)


def make_figures(out_dir: Path, phase: pd.DataFrame, by_regime: pd.DataFrame, baseline_condition: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    non_base = [c for c in sorted(phase["condition"].unique()) if c != baseline_condition]
    if non_base:
        fig, axes = plt.subplots(1, len(non_base), figsize=(6 * len(non_base), 5), squeeze=False)
        for i, cond in enumerate(non_base):
            ax = axes[0][i]
            p = phase[phase["condition"] == cond].copy()
            sc = ax.scatter(
                p["edge_attr_signal_corr"],
                p["heterophily_observed"],
                c=p["delta_nmi_vs_baseline"],
                cmap="coolwarm",
                s=80,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.4,
            )
            ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{cond}: phase map")
            ax.set_xlabel("edge_attr_signal_corr")
            ax.set_ylabel("heterophily_observed (1-h)")
            ax.grid(alpha=0.25)
            fig.colorbar(sc, ax=ax, shrink=0.85, label="delta_nmi_vs_baseline")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_phase_scatter_delta_nmi.png", dpi=180)
        plt.close(fig)

    if non_base:
        fig, axes = plt.subplots(1, len(non_base), figsize=(6 * len(non_base), 5), squeeze=False)
        for i, cond in enumerate(non_base):
            ax = axes[0][i]
            g = by_regime[by_regime["condition"] == cond].copy()
            if g.empty:
                ax.set_visible(False)
                continue
            rows = sorted(g["homophily_target"].unique())
            cols = sorted(g["edge_signal_target"].unique())
            mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
            for r, hv in enumerate(rows):
                for c, sv in enumerate(cols):
                    s = g[(g["homophily_target"] == hv) & (g["edge_signal_target"] == sv)]
                    if not s.empty:
                        mat[r, c] = float(s["delta_nmi_vs_baseline"].iloc[0])
            im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
            ax.set_title(f"{cond}: regime heatmap")
            ax.set_xlabel("edge_signal_target")
            ax.set_ylabel("homophily_target")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{x:.2f}" for x in cols], rotation=45, ha="right")
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{x:.2f}" for x in rows])
            fig.colorbar(im, ax=ax, shrink=0.85, label="delta_nmi_vs_baseline")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_regime_heatmap_delta_nmi.png", dpi=180)
        plt.close(fig)


def write_report(out_dir: Path, summary: pd.DataFrame, by_regime: pd.DataFrame, baseline_condition: str) -> None:
    report = out_dir / "README.md"
    lines = []
    lines.append("# Mechanism Validation Report")
    lines.append("")
    lines.append("This folder contains synthetic mechanism-validation runs over controllable regimes.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `runs.csv`: raw run-level metrics and dataset meta.")
    lines.append("- `summary_by_condition.csv`: overall average metrics by condition.")
    lines.append("- `delta_vs_baseline_runs.csv`: per-run delta against baseline.")
    lines.append("- `delta_vs_baseline_by_dataset_condition.csv`: per-dataset-condition averages.")
    lines.append("- `regime_summary.csv`: aggregated by (`homophily_target`, `edge_signal_target`).")
    lines.append("- `phase_diagram_points.csv`: phase-map points for plotting.")
    lines.append("- `figures/fig1_phase_scatter_delta_nmi.png`")
    lines.append("- `figures/fig2_regime_heatmap_delta_nmi.png`")
    lines.append("")
    lines.append(f"Baseline condition: `{baseline_condition}`")
    lines.append("")
    if not summary.empty:
        top = summary.iloc[0]
        lines.append("## Top Condition (by mean NMI)")
        lines.append(
            f"- `{top['condition']}`: NMI={safe_float(top['nmi_mean']):.4f}, "
            f"ARI={safe_float(top['ari_mean']):.4f}, SI-loss={safe_float(top['si_loss_mean']):.4f}"
        )
        lines.append("")
    if not by_regime.empty and "delta_nmi_vs_baseline" in by_regime.columns:
        best = by_regime.dropna(subset=["delta_nmi_vs_baseline"]).sort_values("delta_nmi_vs_baseline", ascending=False)
        if not best.empty:
            r = best.iloc[0]
            lines.append("## Best Regime Delta")
            lines.append(
                f"- condition=`{r['condition']}`, homophily={safe_float(r['homophily_target']):.2f}, "
                f"edge_signal={safe_float(r['edge_signal_target']):.2f}, "
                f"delta_nmi={safe_float(r['delta_nmi_vs_baseline']):+.4f}"
            )
            lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic mechanism-validation suite for the current active mainline family.")
    parser.add_argument("--tag", type=str, default="benchmark_mechanism_synth_v1")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="synth_mech")
    parser.add_argument("--datasets", type=str, default="", help="Optional dataset list, comma-separated.")
    parser.add_argument("--conditions", type=str, default="baseline_v1,g20_se_consistent_main,b45_v31_msgcond_gs050")
    parser.add_argument("--baseline_condition", type=str, default="baseline_v1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_freq", type=int, default=180)
    parser.add_argument("--train_log_interval", type=int, default=30)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--max_nums", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / args.root_path
    out_dir = repo_root / "results" / "mainline_evidence" / args.tag
    raw_runs_root = repo_root / "results" / "mainline_evidence" / "raw_runs" / args.tag
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_root.mkdir(parents=True, exist_ok=True)

    if args.datasets.strip():
        datasets = parse_list(args.datasets)
    else:
        datasets = discover_datasets(root_path=data_root, prefix=args.prefix)
    if not datasets:
        raise ValueError(
            f"No synthetic datasets found under {data_root} with prefix '{args.prefix}'. "
            "Run tools/prepare_mechanism_synth_datasets.py first."
        )

    seeds = parse_int_list(args.seeds)
    conditions = parse_list(args.conditions)
    presets = {c: load_preset(repo_root, c) for c in conditions}
    meta_map = {d: read_dataset_meta(data_root, d) for d in datasets}

    rows: List[Dict[str, object]] = []
    total = len(datasets) * len(seeds) * len(conditions)
    done = 0
    for dataset in datasets:
        for seed in seeds:
            for cond in conditions:
                done += 1
                version = f"mainline_evidence/raw_runs/{args.tag}/{cond}_{dataset}_s{seed}"
                metrics_path = repo_root / "results" / version / f"{dataset}_metrics.json"
                log_path = logs_dir / f"{version}.log"
                cmd = build_cmd(
                    python_bin=sys.executable,
                    dataset=dataset,
                    seed=seed,
                    version=version,
                    preset=presets[cond],
                    args=args,
                )
                print(f"[{done}/{total}] {dataset} | {cond} | seed={seed}")
                t0 = time.time()
                if args.dry_run:
                    print("CMD:", " ".join(cmd))
                    rc = 0
                elif args.resume and metrics_path.exists():
                    rc = 0
                else:
                    rc = run_one(cmd=cmd, cwd=repo_root, log_path=log_path)
                sec = time.time() - t0

                metrics = {}
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                    except Exception:
                        metrics = {}

                m = metrics if isinstance(metrics, dict) else {}
                row = {
                    "dataset": dataset,
                    "condition": cond,
                    "seed": int(seed),
                    "version": version,
                    "rc": int(rc),
                    "seconds": float(sec),
                    "nmi_mean": safe_float(m.get("nmi_mean")),
                    "ari_mean": safe_float(m.get("ari_mean")),
                    "acc_mean": safe_float(m.get("acc_mean")),
                    "si_loss_mean": safe_float(m.get("si_loss_mean")),
                    "modularity_mean": safe_float(m.get("modularity_mean")),
                    "conductance_weighted_mean": safe_float(m.get("conductance_weighted_mean")),
                    "pred_n_clusters_mean": safe_float(m.get("pred_n_clusters_mean")),
                    "final_graph_alpha_mean": safe_float(m.get("final_graph_alpha_mean")),
                    "final_edge_reliability_mean": safe_float(m.get("final_edge_reliability_mean")),
                    "final_edge_mix_beta_mean": safe_float(m.get("final_edge_mix_beta_mean")),
                    "final_edge_factor_mean_mean": safe_float(m.get("final_edge_factor_mean_mean")),
                    "final_edge_reg_mean": safe_float(m.get("final_edge_reg_mean")),
                    "edge_variant": str(m.get("edge_variant", presets[cond].get("edge_variant", "V1"))),
                }
                for k, v in meta_map.get(dataset, {}).items():
                    if k in row:
                        continue
                    row[k] = v
                rows.append(row)

    aggregate(out_dir=out_dir, rows=rows, baseline_condition=args.baseline_condition)
    print(f"[done] results at {out_dir}")


if __name__ == "__main__":
    main()
