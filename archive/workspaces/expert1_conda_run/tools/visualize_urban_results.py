#!/usr/bin/env python3
"""
Visualize urban benchmark results:
1) overlap performance deltas + speedup
2) convergence curves per dataset (multi-seed)
3) stability metrics overview
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE2_RE = re.compile(r"\[Stage2\]\s+Epoch\s+(\d+):\s+loss=([0-9eE+\-.]+)")


def _short_name(dataset: str) -> str:
    name = dataset.replace("urban_", "").replace("_plot", "")
    return name


def _read_loss_curve(log_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not log_path.exists():
        return np.array([]), np.array([])
    epochs: List[int] = []
    losses: List[float] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STAGE2_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return np.array(epochs, dtype=np.int32), np.array(losses, dtype=np.float64)


def _plot_overlap_compare(df_overlap: pd.DataFrame, out_dir: Path) -> Path:
    if df_overlap.empty:
        raise ValueError("compare_overlap data is empty.")

    plot_df = df_overlap.copy()
    plot_df["short"] = plot_df["dataset"].map(_short_name)
    plot_df = plot_df.sort_values("delta_nmi", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)

    ax = axes[0]
    ax.bar(x - w / 2, plot_df["delta_nmi"].values, width=w, label="ΔNMI", color="#2B8CBE")
    ax.bar(x + w / 2, plot_df["delta_ari"].values, width=w, label="ΔARI", color="#F28E2B")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Accuracy Delta vs Previous Round (Overlap Datasets)")
    ax.set_ylabel("Delta")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["short"].tolist(), rotation=35, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    ax.bar(x, plot_df["speedup_x"].values, color="#1B9E77")
    ax.set_yscale("log")
    ax.set_title("Runtime Speedup (Previous / New)")
    ax.set_ylabel("Speedup x (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["short"].tolist(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2, which="both")

    fig.tight_layout()
    out_path = out_dir / "fig1_overlap_delta_speedup.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_convergence_grid(df_runs: pd.DataFrame, df_stability: pd.DataFrame, out_dir: Path) -> Path:
    if df_runs.empty:
        raise ValueError("runs.csv is empty.")

    datasets = sorted(df_runs["dataset"].unique().tolist())
    n = len(datasets)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.2 * nrows), dpi=180)
    axes = np.array(axes).reshape(nrows, ncols)

    stab_map = {r["dataset"]: r for _, r in df_stability.iterrows()}

    for idx, dataset in enumerate(datasets):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        sub = df_runs[df_runs["dataset"] == dataset].sort_values("seed")
        curves = []
        common_epochs = None

        for _, row in sub.iterrows():
            seed = int(row["seed"])
            log_path = Path(row["log_path"])
            epochs, losses = _read_loss_curve(log_path)
            if len(epochs) == 0:
                continue
            curves.append((seed, epochs, losses))
            if common_epochs is None:
                common_epochs = epochs
            ax.plot(epochs, losses, linewidth=1.2, alpha=0.8, label=f"s{seed}")

        if curves:
            min_len = min(len(x[1]) for x in curves)
            # Align by index (all runs in this suite should share epoch points).
            aligned = np.stack([x[2][:min_len] for x in curves], axis=0)
            mean_curve = aligned.mean(axis=0)
            std_curve = aligned.std(axis=0)
            x_epoch = curves[0][1][:min_len]
            ax.plot(x_epoch, mean_curve, color="black", linewidth=2.0, label="mean")
            ax.fill_between(
                x_epoch,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color="gray",
                alpha=0.15,
                linewidth=0,
            )

        stab = stab_map.get(dataset, {})
        label = stab.get("stability", "n/a")
        ax.set_title(f"{_short_name(dataset)} ({label})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        if len(curves) > 0:
            ax.legend(fontsize=7, ncol=2, frameon=False)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle("Convergence Curves by Dataset (Multi-seed)", y=1.01, fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "fig2_convergence_grid.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_stability_overview(df_stability: pd.DataFrame, out_dir: Path) -> Path:
    if df_stability.empty:
        raise ValueError("per_dataset_stability.csv is empty.")

    df = df_stability.copy()
    df["short"] = df["dataset"].map(_short_name)
    df = df.sort_values("tail_band_pct_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=180)

    ax = axes[0]
    ax.bar(x, df["tail_band_pct_mean"].values, color="#4C78A8")
    ax.set_title("Tail Band (Lower is More Stable)")
    ax.set_ylabel("tail_band_pct_mean")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"].tolist(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    ax.bar(x, df["final_loss_cv_pct"].values, color="#59A14F")
    ax.set_title("Cross-seed Final-loss CV (Lower is Better)")
    ax.set_ylabel("final_loss_cv_pct")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"].tolist(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    sc = ax.scatter(
        df["tail_band_pct_mean"].values,
        df["nmi_final_mean"].values,
        s=np.clip(df["seconds_mean"].values, 0, None) * 8.0 + 30.0,
        c=df["sign_flip_ratio_mean"].values,
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )
    for _, row in df.iterrows():
        ax.annotate(_short_name(row["dataset"]), (row["tail_band_pct_mean"], row["nmi_final_mean"]), fontsize=7, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("sign_flip_ratio_mean")
    ax.set_title("Stability vs NMI (bubble=size by runtime)")
    ax.set_xlabel("tail_band_pct_mean")
    ax.set_ylabel("nmi_final_mean")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path = out_dir / "fig3_stability_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize urban benchmark and convergence results.")
    parser.add_argument(
        "--compare_overlap_csv",
        type=str,
        default="results/urban_batch_eval_phase2_edge_v1/compare_overlap_vs_fast_v2.csv",
        help="CSV containing overlap comparison (delta_nmi, delta_ari, speedup_x).",
    )
    parser.add_argument(
        "--stability_csv",
        type=str,
        default="results/convergence_stability_v2/per_dataset_stability.csv",
        help="CSV containing per-dataset stability metrics.",
    )
    parser.add_argument(
        "--runs_csv",
        type=str,
        default="results/convergence_stability_v2/runs.csv",
        help="CSV containing run metadata and log paths.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/convergence_stability_v2/figures",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()

    compare_overlap_csv = Path(args.compare_overlap_csv)
    stability_csv = Path(args.stability_csv)
    runs_csv = Path(args.runs_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_overlap = pd.read_csv(compare_overlap_csv)
    df_stability = pd.read_csv(stability_csv)
    df_runs = pd.read_csv(runs_csv)

    f1 = _plot_overlap_compare(df_overlap, out_dir)
    f2 = _plot_convergence_grid(df_runs, df_stability, out_dir)
    f3 = _plot_stability_overview(df_stability, out_dir)

    print(f"[ok] {f1}")
    print(f"[ok] {f2}")
    print(f"[ok] {f3}")


if __name__ == "__main__":
    main()

