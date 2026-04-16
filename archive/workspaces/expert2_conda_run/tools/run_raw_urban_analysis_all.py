#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAND_USE_ORDER = [
    "unknown",
    "Commercial",
    "Food",
    "Institutional",
    "Recreational",
    "Entertainment",
    "Civic",
    "Healthcare",
    "Social",
]

LAND_USE_COLORS = {
    "unknown": "#bab0ab",
    "Commercial": "#f58518",
    "Food": "#72b7b2",
    "Institutional": "#b279a2",
    "Recreational": "#ff9da6",
    "Entertainment": "#e45756",
    "Civic": "#4c78a8",
    "Healthcare": "#54a24b",
    "Social": "#9d755d",
}


def _discover_cities(urban_root: Path) -> List[str]:
    return sorted([p.name for p in urban_root.iterdir() if p.is_dir()])


def _run_city_analysis(city: str, urban_root: Path, out_root: Path, corr_method: str, python_exe: str) -> Dict[str, object]:
    script = Path(__file__).with_name("analyze_raw_urban_city.py").resolve()
    cmd = [
        python_exe,
        str(script),
        "--city",
        city,
        "--urban_root",
        str(urban_root),
        "--out_root",
        str(out_root),
        "--corr_method",
        corr_method,
    ]
    subprocess.run(cmd, check=True)
    summary_path = out_root / city / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _read_feature_stat(csv_path: Path, column_name: str, stat_name: str) -> float:
    if not csv_path.exists():
        return float("nan")
    df = pd.read_csv(csv_path)
    if "column" not in df.columns or stat_name not in df.columns:
        return float("nan")
    rows = df[df["column"] == column_name]
    if rows.empty:
        return float("nan")
    try:
        return float(rows.iloc[0][stat_name])
    except Exception:
        return float("nan")


def _offdiag_corr_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {"mean_abs": np.nan, "median_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan, "frac_abs_ge_08": np.nan}
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty or df.shape[0] <= 1:
        return {"mean_abs": np.nan, "median_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan, "frac_abs_ge_08": np.nan}
    vals = df.to_numpy(dtype=np.float64)
    mask = ~np.eye(vals.shape[0], dtype=bool)
    off = np.abs(vals[mask])
    if off.size == 0:
        return {"mean_abs": np.nan, "median_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan, "frac_abs_ge_08": np.nan}
    return {
        "mean_abs": float(off.mean()),
        "median_abs": float(np.median(off)),
        "p95_abs": float(np.quantile(off, 0.95)),
        "max_abs": float(off.max()),
        "frac_abs_ge_08": float(np.mean(off >= 0.8)),
    }


def _cross_corr_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {"mean_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan}
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        return {"mean_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan}
    vals = np.abs(df.to_numpy(dtype=np.float64).reshape(-1))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan}
    return {
        "mean_abs": float(vals.mean()),
        "p95_abs": float(np.quantile(vals, 0.95)),
        "max_abs": float(vals.max()),
    }


def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if den == 0.0:
        return 0.0
    return float(num) / den


def _barh_metric(ax, df: pd.DataFrame, col: str, title: str, xlabel: str, color: str = "#4c78a8") -> None:
    temp = df.sort_values(col, ascending=True)
    ax.barh(temp["city"], temp[col], color=color, alpha=0.92)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def _barh_with_overlay(ax, df: pd.DataFrame, mean_col: str, hi_col: str, title: str, xlabel: str, color: str = "#72b7b2") -> None:
    temp = df[["city", mean_col, hi_col]].copy()
    temp = temp[np.isfinite(temp[mean_col].to_numpy(dtype=np.float64))]
    if temp.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(no comparable data)", fontsize=10)
        return
    temp = temp.sort_values(mean_col, ascending=True)
    y = np.arange(temp.shape[0])
    ax.barh(y, temp[mean_col], color=color, alpha=0.88, label=mean_col)
    ax.scatter(temp[hi_col], y, s=28, color="#e45756", label=hi_col, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(temp["city"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def _heatmap(ax, df: pd.DataFrame, title: str, cmap: str = "viridis") -> None:
    img = ax.imshow(df.to_numpy(dtype=np.float32), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title(title, fontsize=11)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def _write_readme(out_path: Path, cities: Sequence[str], files: Iterable[str]) -> None:
    lines = [
        "# Cross-City Raw Urban Analysis",
        "",
        f"- cities analyzed: `{len(cities)}`",
        f"- city list: `{', '.join(cities)}`",
        "",
        "## Artifacts",
    ]
    for fp in files:
        lines.append(f"- `{fp}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run raw urban analysis and build cross-city comparison figures.")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="results/raw_analysis")
    parser.add_argument("--cities", type=str, default="", help="Comma-separated city names. Default: all directories under urban_root.")
    parser.add_argument("--corr_method", type=str, default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable used for per-city runs.")
    args = parser.parse_args()

    urban_root = Path(args.urban_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cities.strip():
        cities = sorted([x.strip() for x in args.cities.split(",") if x.strip()])
    else:
        cities = _discover_cities(urban_root)
    if not cities:
        raise ValueError("No cities found to analyze.")

    summaries: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    landuse_rows: List[Dict[str, object]] = []
    corr_rows: List[Dict[str, object]] = []

    for city in cities:
        summary = _run_city_analysis(
            city=city,
            urban_root=urban_root,
            out_root=out_root,
            corr_method=args.corr_method,
            python_exe=args.python_exe,
        )
        summaries[city] = summary
        city_dir = out_root / city

        row_counts = summary["row_counts"]
        dom = summary["dominant_landuse_counts"]
        plots = float(row_counts["plot"])
        unknown = float(dom.get("unknown", 0))
        known = max(plots - unknown, 0.0)
        population_col = str(summary.get("plot_population_column", "") or "")
        built_col = str(summary.get("plot_built_column", "") or "")

        plot_summary_fp = city_dir / "plot_numeric_summary.csv"
        building_summary_fp = city_dir / "building_numeric_summary.csv"
        morph_corr_fp = city_dir / "plot_morph_corr.csv"
        env_corr_fp = city_dir / "plot_env_corr.csv"
        building_corr_fp = city_dir / "building_corr.csv"
        intersection_corr_fp = city_dir / "intersection_corr.csv"
        feature_vs_landuse_fp = city_dir / "plot_feature_vs_landuse_corr.csv"

        rows.append(
            {
                "city": city,
                "plot_rows": int(row_counts["plot"]),
                "street_rows": int(row_counts["street"]),
                "building_rows": int(row_counts["building"]),
                "intersection_rows": int(row_counts["intersection"]),
                "unknown_rate": _safe_ratio(unknown, plots),
                "known_rate": _safe_ratio(known, plots),
                "plot_street_mean": float(summary["plot_street_membership_stats"]["mean"]),
                "plot_street_p50": float(summary["plot_street_membership_stats"]["p50"]),
                "plot_street_p95": float(summary["plot_street_membership_stats"]["p95"]),
                "plot_bid_mean": float(summary["plot_bid_membership_stats"]["mean"]),
                "plot_bid_p50": float(summary["plot_bid_membership_stats"]["p50"]),
                "plot_bid_p95": float(summary["plot_bid_membership_stats"]["p95"]),
                "plot_junction_mean": float(summary["plot_junction_membership_stats"]["mean"]),
                "plot_junction_p50": float(summary["plot_junction_membership_stats"]["p50"]),
                "plot_junction_p95": float(summary["plot_junction_membership_stats"]["p95"]),
                "street_length_mean": float(summary["street_length_stats"]["mean"]),
                "street_length_p50": float(summary["street_length_stats"]["p50"]),
                "street_length_p95": float(summary["street_length_stats"]["p95"]),
                "street_endpoint_degree_mean": float(summary["street_endpoint_degree_stats"]["mean"]),
                "street_endpoint_degree_p95": float(summary["street_endpoint_degree_stats"]["p95"]),
                "plot_area_p50": _read_feature_stat(plot_summary_fp, "plot_area", "p50"),
                "plot_area_p99": _read_feature_stat(plot_summary_fp, "plot_area", "p99"),
                "plot_population_column": population_col,
                "plot_built_column": built_col,
                "plot_pop_p50": _read_feature_stat(plot_summary_fp, population_col, "p50") if population_col else np.nan,
                "plot_pop_p99": _read_feature_stat(plot_summary_fp, population_col, "p99") if population_col else np.nan,
                "plot_built_p50": _read_feature_stat(plot_summary_fp, built_col, "p50") if built_col else np.nan,
                "plot_built_p99": _read_feature_stat(plot_summary_fp, built_col, "p99") if built_col else np.nan,
                "building_height_p50": _read_feature_stat(building_summary_fp, "bid_height", "p50"),
                "building_height_p95": _read_feature_stat(building_summary_fp, "bid_height", "p99"),
            }
        )

        lu_row: Dict[str, object] = {"city": city}
        for label in LAND_USE_ORDER:
            count = int(dom.get(label, 0))
            lu_row[f"{label}_count"] = count
            lu_row[f"{label}_frac_all"] = _safe_ratio(count, plots)
            lu_row[f"{label}_frac_known"] = _safe_ratio(count, known) if label != "unknown" else 0.0
        landuse_rows.append(lu_row)

        morph_stats = _offdiag_corr_stats(morph_corr_fp)
        env_stats = _offdiag_corr_stats(env_corr_fp)
        building_stats = _offdiag_corr_stats(building_corr_fp)
        intersection_stats = _offdiag_corr_stats(intersection_corr_fp)
        feature_landuse_stats = _cross_corr_stats(feature_vs_landuse_fp)
        corr_rows.append(
            {
                "city": city,
                "plot_morph_mean_abs": morph_stats["mean_abs"],
                "plot_morph_p95_abs": morph_stats["p95_abs"],
                "plot_morph_max_abs": morph_stats["max_abs"],
                "plot_morph_frac_abs_ge_08": morph_stats["frac_abs_ge_08"],
                "plot_env_mean_abs": env_stats["mean_abs"],
                "plot_env_p95_abs": env_stats["p95_abs"],
                "plot_env_max_abs": env_stats["max_abs"],
                "plot_env_frac_abs_ge_08": env_stats["frac_abs_ge_08"],
                "building_mean_abs": building_stats["mean_abs"],
                "building_p95_abs": building_stats["p95_abs"],
                "building_max_abs": building_stats["max_abs"],
                "building_frac_abs_ge_08": building_stats["frac_abs_ge_08"],
                "intersection_mean_abs": intersection_stats["mean_abs"],
                "intersection_p95_abs": intersection_stats["p95_abs"],
                "intersection_max_abs": intersection_stats["max_abs"],
                "intersection_frac_abs_ge_08": intersection_stats["frac_abs_ge_08"],
                "feature_landuse_mean_abs": feature_landuse_stats["mean_abs"],
                "feature_landuse_p95_abs": feature_landuse_stats["p95_abs"],
                "feature_landuse_max_abs": feature_landuse_stats["max_abs"],
            }
        )

    cross_dir = out_root / "_cross_city"
    cross_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
    landuse_df = pd.DataFrame(landuse_rows).sort_values("city").reset_index(drop=True)
    corr_df = pd.DataFrame(corr_rows).sort_values("city").reset_index(drop=True)

    summary_df.to_csv(cross_dir / "cross_city_summary.csv", index=False)
    landuse_df.to_csv(cross_dir / "cross_city_landuse_counts.csv", index=False)
    corr_df.to_csv(cross_dir / "cross_city_corr_redundancy.csv", index=False)
    (cross_dir / "cross_city_run_summary.json").write_text(
        json.dumps(
            {
                "cities": cities,
                "corr_method": args.corr_method,
                "python_exe": args.python_exe,
                "out_root": str(out_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figure 1: entity counts.
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=180)
    _barh_metric(axes[0, 0], summary_df, "plot_rows", "Plots by City", "count")
    _barh_metric(axes[0, 1], summary_df, "street_rows", "Streets by City", "count", color="#f58518")
    _barh_metric(axes[1, 0], summary_df, "building_rows", "Buildings by City", "count", color="#54a24b")
    _barh_metric(axes[1, 1], summary_df, "intersection_rows", "Intersections by City", "count", color="#e45756")
    fig.suptitle("Cross-City Entity Counts", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(cross_dir / "fig1_cross_city_entity_counts.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: land-use composition.
    sort_landuse = summary_df.sort_values("unknown_rate", ascending=True)["city"].tolist()
    landuse_plot_df = landuse_df.set_index("city").loc[sort_landuse]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=180)
    axes[0].barh(sort_landuse, summary_df.set_index("city").loc[sort_landuse]["unknown_rate"], color=LAND_USE_COLORS["unknown"])
    axes[0].set_title("Unknown Dominant Land-Use Rate", fontsize=11)
    axes[0].set_xlabel("fraction of plots", fontsize=9)
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)

    left = np.zeros(len(sort_landuse), dtype=np.float64)
    for label in [x for x in LAND_USE_ORDER if x != "unknown"]:
        vals = landuse_plot_df[f"{label}_frac_known"].to_numpy(dtype=np.float64)
        axes[1].barh(sort_landuse, vals, left=left, color=LAND_USE_COLORS[label], label=label)
        left += vals
    axes[1].set_title("Known Dominant Land-Use Composition", fontsize=11)
    axes[1].set_xlabel("fraction of known-label plots", fontsize=9)
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Cross-City Land-Use Composition", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(cross_dir / "fig2_cross_city_landuse.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: membership comparisons.
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), dpi=180)
    _barh_with_overlay(axes[0], summary_df, "plot_street_mean", "plot_street_p95", "Plot -> Street Membership", "count")
    _barh_with_overlay(axes[1], summary_df, "plot_bid_mean", "plot_bid_p95", "Plot -> Building Membership", "count", color="#f58518")
    _barh_with_overlay(axes[2], summary_df, "plot_junction_mean", "plot_junction_p95", "Plot -> Junction Membership", "count", color="#54a24b")
    fig.suptitle("Cross-City Membership Structure", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(cross_dir / "fig3_cross_city_memberships.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: scale and long-tail summary.
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=180)
    _barh_with_overlay(axes[0, 0], summary_df, "plot_area_p50", "plot_area_p99", "Plot Area", "area")
    _barh_with_overlay(axes[0, 1], summary_df, "plot_pop_p50", "plot_pop_p99", "Plot Population (2025_pop)", "population", color="#e45756")
    _barh_with_overlay(axes[1, 0], summary_df, "building_height_p50", "building_height_p95", "Building Height", "height", color="#4c78a8")
    _barh_with_overlay(axes[1, 1], summary_df, "street_length_p50", "street_length_p95", "Street Length", "length", color="#b279a2")
    fig.suptitle("Cross-City Scale Comparison", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(cross_dir / "fig4_cross_city_scale.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: correlation structure summary.
    corr_heat = corr_df.set_index("city")[
        [
            "plot_morph_mean_abs",
            "plot_env_mean_abs",
            "building_mean_abs",
            "intersection_mean_abs",
            "feature_landuse_mean_abs",
            "plot_morph_frac_abs_ge_08",
            "plot_env_frac_abs_ge_08",
            "building_frac_abs_ge_08",
            "intersection_frac_abs_ge_08",
        ]
    ]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=180)
    _heatmap(ax, corr_heat, "Cross-City Correlation / Redundancy Summary")
    fig.tight_layout()
    fig.savefig(cross_dir / "fig5_cross_city_corr_summary.png", bbox_inches="tight")
    plt.close(fig)

    artifact_files = sorted([p.name for p in cross_dir.iterdir() if p.is_file()])
    _write_readme(cross_dir / "README.md", cities=cities, files=artifact_files)

    print("[ok] cross-city raw analysis complete")
    print(f"cities={len(cities)}")
    print(f"out_dir={cross_dir}")


if __name__ == "__main__":
    main()
