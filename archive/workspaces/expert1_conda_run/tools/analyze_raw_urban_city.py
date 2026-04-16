#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAND_USE_CANDIDATES = [
    "Civic",
    "Commercial",
    "Entertainment",
    "Food",
    "Healthcare",
    "Institutional",
    "Recreational",
    "Social",
]

PLOT_ENV_PREFIXES = ("canopy_", "Canopy height", "lcz_", "2025_")
DEMOGRAPHIC_CANDIDATES = ["PopSum", "Men", "Women", "Elderly", "Youth", "Children", "population"]
PLOT_POPULATION_CANDIDATES = ["2025_pop", "PopSum", "population"]
PLOT_BUILT_CANDIDATES = ["2025_built"]
PLOT_CANOPY_CANDIDATES = ["canopy_mean", "Canopy height_mean"]


def _safe_membership_len(v) -> int:
    if v is None:
        return 0
    if isinstance(v, float) and np.isnan(v):
        return 0
    if isinstance(v, (int, np.integer)) and int(v) == 0:
        return 0
    if isinstance(v, np.ndarray):
        return int(v.size)
    if isinstance(v, (list, tuple, set)):
        return int(len(v))
    return 1


def _membership_lengths(arr: np.ndarray) -> np.ndarray:
    return np.asarray([_safe_membership_len(x) for x in arr], dtype=np.int32)


def _membership_stats(arr: np.ndarray) -> Dict[str, float]:
    lens = _membership_lengths(arr)
    return {
        "min": int(lens.min()),
        "p50": float(np.median(lens)),
        "p95": float(np.quantile(lens, 0.95)),
        "p99": float(np.quantile(lens, 0.99)),
        "max": int(lens.max()),
        "mean": float(lens.mean()),
    }


def _dominant_landuse(plot_df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in LAND_USE_CANDIDATES if c in plot_df.columns]
    if not cols:
        return np.asarray(["unknown"] * len(plot_df), dtype=object)
    scores = plot_df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    best_idx = np.argmax(scores, axis=1)
    best_val = np.max(scores, axis=1)
    labels = np.asarray([cols[i] for i in best_idx], dtype=object)
    labels[best_val <= 0] = "unknown"
    return labels


def _value_counts(arr: Sequence[object]) -> Dict[str, int]:
    uniq, counts = np.unique(np.asarray(arr, dtype=object), return_counts=True)
    order = np.argsort(-counts)
    return {str(uniq[i]): int(counts[i]) for i in order}


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return ""


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _plot_groups(plot_df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = _numeric_columns(plot_df)
    landuse = [c for c in LAND_USE_CANDIDATES if c in numeric]
    morph = [c for c in numeric if c.startswith("plot_") and c != "plot_id"]
    env = [
        c for c in numeric
        if c not in landuse
        and c not in morph
        and (any(c.startswith(prefix) for prefix in PLOT_ENV_PREFIXES) or c in DEMOGRAPHIC_CANDIDATES)
    ]
    other = [c for c in numeric if c not in {"plot_id", *landuse, *morph, *env}]
    return {
        "numeric_all": numeric,
        "landuse": landuse,
        "morph": morph,
        "env": env,
        "other": other,
    }


def _building_numeric(building_df: pd.DataFrame) -> List[str]:
    return [c for c in _numeric_columns(building_df) if c != "bid"]


def _intersection_metrics(intersection_df: pd.DataFrame) -> List[str]:
    blocked = {"intersection_id", "osmid", "x", "y"}
    return [c for c in _numeric_columns(intersection_df) if c not in blocked]


def _street_length_stats(street_df: pd.DataFrame) -> Dict[str, float]:
    length = pd.to_numeric(street_df["length"], errors="coerce").fillna(0.0)
    return {
        "count": int(length.shape[0]),
        "mean": float(length.mean()),
        "p50": float(length.quantile(0.5)),
        "p95": float(length.quantile(0.95)),
        "p99": float(length.quantile(0.99)),
        "max": float(length.max()),
    }


def _street_endpoint_degree_stats(street_df: pd.DataFrame) -> Dict[str, float]:
    endpoints = pd.concat([street_df["u"], street_df["v"]], axis=0)
    deg = endpoints.value_counts()
    return {
        "unique_endpoints": int(deg.shape[0]),
        "mean": float(deg.mean()),
        "p50": float(deg.quantile(0.5)),
        "p95": float(deg.quantile(0.95)),
        "max": int(deg.max()),
    }


def _numeric_summary(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        nonnull = s.notna()
        valid = s[nonnull]
        row: Dict[str, float | str] = {
            "column": col,
            "count": int(s.shape[0]),
            "non_null": int(nonnull.sum()),
            "missing_frac": float(1.0 - nonnull.mean()),
        }
        if valid.empty:
            row.update(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p01": np.nan,
                    "p50": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "skew": np.nan,
                }
            )
        else:
            row.update(
                {
                    "mean": float(valid.mean()),
                    "std": float(valid.std(ddof=0)),
                    "min": float(valid.min()),
                    "p01": float(valid.quantile(0.01)),
                    "p50": float(valid.quantile(0.50)),
                    "p99": float(valid.quantile(0.99)),
                    "max": float(valid.max()),
                    "skew": float(valid.skew()),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["missing_frac", "column"], ascending=[False, True]).reset_index(drop=True)


def _corr(df: pd.DataFrame, cols: Sequence[str], method: str) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    work = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    if work.shape[1] == 1:
        return pd.DataFrame([[1.0]], index=cols, columns=cols)
    corr = work.corr(method=method).fillna(0.0)
    return corr


def _cross_corr(df: pd.DataFrame, rows: Sequence[str], cols: Sequence[str], method: str) -> pd.DataFrame:
    if not rows or not cols:
        return pd.DataFrame()
    block = df[list(dict.fromkeys(list(rows) + list(cols)))].apply(pd.to_numeric, errors="coerce")
    corr = block.corr(method=method).fillna(0.0)
    out = corr.loc[list(rows), list(cols)].copy()
    return out


def _top_abs_corr_pairs(corr_df: pd.DataFrame, topk: int = 20) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    if corr_df.empty:
        return out
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            val = float(corr_df.iloc[i, j])
            out.append({"a": a, "b": b, "corr": val, "abs_corr": abs(val)})
    out.sort(key=lambda x: (-float(x["abs_corr"]), str(x["a"]), str(x["b"])))
    return out[:topk]


def _top_cross_corr(corr_df: pd.DataFrame, topk: int = 30) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    if corr_df.empty:
        return out
    for r in corr_df.index:
        for c in corr_df.columns:
            val = float(corr_df.loc[r, c])
            out.append({"feature": str(r), "target": str(c), "corr": val, "abs_corr": abs(val)})
    out.sort(key=lambda x: (-float(x["abs_corr"]), str(x["feature"]), str(x["target"])))
    return out[:topk]


def _heatmap(
    ax,
    df: pd.DataFrame,
    title: str,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    x_rotation: int = 60,
    y_rotation: int = 0,
) -> None:
    if df.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)", fontsize=10)
        return
    img = ax.imshow(df.to_numpy(dtype=np.float32), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(list(df.columns), fontsize=7, rotation=x_rotation, ha="right")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(list(df.index), fontsize=7, rotation=y_rotation)
    ax.set_title(title, fontsize=11)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def _barh(ax, labels: Sequence[str], values: Sequence[float], title: str, xlabel: str) -> None:
    if not labels:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)", fontsize=10)
        return
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def _hist(ax, arr: Sequence[float], title: str, bins: int = 50, log1p: bool = False) -> None:
    vals = pd.to_numeric(pd.Series(arr), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if vals.size == 0:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)", fontsize=10)
        return
    if log1p:
        vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        title = f"{title} (log1p)"
    lo = float(np.quantile(vals, 0.005))
    hi = float(np.quantile(vals, 0.995))
    if hi > lo:
        vals = vals[(vals >= lo) & (vals <= hi)]
    ax.hist(vals, bins=bins, color="#72b7b2", edgecolor="white", linewidth=0.2)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.2)


def _write_markdown(
    out_path: Path,
    city: str,
    summary: Dict[str, object],
    figs: Sequence[str],
    tables: Sequence[str],
) -> None:
    lines = [
        f"# Raw Data Analysis: {city}",
        "",
        "## Overview",
        f"- plots: `{summary['row_counts']['plot']:,}`",
        f"- streets: `{summary['row_counts']['street']:,}`",
        f"- buildings: `{summary['row_counts']['building']:,}`",
        f"- intersections: `{summary['row_counts']['intersection']:,}`",
        "",
        "## Figures",
    ]
    for fig in figs:
        lines.append(f"- `{fig}`")
    lines.extend(["", "## Tables"])
    for table in tables:
        lines.append(f"- `{table}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed exploratory analysis for raw urban city data.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="results/raw_analysis")
    parser.add_argument("--corr_method", type=str, default="spearman", choices=["spearman", "pearson"])
    args = parser.parse_args()

    city = args.city.strip()
    city_dir = Path(args.urban_root).resolve() / city
    out_dir = Path(args.out_root).resolve() / city
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "boundary": city_dir / "boundary.parquet",
        "building": city_dir / "building.parquet",
        "intersection": city_dir / "intersection.parquet",
        "plot": city_dir / "plot.parquet",
        "street": city_dir / "street.parquet",
        "plot_street": city_dir / "plot_street_id.npy",
        "plot_bid": city_dir / "plot_bid.npy",
    }
    missing = [str(fp) for fp in paths.values() if not fp.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    boundary_df = pd.read_parquet(paths["boundary"])
    building_df = pd.read_parquet(paths["building"])
    intersection_df = pd.read_parquet(paths["intersection"])
    plot_df = pd.read_parquet(paths["plot"])
    street_df = pd.read_parquet(paths["street"])
    plot_street = np.load(paths["plot_street"], allow_pickle=True)
    plot_bid = np.load(paths["plot_bid"], allow_pickle=True)

    plot_groups = _plot_groups(plot_df)
    building_numeric = _building_numeric(building_df)
    intersection_metrics = _intersection_metrics(intersection_df)
    dominant_landuse = _dominant_landuse(plot_df)
    population_col = _first_existing(plot_df.columns, PLOT_POPULATION_CANDIDATES)
    built_col = _first_existing(plot_df.columns, PLOT_BUILT_CANDIDATES)
    canopy_col = _first_existing(plot_df.columns, PLOT_CANOPY_CANDIDATES)

    street_endpoint_map = street_df[["street_id", "u", "v"]].dropna()
    endpoint_map: Dict[int, set[int]] = {}
    for row in street_endpoint_map.itertuples(index=False):
        sid = int(row.street_id)
        endpoint_map.setdefault(sid, set()).update([int(row.u), int(row.v)])
    plot_junction: List[List[int]] = []
    for ids in plot_street:
        agg: set[int] = set()
        if isinstance(ids, np.ndarray):
            iter_ids = ids.tolist()
        elif isinstance(ids, (list, tuple, set)):
            iter_ids = list(ids)
        else:
            iter_ids = [ids]
        for sid in iter_ids:
            try:
                agg.update(endpoint_map.get(int(sid), set()))
            except Exception:
                continue
        plot_junction.append(sorted(agg))
    plot_junction_arr = np.asarray(plot_junction, dtype=object)

    row_counts = {
        "boundary": int(len(boundary_df)),
        "building": int(len(building_df)),
        "intersection": int(len(intersection_df)),
        "plot": int(len(plot_df)),
        "street": int(len(street_df)),
    }

    plot_summary = _numeric_summary(plot_df, plot_groups["numeric_all"])
    building_summary = _numeric_summary(building_df, building_numeric)
    intersection_summary = _numeric_summary(intersection_df, intersection_metrics)
    street_summary = _numeric_summary(street_df, ["length"])

    plot_corr_morph = _corr(plot_df, plot_groups["morph"], args.corr_method)
    plot_corr_env = _corr(plot_df, plot_groups["env"], args.corr_method)
    plot_corr_landuse = _corr(plot_df, plot_groups["landuse"], args.corr_method)
    plot_landuse_cross = _cross_corr(
        plot_df,
        rows=plot_groups["morph"] + plot_groups["env"],
        cols=plot_groups["landuse"],
        method=args.corr_method,
    )
    if not plot_landuse_cross.empty:
        order = plot_landuse_cross.abs().max(axis=1).sort_values(ascending=False).index.tolist()
        plot_landuse_cross = plot_landuse_cross.loc[order]

    building_corr = _corr(building_df, building_numeric, args.corr_method)
    intersection_corr = _corr(intersection_df, intersection_metrics, args.corr_method)

    all_missing = pd.concat(
        [
            plot_summary.assign(layer="plot")[["layer", "column", "missing_frac"]],
            building_summary.assign(layer="building")[["layer", "column", "missing_frac"]],
            intersection_summary.assign(layer="intersection")[["layer", "column", "missing_frac"]],
            street_summary.assign(layer="street")[["layer", "column", "missing_frac"]],
        ],
        ignore_index=True,
    ).sort_values(["missing_frac", "layer", "column"], ascending=[False, True, True])

    top_missing = all_missing.head(20)

    # Save summary tables before plotting.
    plot_summary.to_csv(out_dir / "plot_numeric_summary.csv", index=False)
    building_summary.to_csv(out_dir / "building_numeric_summary.csv", index=False)
    intersection_summary.to_csv(out_dir / "intersection_numeric_summary.csv", index=False)
    street_summary.to_csv(out_dir / "street_numeric_summary.csv", index=False)
    all_missing.to_csv(out_dir / "numeric_missingness.csv", index=False)
    plot_corr_morph.to_csv(out_dir / "plot_morph_corr.csv")
    plot_corr_env.to_csv(out_dir / "plot_env_corr.csv")
    plot_corr_landuse.to_csv(out_dir / "plot_landuse_corr.csv")
    plot_landuse_cross.to_csv(out_dir / "plot_feature_vs_landuse_corr.csv")
    building_corr.to_csv(out_dir / "building_corr.csv")
    intersection_corr.to_csv(out_dir / "intersection_corr.csv")

    # Figure 1: layer inventory and land-use / membership overview.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=170)
    layer_names = list(row_counts.keys())
    layer_vals = [row_counts[k] for k in layer_names]
    axes[0, 0].bar(layer_names, layer_vals, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
    axes[0, 0].set_title("Entity Counts by Layer", fontsize=11)
    axes[0, 0].set_ylabel("rows", fontsize=9)
    axes[0, 0].tick_params(axis="x", labelrotation=20)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.25)

    lc = _value_counts(dominant_landuse)
    _barh(axes[0, 1], list(lc.keys()), list(lc.values()), "Dominant Plot Land-Use", "count")

    _hist(axes[1, 0], _membership_lengths(plot_street), "Plot -> street memberships", bins=60, log1p=True)
    _hist(axes[1, 1], _membership_lengths(plot_bid), "Plot -> building memberships", bins=60, log1p=True)
    fig.suptitle(f"Raw Urban Data Inventory: {city}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig1_inventory.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: plot-level correlations.
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=170)
    _heatmap(axes[0, 0], plot_corr_morph, f"Plot Morphology Correlation ({args.corr_method})")
    _heatmap(axes[0, 1], plot_corr_env, f"Plot Environment Correlation ({args.corr_method})")
    _heatmap(axes[1, 0], plot_corr_landuse, f"Plot Land-Use Score Correlation ({args.corr_method})")
    _heatmap(axes[1, 1], plot_landuse_cross, f"Plot Feature vs Land-Use Correlation ({args.corr_method})")
    fig.suptitle(f"Plot-Level Correlation Analysis: {city}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / "fig2_plot_correlations.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: building/intersection correlations.
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=170)
    _heatmap(axes[0], building_corr, f"Building Correlation ({args.corr_method})")
    _heatmap(axes[1], intersection_corr, f"Intersection Metric Correlation ({args.corr_method})")
    fig.suptitle(f"Building and Network Metric Correlations: {city}", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "fig3_building_intersection_correlations.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: distributions.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=170)
    _hist(axes[0, 0], plot_df["plot_area"], "plot_area", bins=70, log1p=True)
    _hist(axes[0, 1], plot_df["plot_perimeter"], "plot_perimeter", bins=70, log1p=True)
    _hist(axes[0, 2], plot_df[canopy_col] if canopy_col else [], canopy_col or "canopy", bins=60)
    _hist(axes[0, 3], plot_df[population_col] if population_col else [], population_col or "population", bins=70, log1p=True)
    _hist(axes[1, 0], plot_df[built_col] if built_col else [], built_col or "built-up", bins=70)
    _hist(axes[1, 1], building_df["bid_area"], "bid_area", bins=70, log1p=True)
    _hist(axes[1, 2], building_df["bid_height"], "bid_height", bins=70, log1p=True)
    _hist(axes[1, 3], street_df["length"], "street.length", bins=70, log1p=True)
    fig.suptitle(f"Key Raw Feature Distributions: {city}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig4_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: missingness.
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=170)
    labels = [f"{row.layer}.{row.column}" for row in top_missing.itertuples(index=False)]
    values = [float(v) for v in top_missing["missing_frac"].tolist()]
    _barh(ax, labels, values, "Top Numeric Missingness", "missing fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_missingness.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "city": city,
        "corr_method": args.corr_method,
        "row_counts": row_counts,
        "plot_groups": plot_groups,
        "plot_canopy_column": canopy_col,
        "plot_population_column": population_col,
        "plot_built_column": built_col,
        "building_numeric_columns": building_numeric,
        "intersection_metric_columns": intersection_metrics,
        "dominant_landuse_counts": lc,
        "plot_street_membership_stats": _membership_stats(plot_street),
        "plot_bid_membership_stats": _membership_stats(plot_bid),
        "plot_junction_membership_stats": _membership_stats(plot_junction_arr),
        "street_length_stats": _street_length_stats(street_df),
        "street_endpoint_degree_stats": _street_endpoint_degree_stats(street_df),
        "top_plot_morph_corr_pairs": _top_abs_corr_pairs(plot_corr_morph, topk=20),
        "top_plot_env_corr_pairs": _top_abs_corr_pairs(plot_corr_env, topk=20),
        "top_building_corr_pairs": _top_abs_corr_pairs(building_corr, topk=20),
        "top_intersection_corr_pairs": _top_abs_corr_pairs(intersection_corr, topk=20),
        "top_plot_feature_vs_landuse_corr": _top_cross_corr(plot_landuse_cross, topk=30),
        "figures": [
            "fig1_inventory.png",
            "fig2_plot_correlations.png",
            "fig3_building_intersection_correlations.png",
            "fig4_distributions.png",
            "fig5_missingness.png",
        ],
        "tables": [
            "plot_numeric_summary.csv",
            "building_numeric_summary.csv",
            "intersection_numeric_summary.csv",
            "street_numeric_summary.csv",
            "numeric_missingness.csv",
            "plot_morph_corr.csv",
            "plot_env_corr.csv",
            "plot_landuse_corr.csv",
            "plot_feature_vs_landuse_corr.csv",
            "building_corr.csv",
            "intersection_corr.csv",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(
        out_path=out_dir / "README.md",
        city=city,
        summary=summary,
        figs=summary["figures"],
        tables=summary["tables"],
    )

    print(f"[ok] raw analysis complete for {city}")
    print(f"out_dir={out_dir}")
    print(f"plots={row_counts['plot']}, streets={row_counts['street']}, buildings={row_counts['building']}, intersections={row_counts['intersection']}")


if __name__ == "__main__":
    main()
