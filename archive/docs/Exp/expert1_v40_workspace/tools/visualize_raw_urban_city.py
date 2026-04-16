#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch


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

LAND_USE_PALETTE = {
    "Civic": "#4c78a8",
    "Commercial": "#f58518",
    "Entertainment": "#e45756",
    "Food": "#72b7b2",
    "Healthcare": "#54a24b",
    "Institutional": "#b279a2",
    "Recreational": "#ff9da6",
    "Social": "#9d755d",
    "unknown": "#bab0ab",
}


def _read_u32(buf: bytes, offset: int, endian: str) -> Tuple[int, int]:
    return int(struct.unpack_from(f"{endian}I", buf, offset)[0]), offset + 4


def _read_xy_block(buf: bytes, offset: int, endian: str, n_points: int) -> Tuple[np.ndarray, int]:
    if n_points <= 0:
        return np.zeros((0, 2), dtype=np.float64), offset
    arr = np.frombuffer(buf, dtype=np.dtype(f"{endian}f8"), count=2 * n_points, offset=offset)
    arr = np.asarray(arr, dtype=np.float64).reshape(n_points, 2).copy()
    return arr, offset + 16 * n_points


def _parse_wkb(buf: bytes, offset: int = 0) -> Tuple[Dict[str, object], int]:
    if not buf:
        raise ValueError("empty WKB")
    byte_order = buf[offset]
    if byte_order == 1:
        endian = "<"
    elif byte_order == 0:
        endian = ">"
    else:
        raise ValueError(f"Unsupported WKB byte order: {byte_order}")
    offset += 1

    geom_type, offset = _read_u32(buf, offset, endian)
    if geom_type == 2:  # LineString
        n_points, offset = _read_u32(buf, offset, endian)
        coords, offset = _read_xy_block(buf, offset, endian, n_points)
        return {"type": "LineString", "coords": coords}, offset

    if geom_type == 3:  # Polygon
        n_rings, offset = _read_u32(buf, offset, endian)
        rings: List[np.ndarray] = []
        for _ in range(n_rings):
            n_points, offset = _read_u32(buf, offset, endian)
            coords, offset = _read_xy_block(buf, offset, endian, n_points)
            rings.append(coords)
        return {"type": "Polygon", "rings": rings}, offset

    if geom_type == 5:  # MultiLineString
        n_geoms, offset = _read_u32(buf, offset, endian)
        parts: List[np.ndarray] = []
        for _ in range(n_geoms):
            child, offset = _parse_wkb(buf, offset)
            if child["type"] == "LineString":
                parts.append(child["coords"])
        return {"type": "MultiLineString", "parts": parts}, offset

    if geom_type == 6:  # MultiPolygon
        n_geoms, offset = _read_u32(buf, offset, endian)
        polys: List[List[np.ndarray]] = []
        for _ in range(n_geoms):
            child, offset = _parse_wkb(buf, offset)
            if child["type"] == "Polygon":
                polys.append(child["rings"])
        return {"type": "MultiPolygon", "polys": polys}, offset

    raise ValueError(f"Unsupported WKB geometry type: {geom_type}")


def _extract_lines(wkb: bytes) -> List[np.ndarray]:
    geom, _ = _parse_wkb(bytes(wkb))
    gtype = geom["type"]
    if gtype == "LineString":
        coords = geom["coords"]
        return [coords] if len(coords) >= 2 else []
    if gtype == "MultiLineString":
        return [p for p in geom["parts"] if len(p) >= 2]
    return []


def _extract_polygon_outers(wkb: bytes) -> List[np.ndarray]:
    geom, _ = _parse_wkb(bytes(wkb))
    gtype = geom["type"]
    if gtype == "Polygon":
        rings = geom["rings"]
        return [rings[0]] if rings and len(rings[0]) >= 3 else []
    if gtype == "MultiPolygon":
        out: List[np.ndarray] = []
        for rings in geom["polys"]:
            if rings and len(rings[0]) >= 3:
                out.append(rings[0])
        return out
    return []


def _dominant_landuse(plot_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in LAND_USE_CANDIDATES if c in plot_df.columns]
    if not cols:
        return np.array(["unknown"] * len(plot_df), dtype=object), cols
    scores = plot_df[cols].fillna(0.0).to_numpy(dtype=np.float32)
    best_idx = np.argmax(scores, axis=1)
    best_val = np.max(scores, axis=1)
    labels = np.array([cols[i] for i in best_idx], dtype=object)
    labels[best_val <= 0] = "unknown"
    return labels, cols


def _safe_membership_len(v) -> int:
    if v is None:
        return 0
    if isinstance(v, float) and np.isnan(v):
        return 0
    if isinstance(v, np.ndarray):
        return int(v.size)
    if isinstance(v, (list, tuple, set)):
        return int(len(v))
    return 1


def _membership_stats(arr: np.ndarray) -> Dict[str, float]:
    lens = np.array([_safe_membership_len(x) for x in arr], dtype=np.int32)
    return {
        "min": int(lens.min()),
        "p50": float(np.median(lens)),
        "p95": float(np.quantile(lens, 0.95)),
        "max": int(lens.max()),
        "mean": float(lens.mean()),
    }


def _sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def _plot_bounds(parts: Iterable[np.ndarray]) -> Tuple[float, float, float, float]:
    xmin = float("inf")
    ymin = float("inf")
    xmax = float("-inf")
    ymax = float("-inf")
    any_found = False
    for coords in parts:
        if coords.size == 0:
            continue
        any_found = True
        xmin = min(xmin, float(np.min(coords[:, 0])))
        ymin = min(ymin, float(np.min(coords[:, 1])))
        xmax = max(xmax, float(np.max(coords[:, 0])))
        ymax = max(ymax, float(np.max(coords[:, 1])))
    if not any_found:
        return 0.0, 0.0, 1.0, 1.0
    return xmin, ymin, xmax, ymax


def _filter_to_window(parts: Sequence[np.ndarray], xmin: float, ymin: float, xmax: float, ymax: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for coords in parts:
        if coords.size == 0:
            continue
        cminx = float(np.min(coords[:, 0]))
        cmaxx = float(np.max(coords[:, 0]))
        cminy = float(np.min(coords[:, 1]))
        cmaxy = float(np.max(coords[:, 1]))
        if cmaxx < xmin or cminx > xmax or cmaxy < ymin or cminy > ymax:
            continue
        out.append(coords)
    return out


def _intersects_window(coords: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
    if coords.size == 0:
        return False
    cminx = float(np.min(coords[:, 0]))
    cmaxx = float(np.max(coords[:, 0]))
    cminy = float(np.min(coords[:, 1]))
    cmaxy = float(np.max(coords[:, 1]))
    return not (cmaxx < xmin or cminx > xmax or cmaxy < ymin or cminy > ymax)


def _set_axis_extent(ax, xmin: float, ymin: float, xmax: float, ymax: float, pad_ratio: float = 0.03) -> None:
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    padx = dx * pad_ratio
    pady = dy * pad_ratio
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _summary_landuse(labels: np.ndarray) -> Dict[str, int]:
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    return {str(uniq[i]): int(counts[i]) for i in order}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw urban city data from plot/street parquet files.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--sample_plots", type=int, default=5000)
    parser.add_argument("--sample_streets", type=int, default=80000)
    parser.add_argument("--zoom_ratio", type=float, default=0.18, help="Fraction of city extent for zoom window.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    city = args.city.strip()
    urban_root = Path(args.urban_root).resolve()
    city_dir = urban_root / city
    plot_fp = city_dir / "plot.parquet"
    street_fp = city_dir / "street.parquet"
    plot_street_fp = city_dir / "plot_street_id.npy"
    plot_bid_fp = city_dir / "plot_bid.npy"

    for fp in (plot_fp, street_fp, plot_street_fp, plot_bid_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    plot_df = pd.read_parquet(plot_fp)
    street_df = pd.read_parquet(street_fp)
    n_plots = len(plot_df)
    n_streets = len(street_df)
    if n_plots == 0 or n_streets == 0:
        raise ValueError(f"Empty city data: n_plots={n_plots}, n_streets={n_streets}")

    landuse_labels, landuse_cols = _dominant_landuse(plot_df)
    plot_sample_idx = _sample_indices(n_plots, int(args.sample_plots), int(args.seed))
    street_sample_idx = _sample_indices(n_streets, int(args.sample_streets), int(args.seed) + 1)

    plot_polys: List[np.ndarray] = []
    plot_colors: List[str] = []
    plot_centers: List[Tuple[float, float]] = []
    for idx in plot_sample_idx.tolist():
        g = plot_df.iloc[idx]["geometry"]
        try:
            outers = _extract_polygon_outers(g)
        except Exception:
            continue
        if not outers:
            continue
        outer = max(outers, key=lambda x: len(x))
        plot_polys.append(outer)
        plot_colors.append(LAND_USE_PALETTE.get(str(landuse_labels[idx]), LAND_USE_PALETTE["unknown"]))
        plot_centers.append((float(np.mean(outer[:, 0])), float(np.mean(outer[:, 1]))))

    street_lines: List[np.ndarray] = []
    for idx in street_sample_idx.tolist():
        g = street_df.iloc[idx]["geometry"]
        try:
            parts = _extract_lines(g)
        except Exception:
            continue
        street_lines.extend(parts)

    if not plot_polys or not street_lines:
        raise RuntimeError("Failed to decode enough plot/street geometries for visualization.")

    city_xmin, city_ymin, city_xmax, city_ymax = _plot_bounds(list(plot_polys) + list(street_lines))
    centers = np.array(plot_centers, dtype=np.float64)
    center_x = float(np.median(centers[:, 0]))
    center_y = float(np.median(centers[:, 1]))
    extent = max(city_xmax - city_xmin, city_ymax - city_ymin)
    zoom_half = max(extent * float(args.zoom_ratio) * 0.5, 1e-4)
    zoom_xmin = center_x - zoom_half
    zoom_xmax = center_x + zoom_half
    zoom_ymin = center_y - zoom_half
    zoom_ymax = center_y + zoom_half

    zoom_plot_pairs = [
        (poly, color)
        for poly, color in zip(plot_polys, plot_colors)
        if _intersects_window(poly, zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax)
    ]
    zoom_plot_polys = [poly for poly, _ in zoom_plot_pairs]
    zoom_streets = _filter_to_window(street_lines, zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax)
    zoom_plot_colors = [color for _, color in zoom_plot_pairs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
    street_style = dict(colors="#2f3640", linewidths=0.16, alpha=0.28)
    street_zoom_style = dict(colors="#2f3640", linewidths=0.28, alpha=0.35)

    ax0, ax1, ax2 = axes
    ax0.add_collection(LineCollection(street_lines, **street_style))
    _set_axis_extent(ax0, city_xmin, city_ymin, city_xmax, city_ymax)
    ax0.set_title(f"{city.title()} streets\nraw street.parquet sample", fontsize=11)

    ax1.add_collection(LineCollection(street_lines, **street_style))
    ax1.add_collection(
        PolyCollection(plot_polys, facecolors=plot_colors, edgecolors="none", alpha=0.50)
    )
    _set_axis_extent(ax1, city_xmin, city_ymin, city_xmax, city_ymax)
    ax1.set_title(f"{city.title()} plots\nraw plot.parquet sample", fontsize=11)

    ax2.add_collection(LineCollection(zoom_streets, **street_zoom_style))
    if zoom_plot_polys:
        ax2.add_collection(
            PolyCollection(zoom_plot_polys, facecolors=zoom_plot_colors, edgecolors="white", linewidths=0.04, alpha=0.68)
        )
    _set_axis_extent(ax2, zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax, pad_ratio=0.01)
    ax2.set_title("Local zoom\nplots colored by dominant land-use score", fontsize=11)

    legend_labels = [x for x in LAND_USE_CANDIDATES if x in landuse_cols]
    if "unknown" in set(landuse_labels.tolist()):
        legend_labels.append("unknown")
    handles = [Patch(facecolor=LAND_USE_PALETTE[k], label=k) for k in legend_labels]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, max(1, len(handles))), frameon=False, bbox_to_anchor=(0.5, -0.01))

    info_text = (
        f"plot rows={n_plots:,}  street rows={n_streets:,}\n"
        f"plot cols={len(plot_df.columns)}  street cols={len(street_df.columns)}\n"
        f"sampled plots={len(plot_polys):,}  sampled streets={len(street_lines):,}"
    )
    fig.text(0.012, 0.02, info_text, ha="left", va="bottom", fontsize=9, color="#303030")
    fig.suptitle("Raw Urban City Data Overview", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))

    out_png = Path(args.out_png).resolve() if args.out_png.strip() else Path(f"results/raw_maps/{city}_raw_overview.png").resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    plot_street = np.load(plot_street_fp, allow_pickle=True)
    plot_bid = np.load(plot_bid_fp, allow_pickle=True)
    summary = {
        "city": city,
        "plot_shape": [int(n_plots), int(len(plot_df.columns))],
        "street_shape": [int(n_streets), int(len(street_df.columns))],
        "plot_columns": list(plot_df.columns),
        "street_columns": list(street_df.columns),
        "landuse_columns": landuse_cols,
        "dominant_landuse_counts": _summary_landuse(landuse_labels),
        "plot_street_membership_stats": _membership_stats(plot_street),
        "plot_bid_membership_stats": _membership_stats(plot_bid),
        "sampled_plot_count": int(len(plot_polys)),
        "sampled_street_count": int(len(street_lines)),
        "city_bounds_lonlat": {
            "xmin": float(city_xmin),
            "ymin": float(city_ymin),
            "xmax": float(city_xmax),
            "ymax": float(city_ymax),
        },
        "zoom_bounds_lonlat": {
            "xmin": float(zoom_xmin),
            "ymin": float(zoom_ymin),
            "xmax": float(zoom_xmax),
            "ymax": float(zoom_ymax),
        },
        "out_png": str(out_png),
    }
    out_json = Path(args.out_json).resolve() if args.out_json.strip() else Path(f"results/raw_maps/{city}_raw_overview_summary.json").resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[ok] raw urban overview generated")
    print(f" city={city}")
    print(f" out_png={out_png}")
    print(f" out_json={out_json}")
    print(f" n_plots={n_plots}, n_streets={n_streets}")
    print(f" sampled_plot_polys={len(plot_polys)}, sampled_street_parts={len(street_lines)}")


if __name__ == "__main__":
    main()
