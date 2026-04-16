#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import prepare_urban_plot_graph_v3 as v3


V3_REL_COLORS = {
    "street_only": "#4c78a8",
    "street_plus_junction": "#f58518",
    "junction_only": "#54a24b",
    "geom_fallback": "#e45756",
}

V3B_REL_COLORS = {
    "street_backed": "#4c78a8",
    "junction_only": "#54a24b",
    "geom_fallback": "#e45756",
}


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _load_graph(dataset_name: str, data_root: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    base = data_root / dataset_name
    edge_index = np.load(base / f"{dataset_name}_edge_index.npy")
    edge_attr = np.load(base / f"{dataset_name}_edge_attr.npy")
    meta = json.loads((base / f"{dataset_name}_meta.json").read_text(encoding="utf-8"))
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError(f"{dataset_name}: edge_index/edge_attr shape mismatch")
    half = edge_index.shape[1] // 2
    return edge_index[:, :half].astype(np.int64), edge_attr[:half].astype(np.float32), meta


def _load_centroids(city: str, urban_root: Path) -> np.ndarray:
    plot_df = pd.read_parquet(urban_root / city / "plot.parquet")
    centroids_lonlat = v3._geometry_centroids_from_wkb(plot_df["geometry"])
    return v3._lonlat_to_local_m(centroids_lonlat)


def _dense_window(centroids: np.ndarray, bins: int = 36, window_ratio: float = 0.16) -> Tuple[float, float, float, float]:
    x = centroids[:, 0]
    y = centroids[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    ix, iy = np.unravel_index(np.argmax(hist), hist.shape)
    xc = 0.5 * (xedges[ix] + xedges[ix + 1])
    yc = 0.5 * (yedges[iy] + yedges[iy + 1])
    dx = max((xmax - xmin) * window_ratio, 1.0)
    dy = max((ymax - ymin) * window_ratio, 1.0)
    return xc - dx, yc - dy, xc + dx, yc + dy


def _edges_in_window(src: np.ndarray, dst: np.ndarray, centroids: np.ndarray, window: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = window
    pts = centroids
    in_window = (
        (pts[:, 0] >= xmin)
        & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin)
        & (pts[:, 1] <= ymax)
    )
    return in_window[src] & in_window[dst]


def _segments(src: np.ndarray, dst: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.stack([centroids[src], centroids[dst]], axis=1).astype(np.float32)


def _set_extent(ax, window: Tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = window
    padx = (xmax - xmin) * 0.04
    pady = (ymax - ymin) * 0.04
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _v3_relation_labels(edge_attr: np.ndarray) -> np.ndarray:
    rel_street = edge_attr[:, 0] > 0
    rel_junction = edge_attr[:, 1] > 0
    rel_geom = edge_attr[:, 2] > 0
    labels = np.full(edge_attr.shape[0], "junction_only", dtype=object)
    labels[rel_street & ~rel_junction & ~rel_geom] = "street_only"
    labels[rel_street & rel_junction & ~rel_geom] = "street_plus_junction"
    labels[rel_geom] = "geom_fallback"
    return labels


def _v3b_relation_labels(edge_attr: np.ndarray) -> np.ndarray:
    is_street = edge_attr[:, 0] > 0
    is_junction_only = edge_attr[:, 1] > 0
    is_geom = edge_attr[:, 2] > 0
    labels = np.full(edge_attr.shape[0], "junction_only", dtype=object)
    labels[is_street] = "street_backed"
    labels[is_geom] = "geom_fallback"
    labels[is_junction_only] = "junction_only"
    return labels


def _plot_local(ax, centroids: np.ndarray, src: np.ndarray, dst: np.ndarray, labels: np.ndarray, window: Tuple[float, float, float, float], palette: Dict[str, str], title: str) -> None:
    mask = _edges_in_window(src, dst, centroids, window)
    src_l = src[mask]
    dst_l = dst[mask]
    labels_l = labels[mask]
    node_mask = (
        (centroids[:, 0] >= window[0])
        & (centroids[:, 0] <= window[2])
        & (centroids[:, 1] >= window[1])
        & (centroids[:, 1] <= window[3])
    )
    pts = centroids[node_mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=2.0, c="#d3d3d3", alpha=0.45, linewidths=0)
    for name, color in palette.items():
        take = labels_l == name
        if not np.any(take):
            continue
        segs = _segments(src_l[take], dst_l[take], centroids)
        lc = LineCollection(segs, colors=color, linewidths=0.5, alpha=0.85)
        ax.add_collection(lc)
    ax.set_title(title, fontsize=11)
    _set_extent(ax, window)


def _nonzero_rate(edge_attr: np.ndarray, idx: int) -> float:
    return float(np.mean(edge_attr[:, idx] > 0))


def _sample_corr(edge_attr: np.ndarray, max_rows: int = 120_000) -> np.ndarray:
    if edge_attr.shape[0] > max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(edge_attr.shape[0], size=max_rows, replace=False))
        x = edge_attr[idx]
    else:
        x = edge_attr
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    keep = (std > 1e-12).reshape(-1)
    corr = np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)
    if np.any(keep):
        xk = x[:, keep] / std[:, keep]
        corr_k = (xk.T @ xk) / max(xk.shape[0], 1)
        corr[np.ix_(keep, keep)] = np.asarray(corr_k, dtype=np.float32)
    np.fill_diagonal(corr, 1.0)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def _heatmap(ax, corr: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    return im


def _city_summary(city: str, v3_attr: np.ndarray, v3b_attr: np.ndarray, meta_v3: Dict[str, object], meta_v3b: Dict[str, object]) -> dict:
    return {
        "city": city,
        "v3_dataset": meta_v3["dataset_name"],
        "v3b_dataset": meta_v3b["dataset_name"],
        "n_edges_undirected": int(meta_v3["n_edges_undirected"]),
        "v3_edge_dim": int(meta_v3["edge_feature_dim"]),
        "v3b_edge_dim": int(meta_v3b["edge_feature_dim"]),
        "v3_rel_shared_street_rate": _nonzero_rate(v3_attr, 0),
        "v3_rel_shared_junction_rate": _nonzero_rate(v3_attr, 1),
        "v3_rel_geom_rate": _nonzero_rate(v3_attr, 2),
        "v3_shared_building_rate": _nonzero_rate(v3_attr, 5),
        "v3b_street_backed_rate": _nonzero_rate(v3b_attr, 0),
        "v3b_junction_only_rate": _nonzero_rate(v3b_attr, 1),
        "v3b_geom_rate": _nonzero_rate(v3b_attr, 2),
        "v3b_shared_building_rate": _nonzero_rate(v3b_attr, 3),
    }


def _bar_compare(ax, city: str, v3_attr: np.ndarray, v3b_attr: np.ndarray) -> None:
    labels = [
        "street bit",
        "junction bit",
        "geom bit",
        "shared building",
    ]
    v3_vals = [
        _nonzero_rate(v3_attr, 0),
        _nonzero_rate(v3_attr, 1),
        _nonzero_rate(v3_attr, 2),
        _nonzero_rate(v3_attr, 5),
    ]
    v3b_vals = [
        _nonzero_rate(v3b_attr, 0),
        _nonzero_rate(v3b_attr, 1),
        _nonzero_rate(v3b_attr, 2),
        _nonzero_rate(v3b_attr, 3),
    ]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, v3_vals, width=w, color="#9ecae9", label="V3")
    ax.bar(x + w / 2, v3b_vals, width=w, color="#3182bd", label="V3b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Edge Ratio")
    ax.set_title(f"{city}: relation-bit activation", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual compare V3 and V3b urban graph datasets.")
    parser.add_argument("--cities", type=str, default="beijing,shanghai,paris")
    parser.add_argument("--variant_v3", type=str, default="v3sjg")
    parser.add_argument("--variant_v3b", type=str, default="v3bsjg")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_dir", type=str, default="results/urban_v3_v3b_compare")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve()
    urban_root = (repo_root / args.urban_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cities = _parse_csv(args.cities)
    for city in cities:
        v3_name = f"urban_{city}_plot_{args.variant_v3}"
        v3b_name = f"urban_{city}_plot_{args.variant_v3b}"
        edge_index_v3, edge_attr_v3, meta_v3 = _load_graph(v3_name, data_root)
        edge_index_v3b, edge_attr_v3b, meta_v3b = _load_graph(v3b_name, data_root)
        centroids = _load_centroids(city, urban_root)
        window = _dense_window(centroids)

        rows.append(_city_summary(city, edge_attr_v3, edge_attr_v3b, meta_v3, meta_v3b))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
        _plot_local(
            axes[0],
            centroids=centroids,
            src=edge_index_v3[0],
            dst=edge_index_v3[1],
            labels=_v3_relation_labels(edge_attr_v3),
            window=window,
            palette=V3_REL_COLORS,
            title=f"{city}: V3 local relations",
        )
        _plot_local(
            axes[1],
            centroids=centroids,
            src=edge_index_v3b[0],
            dst=edge_index_v3b[1],
            labels=_v3b_relation_labels(edge_attr_v3b),
            window=window,
            palette=V3B_REL_COLORS,
            title=f"{city}: V3b local relations",
        )
        _bar_compare(axes[2], city, edge_attr_v3, edge_attr_v3b)
        fig.suptitle(f"Urban Graph Semantic Check: {city}", fontsize=13)
        fig.savefig(out_dir / f"{city}_local_compare.png", dpi=220)
        plt.close(fig)

        corr_v3 = _sample_corr(edge_attr_v3)
        corr_v3b = _sample_corr(edge_attr_v3b)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        im = _heatmap(axes[0], corr_v3, list(meta_v3["edge_feature_names"]), f"{city}: V3 edge-attr correlation")
        _heatmap(axes[1], corr_v3b, list(meta_v3b["edge_feature_names"]), f"{city}: V3b edge-attr correlation")
        cbar = fig.colorbar(im, ax=axes, shrink=0.82)
        cbar.ax.set_ylabel("Pearson r", rotation=90)
        fig.savefig(out_dir / f"{city}_corr_compare.png", dpi=220)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.36
    axes[0].bar(x - w / 2, df["v3_edge_dim"], width=w, color="#9ecae9", label="V3")
    axes[0].bar(x + w / 2, df["v3b_edge_dim"], width=w, color="#3182bd", label="V3b")
    axes[0].set_title("Edge Feature Dim")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["city"], rotation=20)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - w / 2, df["v3_rel_shared_junction_rate"], width=w, color="#fdd0a2", label="V3 rel_shared_junction")
    axes[1].bar(x + w / 2, df["v3b_junction_only_rate"], width=w, color="#31a354", label="V3b is_junction_only")
    axes[1].set_title("Junction Bit Activation")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["city"], rotation=20)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x - w / 2, df["v3_rel_geom_rate"], width=w, color="#fcae91", label="V3 geom")
    axes[2].bar(x + w / 2, df["v3b_geom_rate"], width=w, color="#fb6a4a", label="V3b geom")
    axes[2].set_title("Geom Fallback Ratio")
    axes[2].set_ylim(0.0, max(0.06, float(max(df["v3_rel_geom_rate"].max(), df["v3b_geom_rate"].max()) * 1.15)))
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["city"], rotation=20)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "overview.png", dpi=220)
    plt.close(fig)

    readme = "\n".join(
        [
            "# V3 vs V3b Visual Check",
            "",
            "Files:",
            "- `overview.png`: city-level summary comparing edge feature dimensionality and key relation-bit activation rates.",
            "- `<city>_local_compare.png`: local spatial relation map for V3 and V3b plus relation-bit activation bars.",
            "- `<city>_corr_compare.png`: edge-attribute correlation heatmaps for V3 and V3b.",
            "- `summary.csv` / `summary.json`: machine-readable comparison summary.",
            "",
            "Interpretation guide:",
            "- V3 and V3b should have the same topology for the same city/variant pair.",
            "- V3b should keep `geom` sparse and reduce the saturation of the junction relation bit.",
            "- V3b edge attributes should look cleaner because symmetric endpoint context and orientation-based geometry replace several weaker or redundant fields.",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    print(f"[ok] wrote visual comparison pack to {out_dir}")


if __name__ == "__main__":
    main()
