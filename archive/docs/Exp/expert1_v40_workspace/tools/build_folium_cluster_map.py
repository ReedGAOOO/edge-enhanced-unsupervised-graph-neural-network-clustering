#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]


def _import_geo_libs():
    try:
        import folium
        from folium.plugins import FastMarkerCluster
        from shapely import wkb
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies for map visualization. "
            "Install with:\n"
            "  /home/aitx/miniconda3/envs/gnn/bin/pip install folium shapely"
        ) from exc
    return folium, FastMarkerCluster, wkb


def _load_labels(
    n_nodes: int,
    labels_npy: Path | None,
    labels_csv: Path | None,
    processed_dataset_dir: Path | None,
) -> np.ndarray:
    if labels_npy is not None:
        arr = np.load(labels_npy, allow_pickle=False).reshape(-1)
        if len(arr) != n_nodes:
            raise ValueError(f"labels_npy length mismatch: {len(arr)} vs n_nodes={n_nodes}")
        return arr.astype(np.int64, copy=False)

    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
        col = None
        for cand in ("cluster", "label", "pred", "y"):
            if cand in df.columns:
                col = cand
                break
        if col is None:
            raise ValueError(f"No label-like column in {labels_csv}. Expected one of cluster/label/pred/y.")
        arr = df[col].to_numpy()
        if len(arr) != n_nodes:
            raise ValueError(f"labels_csv length mismatch: {len(arr)} vs n_nodes={n_nodes}")
        return arr.astype(np.int64, copy=False)

    if processed_dataset_dir is None:
        raise ValueError("No labels provided. Use --labels_npy/--labels_csv or --processed_dataset_dir.")
    ds_name = processed_dataset_dir.name
    fallback = processed_dataset_dir / f"{ds_name}_label.npy"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Default label file not found: {fallback}. "
            "Please provide --labels_npy with model predicted clusters."
        )
    arr = np.load(fallback, allow_pickle=False).reshape(-1)
    if len(arr) != n_nodes:
        raise ValueError(f"default label length mismatch: {len(arr)} vs n_nodes={n_nodes}")
    return arr.astype(np.int64, copy=False)


def _extract_centroids_wkb(geometry_col: pd.Series, wkb_lib, sample_idx: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_idx is None:
        sample_idx = np.arange(len(geometry_col), dtype=np.int64)
    lats: List[float] = []
    lons: List[float] = []
    keep: List[int] = []
    for idx in sample_idx.tolist():
        g = geometry_col.iloc[idx]
        if g is None:
            continue
        try:
            geom = wkb_lib.loads(bytes(g))
            c = geom.centroid
            lon = float(c.x)
            lat = float(c.y)
            if np.isfinite(lat) and np.isfinite(lon):
                lats.append(lat)
                lons.append(lon)
                keep.append(int(idx))
        except Exception:
            continue
    return np.array(keep, dtype=np.int64), np.array(lats, dtype=np.float64), np.array(lons, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Folium map for urban clustering points.")
    parser.add_argument("--city", type=str, required=True, help="City name under urban_root, e.g. beijing")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--processed_dataset_dir", type=str, default="", help="e.g. data/urban_beijing_plot")
    parser.add_argument("--labels_npy", type=str, default="", help="Predicted cluster labels .npy (1D, length=N)")
    parser.add_argument("--labels_csv", type=str, default="", help="CSV labels with one column in cluster/label/pred/y")
    parser.add_argument("--sample_size", type=int, default=12000, help="Number of points to draw; <=0 means all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point_radius", type=float, default=2.8)
    parser.add_argument("--point_opacity", type=float, default=0.75)
    parser.add_argument("--use_fast_cluster", action="store_true", help="Use FastMarkerCluster for very large point sets.")
    parser.add_argument("--out_html", type=str, default="", help="Output html path")
    parser.add_argument("--out_csv", type=str, default="", help="Optional export of mapped points csv")
    args = parser.parse_args()

    folium, FastMarkerCluster, wkb_lib = _import_geo_libs()

    urban_root = Path(args.urban_root).resolve()
    city = args.city.strip()
    plot_fp = urban_root / city / "plot.parquet"
    if not plot_fp.exists():
        raise FileNotFoundError(f"plot.parquet not found: {plot_fp}")

    plot_df = pd.read_parquet(plot_fp, columns=["plot_id", "geometry"])
    n_nodes = len(plot_df)
    if n_nodes == 0:
        raise ValueError(f"No rows in {plot_fp}")

    processed_dir = Path(args.processed_dataset_dir).resolve() if args.processed_dataset_dir.strip() else None
    labels_npy = Path(args.labels_npy).resolve() if args.labels_npy.strip() else None
    labels_csv = Path(args.labels_csv).resolve() if args.labels_csv.strip() else None
    labels = _load_labels(n_nodes=n_nodes, labels_npy=labels_npy, labels_csv=labels_csv, processed_dataset_dir=processed_dir)

    if args.sample_size > 0 and args.sample_size < n_nodes:
        rng = np.random.default_rng(int(args.seed))
        sample_idx = np.sort(rng.choice(n_nodes, size=args.sample_size, replace=False))
    else:
        sample_idx = np.arange(n_nodes, dtype=np.int64)

    keep_idx, lats, lons = _extract_centroids_wkb(plot_df["geometry"], wkb_lib=wkb_lib, sample_idx=sample_idx)
    if len(keep_idx) == 0:
        raise RuntimeError("No valid geometries decoded to centroids.")
    labels_kept = labels[keep_idx]

    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron", control_scale=True)

    uniq = np.unique(labels_kept)
    color_map = {int(c): PALETTE[i % len(PALETTE)] for i, c in enumerate(uniq.tolist())}

    if args.use_fast_cluster:
        points = [[float(la), float(lo)] for la, lo in zip(lats, lons)]
        FastMarkerCluster(points).add_to(fmap)
    else:
        fg = folium.FeatureGroup(name="clusters", show=True)
        for lat, lon, c in zip(lats, lons, labels_kept):
            cc = int(c)
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=float(args.point_radius),
                color=color_map[cc],
                fill=True,
                fill_color=color_map[cc],
                fill_opacity=float(args.point_opacity),
                opacity=float(args.point_opacity),
                weight=0,
                tooltip=f"cluster={cc}",
            ).add_to(fg)
        fg.add_to(fmap)
        legend_html = [
            '<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; '
            'background: white; border: 1px solid #ccc; padding: 10px; font-size: 12px;">',
            "<b>Clusters</b><br>",
        ]
        for c in uniq.tolist()[:40]:
            legend_html.append(
                f'<span style="display:inline-block;width:10px;height:10px;background:{color_map[int(c)]};'
                f'margin-right:6px;"></span>{int(c)}<br>'
            )
        if len(uniq) > 40:
            legend_html.append(f"... total {len(uniq)} clusters")
        legend_html.append("</div>")
        fmap.get_root().html.add_child(folium.Element("".join(legend_html)))

    folium.LayerControl(collapsed=False).add_to(fmap)

    if args.out_html.strip():
        out_html = Path(args.out_html).resolve()
    else:
        out_html = Path(f"results/maps/{city}_clusters_map.html").resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))

    if args.out_csv.strip():
        out_csv = Path(args.out_csv).resolve()
    else:
        out_csv = Path(f"results/maps/{city}_clusters_points.csv").resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "node_idx": keep_idx,
            "plot_id": plot_df.iloc[keep_idx]["plot_id"].to_numpy(),
            "lat": lats,
            "lon": lons,
            "cluster": labels_kept,
        }
    )
    out_df.to_csv(out_csv, index=False)

    print("[ok] map generated")
    print(f" city={city}")
    print(f" n_total={n_nodes}, n_mapped={len(keep_idx)}, n_clusters={len(uniq)}")
    print(f" html={out_html}")
    print(f" points_csv={out_csv}")


if __name__ == "__main__":
    main()

