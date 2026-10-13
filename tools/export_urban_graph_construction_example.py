#!/usr/bin/env python3
"""Export a local Urban graph construction example for R map plotting.

The exported files are intentionally simple GeoJSON/CSV artifacts so that the
final figure can be drawn with R packages such as sf and ggplot2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import LineString, mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_urban_data_paper_pipeline import _assign_tiles, _edge_context, _node_context  # noqa: E402
from urban_factor_experiment_utils import city_label, city_to_dataset, ensure_dir, load_sparse_graph  # noqa: E402


DEFAULT_BASE_GRAPH_ROOT = "/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"
DEFAULT_URBAN_ROOT = "/mnt/e/urban network datasets"


def _loads_geom(raw: Any):
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = bytes.fromhex(raw)
        return wkb.loads(bytes(raw))
    except Exception:
        return None


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(_feature_collection(features), ensure_ascii=False), encoding="utf-8")


def _select_target_tile(args: argparse.Namespace, scores_path: Path) -> tuple[int, int, str]:
    scores = pd.read_csv(scores_path)
    city_scores = scores[scores["city"] == args.city].copy()
    if city_scores.empty:
        raise ValueError(f"No scores found for city={args.city} in {scores_path}")
    if args.target_tile:
        row = city_scores[city_scores["tile_id"].astype(str) == str(args.target_tile)]
        if row.empty:
            raise ValueError(f"target_tile={args.target_tile} not found for city={args.city}")
        r = row.iloc[0]
    else:
        top = city_scores[city_scores["city_anomaly_rank"] <= int(args.top_n)].copy()
        if top.empty:
            top = city_scores.copy()
        r = top.sort_values(["geom_edge_share", "combined_anomaly_score"], ascending=False).iloc[0]
    return int(r["tile_x"]), int(r["tile_y"]), str(r["tile_id"])


def _edge_relation(row: pd.Series) -> str:
    if float(row.get("is_geom_fallback", 0.0)) > 0:
        return "geometry fallback"
    if float(row.get("is_street_backed", 0.0)) > 0:
        return "shared street"
    if float(row.get("is_junction_only", 0.0)) > 0:
        return "same junction"
    if float(row.get("is_shared_building", 0.0)) > 0:
        return "shared building attribute"
    return "other"


def export(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    base_root = Path(args.base_graph_root)
    urban_root = Path(args.urban_root)
    dataset = city_to_dataset(args.city, "v3bsjg")
    graph = load_sparse_graph(base_root, dataset)
    node_df = _assign_tiles(_node_context(base_root, urban_root, args.city, graph), float(args.tile_size_m))
    edge_df = _edge_context(graph, node_df)

    target_x, target_y, target_tile = _select_target_tile(args, Path(args.results_root) / "tile_anomaly_scores.csv")
    window = int(args.window_tiles)
    node_keep = (
        (node_df["tile_x"] >= target_x - window)
        & (node_df["tile_x"] <= target_x + window)
        & (node_df["tile_y"] >= target_y - window)
        & (node_df["tile_y"] <= target_y + window)
        & (node_df["tile_id"] != "invalid")
        & np.isfinite(node_df["lon"].to_numpy(dtype=float))
        & np.isfinite(node_df["lat"].to_numpy(dtype=float))
        & ~((node_df["lon"].abs() < 1e-8) & (node_df["lat"].abs() < 1e-8))
    )
    local_nodes = node_df[node_keep].copy()
    if local_nodes.empty:
        raise ValueError("No local nodes selected.")
    local_ids = set(local_nodes["node_idx"].astype(int).tolist())

    local_edges = edge_df[
        edge_df["src"].astype(int).isin(local_ids) & edge_df["dst"].astype(int).isin(local_ids)
    ].copy()
    if len(local_edges) > int(args.max_edges):
        local_edges = local_edges.sort_values(
            ["is_geom_fallback", "is_street_backed", "is_junction_only", "centroid_dist_m"],
            ascending=[False, False, False, True],
        ).head(int(args.max_edges))
    local_edges["relation_type"] = local_edges.apply(_edge_relation, axis=1)
    local_edges["src_lon"] = node_df.loc[local_edges["src"].to_numpy(dtype=int), "lon"].to_numpy(dtype=float)
    local_edges["src_lat"] = node_df.loc[local_edges["src"].to_numpy(dtype=int), "lat"].to_numpy(dtype=float)
    local_edges["dst_lon"] = node_df.loc[local_edges["dst"].to_numpy(dtype=int), "lon"].to_numpy(dtype=float)
    local_edges["dst_lat"] = node_df.loc[local_edges["dst"].to_numpy(dtype=int), "lat"].to_numpy(dtype=float)

    plot_df = pd.read_parquet(urban_root / args.city / "plot.parquet")
    plot_features: list[dict[str, Any]] = []
    for row in local_nodes.itertuples(index=False):
        geom = _loads_geom(plot_df.iloc[int(row.node_idx)].get("geometry"))
        if geom is None or geom.is_empty:
            continue
        plot_features.append(
            {
                "type": "Feature",
                "properties": {
                    "node_idx": int(row.node_idx),
                    "tile_id": str(row.tile_id),
                    "is_target_tile": str(row.tile_id) == target_tile,
                    "known_label": bool(row.known_label),
                    "plot_area": float(row.plot_area) if np.isfinite(float(row.plot_area)) else None,
                },
                "geometry": mapping(geom),
            }
        )

    xmin, ymin, xmax, ymax = shape(_feature_collection(plot_features)["features"][0]["geometry"]).bounds
    for f in plot_features[1:]:
        bxmin, bymin, bxmax, bymax = shape(f["geometry"]).bounds
        xmin, ymin, xmax, ymax = min(xmin, bxmin), min(ymin, bymin), max(xmax, bxmax), max(ymax, bymax)
    dx = max((xmax - xmin) * 0.08, 0.002)
    dy = max((ymax - ymin) * 0.08, 0.002)
    bbox = (xmin - dx, ymin - dy, xmax + dx, ymax + dy)

    street_features: list[dict[str, Any]] = []
    street_path = urban_root / args.city / "street.parquet"
    if street_path.exists():
        street_df = pd.read_parquet(street_path)
        for idx, row in street_df.iterrows():
            geom = _loads_geom(row.get("geometry"))
            if geom is None or geom.is_empty:
                continue
            bxmin, bymin, bxmax, bymax = geom.bounds
            if bxmax < bbox[0] or bxmin > bbox[2] or bymax < bbox[1] or bymin > bbox[3]:
                continue
            street_features.append(
                {
                    "type": "Feature",
                    "properties": {"street_id": int(row.get("street_id", idx)), "length": float(row.get("length", 0.0))},
                    "geometry": mapping(geom),
                }
            )
            if len(street_features) >= int(args.max_streets):
                break

    edge_features: list[dict[str, Any]] = []
    for row in local_edges.itertuples(index=False):
        if not all(np.isfinite([row.src_lon, row.src_lat, row.dst_lon, row.dst_lat])):
            continue
        edge_features.append(
            {
                "type": "Feature",
                "properties": {
                    "src": int(row.src),
                    "dst": int(row.dst),
                    "relation_type": str(row.relation_type),
                    "is_street_backed": float(row.is_street_backed),
                    "is_junction_only": float(row.is_junction_only),
                    "is_geom_fallback": float(row.is_geom_fallback),
                    "is_shared_building": float(row.is_shared_building),
                    "centroid_dist_m": float(row.centroid_dist_m) if np.isfinite(float(row.centroid_dist_m)) else None,
                },
                "geometry": mapping(LineString([(float(row.src_lon), float(row.src_lat)), (float(row.dst_lon), float(row.dst_lat))])),
            }
        )

    node_out = local_nodes[
        [
            "node_idx",
            "city",
            "tile_id",
            "tile_x",
            "tile_y",
            "lon",
            "lat",
            "plot_area",
            "building_count",
            "street_count",
            "known_label",
            "label",
        ]
    ].copy()
    node_out["is_target_tile"] = node_out["tile_id"].astype(str) == target_tile
    node_out.to_csv(out_dir / "construction_nodes.csv", index=False)
    local_edges[
        [
            "src",
            "dst",
            "relation_type",
            "is_street_backed",
            "is_junction_only",
            "is_geom_fallback",
            "is_shared_building",
            "centroid_dist_m",
            "src_lon",
            "src_lat",
            "dst_lon",
            "dst_lat",
        ]
    ].to_csv(out_dir / "construction_edges.csv", index=False)
    _write_geojson(out_dir / "construction_plots.geojson", plot_features)
    _write_geojson(out_dir / "construction_streets.geojson", street_features)
    _write_geojson(out_dir / "construction_edges.geojson", edge_features)
    meta = {
        "city": args.city,
        "city_label": city_label(args.city),
        "dataset": dataset,
        "target_tile": target_tile,
        "target_tile_x": target_x,
        "target_tile_y": target_y,
        "window_tiles": window,
        "n_nodes": int(len(node_out)),
        "n_edges": int(len(local_edges)),
        "n_plot_features": int(len(plot_features)),
        "n_street_features": int(len(street_features)),
        "bbox": list(map(float, bbox)),
        "relation_counts": local_edges["relation_type"].value_counts().to_dict(),
    }
    (out_dir / "construction_example_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local Urban graph construction example.")
    parser.add_argument("--city", type=str, default="singapore")
    parser.add_argument("--target_tile", type=str, default="")
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--window_tiles", type=int, default=4)
    parser.add_argument("--tile_size_m", type=float, default=1000.0)
    parser.add_argument("--max_edges", type=int, default=500)
    parser.add_argument("--max_streets", type=int, default=2000)
    parser.add_argument("--urban_root", type=str, default=DEFAULT_URBAN_ROOT)
    parser.add_argument("--base_graph_root", type=str, default=DEFAULT_BASE_GRAPH_ROOT)
    parser.add_argument("--results_root", type=str, default="results/urban_data_paper")
    parser.add_argument("--out_dir", type=str, default="results/urban_data_paper/graph_construction_mechanism")
    args = parser.parse_args()
    meta = export(args)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
