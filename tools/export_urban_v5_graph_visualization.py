#!/usr/bin/env python3
"""Export V5 graph map layers for R visualization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import LineString, mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_urban_data_paper_pipeline import _assign_tiles, _node_context  # noqa: E402
from urban_factor_experiment_utils import city_label, city_to_dataset, ensure_dir, load_sparse_graph, undirected_unique  # noqa: E402


DEFAULT_DATA_ROOT = "/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"
DEFAULT_URBAN_ROOT = "/mnt/e/urban network datasets"


def _parse_csv(text: str) -> list[str]:
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def _loads_geom(raw: Any):
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = bytes.fromhex(raw)
        return wkb.loads(bytes(raw))
    except Exception:
        return None


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )


def _feature_idx(names: list[str], name: str) -> int | None:
    try:
        return names.index(name)
    except ValueError:
        return None


def _edge_values(edge_attr: np.ndarray, names: list[str], name: str, default: float = 0.0) -> np.ndarray:
    idx = _feature_idx(names, name)
    if idx is None:
        return np.full((edge_attr.shape[0],), float(default), dtype=np.float32)
    out = edge_attr[:, idx].astype(np.float32, copy=False)
    out[~np.isfinite(out)] = float(default)
    return out


def _edge_frame(graph, node_df: pd.DataFrame) -> pd.DataFrame:
    edge_index_u, edge_attr_u, edge_weight_u = undirected_unique(
        graph.edge_index,
        graph.edge_attr,
        graph.edge_weight,
        num_nodes=graph.x.shape[0],
    )
    src = edge_index_u[0].astype(np.int64, copy=False)
    dst = edge_index_u[1].astype(np.int64, copy=False)
    names = list(graph.feature_names)
    out = pd.DataFrame(
        {
            "edge_id": np.arange(src.size, dtype=np.int64),
            "src": src,
            "dst": dst,
            "src_lon": node_df.loc[src, "lon"].to_numpy(dtype=float),
            "src_lat": node_df.loc[src, "lat"].to_numpy(dtype=float),
            "dst_lon": node_df.loc[dst, "lon"].to_numpy(dtype=float),
            "dst_lat": node_df.loc[dst, "lat"].to_numpy(dtype=float),
            "src_tile_id": node_df.loc[src, "tile_id"].to_numpy(),
            "dst_tile_id": node_df.loc[dst, "tile_id"].to_numpy(),
            "src_tile_x": node_df.loc[src, "tile_x"].to_numpy(dtype=int),
            "src_tile_y": node_df.loc[src, "tile_y"].to_numpy(dtype=int),
            "dst_tile_x": node_df.loc[dst, "tile_x"].to_numpy(dtype=int),
            "dst_tile_y": node_df.loc[dst, "tile_y"].to_numpy(dtype=int),
            "is_street_backed": _edge_values(edge_attr_u, names, "is_street_backed"),
            "is_junction_only": _edge_values(edge_attr_u, names, "is_junction_only"),
            "is_geom_fallback": _edge_values(edge_attr_u, names, "is_geom_fallback"),
            "is_shared_building": _edge_values(edge_attr_u, names, "is_shared_building"),
            "is_street_intersection_path": _edge_values(edge_attr_u, names, "is_street_intersection_path"),
            "is_building_neighbor": _edge_values(edge_attr_u, names, "is_building_neighbor"),
            "native_relation_count": _edge_values(edge_attr_u, names, "native_relation_count"),
            "native_edge_confidence": _edge_values(edge_attr_u, names, "native_edge_confidence", default=np.nan),
            "centroid_dist_m": _edge_values(edge_attr_u, names, "centroid_dist_m", default=np.nan),
            "edge_weight": edge_weight_u if edge_weight_u is not None else np.nan,
        }
    )
    out["finite_coords"] = np.isfinite(out[["src_lon", "src_lat", "dst_lon", "dst_lat"]]).all(axis=1)
    out["relation"] = out.apply(_primary_relation, axis=1)
    return out


def _primary_relation(row: pd.Series) -> str:
    if float(row.get("is_geom_fallback", 0.0)) > 0:
        return "geometry fallback"
    if float(row.get("is_building_neighbor", row.get("is_shared_building", 0.0))) > 0:
        return "building-neighbor"
    if float(row.get("is_street_backed", 0.0)) > 0:
        return "street-adjacency"
    if float(row.get("is_street_intersection_path", row.get("is_junction_only", 0.0))) > 0:
        return "junction-path"
    return "other"


def _sample_edges(df: pd.DataFrame, max_per_relation: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for relation, rdf in df[df["finite_coords"]].groupby("relation", sort=False):
        if len(rdf) > int(max_per_relation):
            rdf = rdf.iloc[np.sort(rng.choice(len(rdf), size=int(max_per_relation), replace=False))]
        rows.append(rdf)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _select_target_tile(scores: pd.DataFrame | None, node_df: pd.DataFrame, city: str) -> tuple[int, int, str, str]:
    if scores is not None and not scores.empty:
        cdf = scores[scores["city"] == city].copy()
        if not cdf.empty and {"city_anomaly_rank", "tile_x", "tile_y", "tile_id"}.issubset(cdf.columns):
            row = cdf.sort_values(["city_anomaly_rank"]).iloc[0]
            return int(row["tile_x"]), int(row["tile_y"]), str(row["tile_id"]), "top_v5_anomaly_tile"

    valid = node_df[node_df["tile_id"] != "invalid"].copy()
    if valid.empty:
        raise ValueError(f"No valid tiles for city={city}")
    tile_counts = valid.groupby(["tile_x", "tile_y", "tile_id"]).size().reset_index(name="n")
    row = tile_counts.sort_values("n", ascending=False).iloc[0]
    return int(row["tile_x"]), int(row["tile_y"]), str(row["tile_id"]), "densest_tile"


def _bounds_from_features(features: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    if not features:
        return None
    geom = shape(features[0]["geometry"])
    xmin, ymin, xmax, ymax = geom.bounds
    for f in features[1:]:
        bxmin, bymin, bxmax, bymax = shape(f["geometry"]).bounds
        xmin, ymin, xmax, ymax = min(xmin, bxmin), min(ymin, bymin), max(xmax, bxmax), max(ymax, bymax)
    dx = max((xmax - xmin) * 0.08, 0.002)
    dy = max((ymax - ymin) * 0.08, 0.002)
    return xmin - dx, ymin - dy, xmax + dx, ymax + dy


def _export_local_layers(
    city: str,
    graph,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    urban_root: Path,
    scores: pd.DataFrame | None,
    window_tiles: int,
    max_local_edges: int,
    max_streets: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    target_x, target_y, target_tile, target_reason = _select_target_tile(scores, node_df, city)
    keep = (
        (node_df["tile_x"] >= target_x - int(window_tiles))
        & (node_df["tile_x"] <= target_x + int(window_tiles))
        & (node_df["tile_y"] >= target_y - int(window_tiles))
        & (node_df["tile_y"] <= target_y + int(window_tiles))
        & (node_df["tile_id"] != "invalid")
        & np.isfinite(node_df["lon"].to_numpy(dtype=float))
        & np.isfinite(node_df["lat"].to_numpy(dtype=float))
    )
    local_nodes = node_df[keep].copy()
    local_ids = set(local_nodes["node_idx"].astype(int).tolist())
    local_edges = edge_df[
        edge_df["src"].astype(int).isin(local_ids) & edge_df["dst"].astype(int).isin(local_ids) & edge_df["finite_coords"]
    ].copy()
    if len(local_edges) > int(max_local_edges):
        rel_order = {"mixed native": 0, "building-neighbor": 1, "street-adjacency": 2, "junction-path": 3, "other": 4}
        local_edges["_rel_order"] = local_edges["relation"].map(rel_order).fillna(5)
        parts = []
        rels = local_edges.sort_values("_rel_order")["relation"].drop_duplicates().tolist()
        per_rel = max(1, int(max_local_edges) // max(len(rels), 1))
        selected: set[int] = set()
        for rel in rels:
            part = (
                local_edges[local_edges["relation"] == rel]
                .sort_values(["centroid_dist_m", "native_edge_confidence"], ascending=[True, False])
                .head(per_rel)
            )
            parts.append(part)
            selected.update(part["edge_id"].astype(int).tolist())
        local_edges = pd.concat(parts, ignore_index=True) if parts else local_edges.head(0)
        if len(local_edges) < int(max_local_edges):
            fill = (
                edge_df[
                    edge_df["src"].astype(int).isin(local_ids)
                    & edge_df["dst"].astype(int).isin(local_ids)
                    & edge_df["finite_coords"]
                    & ~edge_df["edge_id"].astype(int).isin(selected)
                ]
                .assign(_rel_order=lambda d: d["relation"].map(rel_order).fillna(5))
                .sort_values(["_rel_order", "centroid_dist_m"], ascending=[True, True])
                .head(int(max_local_edges) - len(local_edges))
            )
            if not fill.empty:
                local_edges = pd.concat([local_edges, fill], ignore_index=True)

    plot_df = pd.read_parquet(urban_root / city / "plot.parquet")
    plot_features: list[dict[str, Any]] = []
    for row in local_nodes.itertuples(index=False):
        geom = _loads_geom(plot_df.iloc[int(row.node_idx)].get("geometry"))
        if geom is None or geom.is_empty:
            continue
        plot_features.append(
            {
                "type": "Feature",
                "properties": {
                    "city": city,
                    "city_label": city_label(city),
                    "node_idx": int(row.node_idx),
                    "tile_id": str(row.tile_id),
                    "is_target_tile": str(row.tile_id) == target_tile,
                    "plot_area": float(row.plot_area) if np.isfinite(float(row.plot_area)) else None,
                },
                "geometry": mapping(geom),
            }
        )

    bbox = _bounds_from_features(plot_features)
    street_features: list[dict[str, Any]] = []
    street_path = urban_root / city / "street.parquet"
    if bbox is not None and street_path.exists():
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
                    "properties": {
                        "city": city,
                        "city_label": city_label(city),
                        "street_id": int(row.get("street_id", idx)),
                        "length": float(row.get("length", 0.0)),
                    },
                    "geometry": mapping(geom),
                }
            )
            if len(street_features) >= int(max_streets):
                break

    edge_features: list[dict[str, Any]] = []
    for row in local_edges.itertuples(index=False):
        edge_features.append(
            {
                "type": "Feature",
                "properties": {
                    "city": city,
                    "city_label": city_label(city),
                    "src": int(row.src),
                    "dst": int(row.dst),
                    "relation": str(row.relation),
                    "native_relation_count": float(row.native_relation_count),
                    "native_edge_confidence": float(row.native_edge_confidence) if np.isfinite(float(row.native_edge_confidence)) else None,
                    "centroid_dist_m": float(row.centroid_dist_m) if np.isfinite(float(row.centroid_dist_m)) else None,
                },
                "geometry": mapping(
                    LineString(
                        [
                            (float(row.src_lon), float(row.src_lat)),
                            (float(row.dst_lon), float(row.dst_lat)),
                        ]
                    )
                ),
            }
        )

    local_nodes_out = local_nodes[
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
    local_nodes_out["city_label"] = city_label(city)
    local_nodes_out["is_target_tile"] = local_nodes_out["tile_id"].astype(str) == target_tile
    meta = {
        "city": city,
        "city_label": city_label(city),
        "target_tile": target_tile,
        "target_tile_x": target_x,
        "target_tile_y": target_y,
        "target_reason": target_reason,
        "window_tiles": int(window_tiles),
        "n_local_nodes": int(len(local_nodes_out)),
        "n_local_edges": int(len(local_edges)),
        "local_relation_counts": local_edges["relation"].value_counts().to_dict(),
    }
    return plot_features, edge_features, street_features, local_nodes_out, meta


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    data_root = Path(args.data_root)
    urban_root = Path(args.urban_root)
    cities = _parse_csv(args.cities)
    rng = np.random.default_rng(int(args.seed))
    scores_path = Path(args.results_root) / "tile_anomaly_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None

    global_edges: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    local_plot_features: list[dict[str, Any]] = []
    local_edge_features: list[dict[str, Any]] = []
    local_street_features: list[dict[str, Any]] = []
    local_node_frames: list[pd.DataFrame] = []
    local_meta: list[dict[str, Any]] = []

    for city in cities:
        dataset = city_to_dataset(city, "v5")
        graph = load_sparse_graph(data_root, dataset)
        node_df = _assign_tiles(_node_context(data_root, urban_root, city, graph), float(args.tile_size_m))
        edge_df = _edge_frame(graph, node_df)
        sampled = _sample_edges(edge_df.assign(city=city, city_label=city_label(city), dataset=dataset), int(args.max_per_relation), rng)
        global_edges.append(sampled)

        total_edges = max(int(len(edge_df)), 1)
        for relation, count in edge_df["relation"].value_counts().items():
            summary_rows.append(
                {
                    "city": city,
                    "city_label": city_label(city),
                    "dataset": dataset,
                    "relation": relation,
                    "n_edges": int(count),
                    "edge_share": float(count / total_edges),
                }
            )
        if not (edge_df["is_geom_fallback"] > 0).any():
            summary_rows.append(
                {
                    "city": city,
                    "city_label": city_label(city),
                    "dataset": dataset,
                    "relation": "geometry fallback",
                    "n_edges": 0,
                    "edge_share": 0.0,
                }
            )

        plots, edges, streets, nodes, meta = _export_local_layers(
            city=city,
            graph=graph,
            node_df=node_df,
            edge_df=edge_df,
            urban_root=urban_root,
            scores=scores,
            window_tiles=int(args.window_tiles),
            max_local_edges=int(args.max_local_edges),
            max_streets=int(args.max_streets),
        )
        local_plot_features.extend(plots)
        local_edge_features.extend(edges)
        local_street_features.extend(streets)
        local_node_frames.append(nodes)
        local_meta.append(meta)

    global_df = pd.concat(global_edges, ignore_index=True) if global_edges else pd.DataFrame()
    global_df.to_csv(out_dir / "v5_global_edge_samples.csv", index=False)
    _write_csv(out_dir / "v5_city_relation_summary.csv", summary_rows)
    _write_geojson(out_dir / "v5_local_plots.geojson", local_plot_features)
    _write_geojson(out_dir / "v5_local_edges.geojson", local_edge_features)
    _write_geojson(out_dir / "v5_local_streets.geojson", local_street_features)
    pd.concat(local_node_frames, ignore_index=True).to_csv(out_dir / "v5_local_nodes.csv", index=False)
    pd.DataFrame(local_meta).to_csv(out_dir / "v5_local_manifest.csv", index=False)
    manifest = {
        "cities": cities,
        "data_root": str(data_root),
        "urban_root": str(urban_root),
        "results_root": str(args.results_root),
        "n_global_edge_samples": int(len(global_df)),
        "n_local_plot_features": int(len(local_plot_features)),
        "n_local_edge_features": int(len(local_edge_features)),
        "n_local_street_features": int(len(local_street_features)),
        "local_examples": local_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V5 graph layers for visualization.")
    parser.add_argument("--cities", type=str, default="singapore,washingtondc,shanghai")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--urban_root", type=str, default=DEFAULT_URBAN_ROOT)
    parser.add_argument("--results_root", type=str, default="results/urban_data_paper_v5_tri_city")
    parser.add_argument("--out_dir", type=str, default="results/urban_v5_graph_visualization")
    parser.add_argument("--tile_size_m", type=float, default=1000.0)
    parser.add_argument("--window_tiles", type=int, default=2)
    parser.add_argument("--max_per_relation", type=int, default=1200)
    parser.add_argument("--max_local_edges", type=int, default=700)
    parser.add_argument("--max_streets", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
