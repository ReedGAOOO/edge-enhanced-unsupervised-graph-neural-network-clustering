#!/usr/bin/env python3
"""
Prepare a plot-centric heterogeneous urban dataset.

Design goals:
- keep real urban object types separate instead of collapsing everything to plots
- use only semantically grounded typed relations for the main graph
- retain plot labels for evaluation while keeping the graph compatible with future hetero GNN work

Output folder example:
  data/urban_beijing_plot_hetero_v1/
    meta.json
    plot_feat.npy
    plot_label.npy
    plot_node_ids.npy
    plot_node_table.csv
    street_feat.npy
    street_node_ids.npy
    street_node_table.csv
    intersection_feat.npy
    intersection_node_ids.npy
    intersection_node_table.csv
    building_feat.npy
    building_node_ids.npy
    building_node_table.csv
    edge_plot__shares_street__plot.npy
    edge_attr_plot__shares_street__plot.npy
    ...
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import prepare_urban_plot_graph_v3 as v3


NODE_TYPE_ORDER = ["plot", "street", "intersection", "building"]

PLOT_FEATURE_NAMES = [
    "log_plot_area",
    "log_plot_perimeter",
    "plot_convexity",
    "log_plot_corners",
    "plot_elongation",
    "sin_plot_orientation",
    "cos_plot_orientation",
    "plot_rectangularity",
    "log_plot_complexity",
    "canopy_mean",
    "canopy_std",
    "log_population",
    "log_population_density",
    "log_built_value",
    "log_built_density",
    "lcz_compact",
    "lcz_open",
    "lcz_sparse",
    "lcz_natural",
    "lcz_entropy",
]

STREET_FEATURE_NAMES = [
    "log_street_length",
    "endpoint_degree_sum",
    "endpoint_degree_gap",
    "endpoint_closeness_mean",
    "endpoint_betweenness_mean",
    "log_adjacent_plot_count",
]

INTERSECTION_FEATURE_NAMES = [
    "intersection_degree",
    "intersection_clustering",
    "intersection_weighted_clustering",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "katz_centrality",
    "pagerank",
]

BUILDING_FEATURE_NAMES = [
    "log_building_area",
    "log_building_perimeter",
    "building_convexity",
    "log_building_corners",
    "building_elongation",
    "sin_building_orientation",
    "cos_building_orientation",
    "building_rectangularity",
    "log_building_complexity",
    "log_building_height",
]

EDGE_ATTR_NAMES = {
    "plot__shares_street__plot": [
        "shared_street_cnt",
        "jaccard_street",
        "shared_street_len_mean",
    ],
    "plot__bounded_by__street": [
        "street_length",
    ],
    "street__rev_bounded_by__plot": [
        "street_length",
    ],
    "plot__contains__building": [
        "building_area_ratio_to_plot",
        "log_building_height",
    ],
    "building__rev_contains__plot": [
        "building_area_ratio_to_plot",
        "log_building_height",
    ],
    "building__near__building": [
        "centroid_dist_m",
        "orientation_diff_deg",
    ],
}


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return ""


def _unique_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        raise KeyError(f"Missing key column: {key}")
    out = df.copy()
    out[key] = _normalize_id_series(out[key])
    out = out.drop_duplicates(subset=[key], keep="first").reset_index(drop=True).copy()
    return out


def _safe_numeric(series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _normalize_id_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    out = np.zeros((len(values),), dtype=np.int64)
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = values[finite].astype(np.int64, copy=False)
        next_synth = int(out[finite].min()) - 1
    else:
        next_synth = -1
    for idx in np.flatnonzero(~finite):
        out[idx] = next_synth
        next_synth -= 1
    return pd.Series(out, index=series.index, dtype="int64")


def _orientation_diff_deg(a_deg: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    diff = np.abs(a_deg[src] - a_deg[dst]).astype(np.float32)
    diff = np.minimum(diff, 360.0 - diff)
    diff = np.minimum(diff, 180.0)
    return diff.astype(np.float32)


def _duplicate_directed(edge_index_undirected: np.ndarray, edge_attr_undirected: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    src = edge_index_undirected[0]
    dst = edge_index_undirected[1]
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    if edge_attr_undirected is None:
        return edge_index, None
    edge_attr = np.concatenate([edge_attr_undirected, edge_attr_undirected], axis=0).astype(np.float32)
    return edge_index, edge_attr


def _save_array(out_dir: Path, name: str, arr: np.ndarray) -> str:
    fp = out_dir / f"{name}.npy"
    np.save(fp, arr)
    return fp.name


def _resolve_membership_to_indices(
    memberships: Sequence[Sequence[int]],
    node_ids: np.ndarray,
    id_to_idx: Dict[int, int],
    relation_name: str,
) -> Tuple[List[List[int]], Dict[str, int | str]]:
    refs: List[int] = []
    for items in memberships:
        for x in items:
            try:
                refs.append(int(x))
            except Exception:
                continue
    unique_refs = sorted(set(refs))
    total = len(unique_refs)
    n = int(len(node_ids))
    id_hits = sum(1 for x in unique_refs if x in id_to_idx)
    one_based_hits = sum(1 for x in unique_refs if 1 <= x <= n)
    zero_based_hits = sum(1 for x in unique_refs if 0 <= x < n)

    mode = "empty"
    if total == 0:
        mode = "empty"
    elif id_hits == total:
        mode = "node_id"
    elif one_based_hits == total and id_hits < total:
        mode = "one_based_position"
    elif zero_based_hits == total and id_hits < total:
        mode = "zero_based_position"
    else:
        mode_scores = {
            "node_id": id_hits,
            "one_based_position": one_based_hits,
            "zero_based_position": zero_based_hits,
        }
        mode = max(mode_scores, key=mode_scores.get)

    resolved: List[List[int]] = []
    for items in memberships:
        out: List[int] = []
        for x in items:
            try:
                ref = int(x)
            except Exception:
                continue
            idx = None
            if mode == "node_id":
                idx = id_to_idx.get(ref)
            elif mode == "one_based_position":
                if 1 <= ref <= n:
                    idx = ref - 1
            elif mode == "zero_based_position":
                if 0 <= ref < n:
                    idx = ref
            if idx is not None:
                out.append(int(idx))
        resolved.append(sorted(set(out)))

    return resolved, {
        "relation_name": relation_name,
        "mode": mode,
        "unique_ref_count": int(total),
        "id_hits": int(id_hits),
        "one_based_hits": int(one_based_hits),
        "zero_based_hits": int(zero_based_hits),
        "target_node_count": int(n),
    }


def _plot_features(plot_df: pd.DataFrame) -> np.ndarray:
    population_col = _first_existing(plot_df.columns, v3.POPULATION_CANDIDATES)
    built_col = _first_existing(plot_df.columns, v3.BUILT_CANDIDATES)
    canopy_mean_col = _first_existing(plot_df.columns, v3.CANOPY_MEAN_CANDIDATES)
    canopy_std_col = _first_existing(plot_df.columns, v3.CANOPY_STD_CANDIDATES)

    area = _safe_numeric(plot_df.get("plot_area", 0.0))
    perimeter = _safe_numeric(plot_df.get("plot_perimeter", 0.0))
    convexity = _safe_numeric(plot_df.get("plot_convexity", 0.0))
    corners = _safe_numeric(plot_df.get("plot_corners", 0.0))
    elongation = _safe_numeric(plot_df.get("plot_elongation", 0.0))
    orientation = _safe_numeric(plot_df.get("plot_orientation", 0.0))
    rectangularity = _safe_numeric(plot_df.get("plot_rectangularity", 0.0))
    complexity = _safe_numeric(plot_df.get("plot_complexity", 0.0))

    canopy_mean = _safe_numeric(plot_df[canopy_mean_col], 0.0) if canopy_mean_col else np.zeros(len(plot_df), dtype=np.float32)
    canopy_std = _safe_numeric(plot_df[canopy_std_col], 0.0) if canopy_std_col else np.zeros(len(plot_df), dtype=np.float32)
    population = _safe_numeric(plot_df[population_col], 0.0) if population_col else np.zeros(len(plot_df), dtype=np.float32)
    built = _safe_numeric(plot_df[built_col], 0.0) if built_col else np.zeros(len(plot_df), dtype=np.float32)

    safe_area = np.maximum(area, 1.0)
    pop_density = population / safe_area
    built_density = built / safe_area
    lcz = v3._lcz_group_values(plot_df)
    orientation_rad = np.deg2rad(orientation.astype(np.float64))

    feature_map = {
        "log_plot_area": np.log1p(np.clip(area, a_min=0.0, a_max=None)),
        "log_plot_perimeter": np.log1p(np.clip(perimeter, a_min=0.0, a_max=None)),
        "plot_convexity": convexity,
        "log_plot_corners": np.log1p(np.clip(corners, a_min=0.0, a_max=None)),
        "plot_elongation": elongation,
        "sin_plot_orientation": np.sin(orientation_rad).astype(np.float32),
        "cos_plot_orientation": np.cos(orientation_rad).astype(np.float32),
        "plot_rectangularity": rectangularity,
        "log_plot_complexity": np.log1p(np.clip(complexity, a_min=0.0, a_max=None)),
        "canopy_mean": canopy_mean,
        "canopy_std": canopy_std,
        "log_population": np.log1p(np.clip(population, a_min=0.0, a_max=None)),
        "log_population_density": np.log1p(np.clip(pop_density, a_min=0.0, a_max=None)),
        "log_built_value": np.log1p(np.clip(built, a_min=0.0, a_max=None)),
        "log_built_density": np.log1p(np.clip(built_density, a_min=0.0, a_max=None)),
        "lcz_compact": lcz["lcz_compact"],
        "lcz_open": lcz["lcz_open"],
        "lcz_sparse": lcz["lcz_sparse"],
        "lcz_natural": lcz["lcz_natural"],
        "lcz_entropy": lcz["lcz_entropy"],
    }
    x = np.stack([feature_map[name].astype(np.float32) for name in PLOT_FEATURE_NAMES], axis=1)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _building_features(building_df: pd.DataFrame) -> np.ndarray:
    area = _safe_numeric(building_df.get("bid_area", 0.0))
    perimeter = _safe_numeric(building_df.get("bid_perimeter", 0.0))
    convexity = _safe_numeric(building_df.get("bid_convexity", 0.0))
    corners = _safe_numeric(building_df.get("bid_corners", 0.0))
    elongation = _safe_numeric(building_df.get("bid_elongation", 0.0))
    orientation = _safe_numeric(building_df.get("bid_orientation", 0.0))
    rectangularity = _safe_numeric(building_df.get("bid_rectangularity", 0.0))
    complexity = _safe_numeric(building_df.get("bid_complexity", 0.0))
    height = _safe_numeric(building_df.get("bid_height", 0.0))
    orientation_rad = np.deg2rad(orientation.astype(np.float64))

    feature_map = {
        "log_building_area": np.log1p(np.clip(area, a_min=0.0, a_max=None)),
        "log_building_perimeter": np.log1p(np.clip(perimeter, a_min=0.0, a_max=None)),
        "building_convexity": convexity,
        "log_building_corners": np.log1p(np.clip(corners, a_min=0.0, a_max=None)),
        "building_elongation": elongation,
        "sin_building_orientation": np.sin(orientation_rad).astype(np.float32),
        "cos_building_orientation": np.cos(orientation_rad).astype(np.float32),
        "building_rectangularity": rectangularity,
        "log_building_complexity": np.log1p(np.clip(complexity, a_min=0.0, a_max=None)),
        "log_building_height": np.log1p(np.clip(height, a_min=0.0, a_max=None)),
    }
    x = np.stack([feature_map[name].astype(np.float32) for name in BUILDING_FEATURE_NAMES], axis=1)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _intersection_features(intersection_df: pd.DataFrame) -> np.ndarray:
    feature_map = {
        "intersection_degree": _safe_numeric(intersection_df.get("intersection_degree", 0.0)),
        "intersection_clustering": _safe_numeric(intersection_df.get("intersection_clustering", 0.0)),
        "intersection_weighted_clustering": _safe_numeric(intersection_df.get("inter_section_weighted_clustering", 0.0)),
        "closeness_centrality": _safe_numeric(intersection_df.get("Closeness Centrality", 0.0)),
        "betweenness_centrality": _safe_numeric(intersection_df.get("Betweenness Centrality", 0.0)),
        "eigenvector_centrality": _safe_numeric(intersection_df.get("Eigenvector Centrality", 0.0)),
        "katz_centrality": _safe_numeric(intersection_df.get("Katz Centrality", 0.0)),
        "pagerank": _safe_numeric(intersection_df.get("PageRank", 0.0)),
    }
    x = np.stack([feature_map[name].astype(np.float32) for name in INTERSECTION_FEATURE_NAMES], axis=1)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _street_features(
    street_df: pd.DataFrame,
    intersection_df: pd.DataFrame,
    inv_street_to_plots: Dict[int, List[int]],
) -> np.ndarray:
    osmid = pd.to_numeric(intersection_df["osmid"], errors="coerce").fillna(-1).astype("int64")
    metric_map = {
        "degree": dict(zip(osmid.tolist(), _safe_numeric(intersection_df.get("intersection_degree", 0.0)).tolist())),
        "closeness": dict(zip(osmid.tolist(), _safe_numeric(intersection_df.get("Closeness Centrality", 0.0)).tolist())),
        "betweenness": dict(zip(osmid.tolist(), _safe_numeric(intersection_df.get("Betweenness Centrality", 0.0)).tolist())),
    }
    length = _safe_numeric(street_df.get("length", 0.0))
    u = pd.to_numeric(street_df.get("u", 0), errors="coerce").fillna(-1).astype("int64").to_numpy()
    v = pd.to_numeric(street_df.get("v", 0), errors="coerce").fillna(-1).astype("int64").to_numpy()

    degree_u = np.asarray([float(metric_map["degree"].get(int(x), 0.0)) for x in u], dtype=np.float32)
    degree_v = np.asarray([float(metric_map["degree"].get(int(x), 0.0)) for x in v], dtype=np.float32)
    closeness_u = np.asarray([float(metric_map["closeness"].get(int(x), 0.0)) for x in u], dtype=np.float32)
    closeness_v = np.asarray([float(metric_map["closeness"].get(int(x), 0.0)) for x in v], dtype=np.float32)
    betweenness_u = np.asarray([float(metric_map["betweenness"].get(int(x), 0.0)) for x in u], dtype=np.float32)
    betweenness_v = np.asarray([float(metric_map["betweenness"].get(int(x), 0.0)) for x in v], dtype=np.float32)
    street_ids = pd.to_numeric(street_df["street_id"], errors="coerce").fillna(-1).astype("int64").tolist()
    adjacent_plot_count = np.asarray([len(set(inv_street_to_plots.get(int(sid), []))) for sid in street_ids], dtype=np.float32)

    feature_map = {
        "log_street_length": np.log1p(np.clip(length, a_min=0.0, a_max=None)),
        "endpoint_degree_sum": degree_u + degree_v,
        "endpoint_degree_gap": np.abs(degree_u - degree_v),
        "endpoint_closeness_mean": 0.5 * (closeness_u + closeness_v),
        "endpoint_betweenness_mean": 0.5 * (betweenness_u + betweenness_v),
        "log_adjacent_plot_count": np.log1p(np.clip(adjacent_plot_count, a_min=0.0, a_max=None)),
    }
    x = np.stack([feature_map[name].astype(np.float32) for name in STREET_FEATURE_NAMES], axis=1)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def _edge_plot_street(
    plot_street: Sequence[Sequence[int]],
    street_id_to_idx: Dict[int, int],
    street_len_map: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    for plot_idx, street_ids in enumerate(plot_street):
        for sid in sorted(set(street_ids)):
            if sid not in street_id_to_idx:
                continue
            src.append(plot_idx)
            dst.append(street_id_to_idx[int(sid)])
            attr.append([float(street_len_map.get(int(sid), 0.0))])
    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    return edge_index, edge_attr


def _edge_plot_building(
    plot_building_idx: Sequence[Sequence[int]],
    plot_area: np.ndarray,
    building_area: np.ndarray,
    building_height_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    for plot_idx, building_indices in enumerate(plot_building_idx):
        p_area = max(float(plot_area[plot_idx]), 1.0)
        for b_idx in sorted(set(building_indices)):
            ratio = float(building_area[b_idx] / p_area)
            src.append(plot_idx)
            dst.append(int(b_idx))
            attr.append([ratio, float(building_height_log[b_idx])])
    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    return edge_index, edge_attr


def _edge_street_intersection(
    street_df: pd.DataFrame,
    street_id_to_idx: Dict[int, int],
    osmid_to_idx: Dict[int, int],
) -> np.ndarray:
    src: List[int] = []
    dst: List[int] = []
    for row in street_df[["street_id", "u", "v"]].itertuples(index=False):
        sid = int(row.street_id)
        s_idx = street_id_to_idx.get(sid)
        if s_idx is None:
            continue
        for endpoint in [row.u, row.v]:
            try:
                endpoint_int = int(endpoint)
            except Exception:
                continue
            i_idx = osmid_to_idx.get(endpoint_int)
            if i_idx is None:
                continue
            src.append(s_idx)
            dst.append(i_idx)
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    return edge_index


def _edge_plot_plot_shared_street(
    plot_street: Sequence[Sequence[int]],
    street_len_map: Dict[int, float],
    street_cap: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    inv_street = v3._build_inverse_index(plot_street)
    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}
    plot_street_count = np.asarray([len(set(ids)) for ids in plot_street], dtype=np.float32)
    for sid, nodes in inv_street.items():
        stats["entities_total"] += 1
        uniq = sorted(set(nodes))
        m = len(uniq)
        if m < 2:
            continue
        if m > int(street_cap):
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        s_len = float(street_len_map.get(int(sid), 0.0))
        for i, j in combinations(uniq, 2):
            key = (i, j)
            if key not in pair_stats:
                pair_stats[key] = [0.0, 0.0]
            pair_stats[key][0] += 1.0
            pair_stats[key][1] += s_len

    if not pair_stats:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
            stats,
        )

    pairs = np.asarray(list(pair_stats.keys()), dtype=np.int64)
    vals = np.asarray(list(pair_stats.values()), dtype=np.float32)
    src = pairs[:, 0]
    dst = pairs[:, 1]
    shared_cnt = vals[:, 0]
    shared_len_mean = np.divide(vals[:, 1], np.maximum(shared_cnt, 1.0))
    jaccard = np.divide(shared_cnt, np.maximum(plot_street_count[src] + plot_street_count[dst] - shared_cnt, 1.0))
    edge_index_undirected = np.vstack([src, dst]).astype(np.int64)
    edge_attr_undirected = np.stack([shared_cnt, jaccard, shared_len_mean], axis=1).astype(np.float32)
    edge_index, edge_attr = _duplicate_directed(edge_index_undirected, edge_attr_undirected)
    return edge_index, edge_attr, stats


def _edge_building_building_knn(
    building_df: pd.DataFrame,
    building_centroids_m: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = building_centroids_m.shape[0]
    if n < 2 or k <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    k_eff = min(int(k) + 1, n)
    tree = cKDTree(building_centroids_m)
    dists, nbrs = tree.query(building_centroids_m, k=k_eff)
    dists = np.atleast_2d(dists)
    nbrs = np.atleast_2d(nbrs)
    orientation_deg = _safe_numeric(building_df.get("bid_orientation", 0.0))

    pair_to_dist: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for dist, j in zip(dists[i, 1:], nbrs[i, 1:]):
            j = int(j)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            cur = pair_to_dist.get(key)
            val = float(dist)
            if cur is None or val < cur:
                pair_to_dist[key] = val

    if not pair_to_dist:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    pairs = np.asarray(list(pair_to_dist.keys()), dtype=np.int64)
    dist = np.asarray(list(pair_to_dist.values()), dtype=np.float32)
    orient_diff = _orientation_diff_deg(orientation_deg, pairs[:, 0], pairs[:, 1])
    edge_index_undirected = np.vstack([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    edge_attr_undirected = np.stack([dist, orient_diff], axis=1).astype(np.float32)
    edge_index, edge_attr = _duplicate_directed(edge_index_undirected, edge_attr_undirected)
    return edge_index, edge_attr


def _maybe_standardize(x: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return x.astype(np.float32)
    return v3._zscore(x.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare plot-centric heterogeneous urban dataset v1.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="", help="Default: urban_<city>_plot_hetero_v1")
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--building_knn_k", type=int, default=5)
    parser.add_argument("--street_entity_cap", type=int, default=0, help="0 = adaptive cap for plot-plot via shared streets")
    parser.add_argument("--street_cap_quantile", type=float, default=0.995)
    parser.add_argument("--standardize_node_features", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    city = args.city.strip()
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_hetero_v1"
    urban_root = Path(args.urban_root).resolve()
    city_dir = urban_root / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")

    required = [
        "plot.parquet",
        "street.parquet",
        "building.parquet",
        "intersection.parquet",
        "plot_street_id.npy",
        "plot_bid.npy",
    ]
    for fn in required:
        fp = city_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    out_dir = Path(args.out_root).resolve() / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (not args.force) and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty. Use --force and remove old files first.")
    if args.force:
        for fp in out_dir.iterdir():
            if fp.is_file():
                fp.unlink()
            elif fp.is_dir():
                raise ValueError(f"Unexpected subdirectory in output dir: {fp}")

    plot_df = _unique_by(pd.read_parquet(city_dir / "plot.parquet"), "plot_id")
    street_df = _unique_by(pd.read_parquet(city_dir / "street.parquet"), "street_id")
    building_df = _unique_by(pd.read_parquet(city_dir / "building.parquet"), "bid")
    intersection_df = _unique_by(pd.read_parquet(city_dir / "intersection.parquet"), "osmid")

    plot_ids = plot_df["plot_id"].to_numpy(dtype=np.int64, copy=True)
    street_ids = street_df["street_id"].to_numpy(dtype=np.int64, copy=True)
    building_ids = building_df["bid"].to_numpy(dtype=np.int64, copy=True)
    osmid_ids = intersection_df["osmid"].to_numpy(dtype=np.int64, copy=True)

    n_plot = len(plot_df)
    plot_street = v3._load_membership_array(city_dir / "plot_street_id.npy", n_plot, zero_scalar_empty=False)
    plot_bid = v3._load_membership_array(city_dir / "plot_bid.npy", n_plot, zero_scalar_empty=True)

    plot_x = _maybe_standardize(_plot_features(plot_df), enable=bool(args.standardize_node_features))
    street_id_to_idx = {int(sid): idx for idx, sid in enumerate(street_ids.tolist())}
    bid_to_idx = {int(bid): idx for idx, bid in enumerate(building_ids.tolist())}
    osmid_to_idx = {int(osmid): idx for idx, osmid in enumerate(osmid_ids.tolist())}
    plot_building_idx, plot_building_ref_stats = _resolve_membership_to_indices(
        memberships=plot_bid,
        node_ids=building_ids,
        id_to_idx=bid_to_idx,
        relation_name="plot_bid",
    )

    inv_street_to_plots = v3._build_inverse_index(plot_street)
    street_x = _maybe_standardize(_street_features(street_df, intersection_df, inv_street_to_plots), enable=bool(args.standardize_node_features))
    intersection_x = _maybe_standardize(_intersection_features(intersection_df), enable=bool(args.standardize_node_features))
    building_x = _maybe_standardize(_building_features(building_df), enable=bool(args.standardize_node_features))

    plot_y, label_mapping = v3._build_labels(plot_df, args.label_mode)

    street_len_map = v3._street_length_map(street_df)
    plot_area = _safe_numeric(plot_df.get("plot_area", 0.0))
    building_area = _safe_numeric(building_df.get("bid_area", 0.0))
    building_height_log = np.log1p(np.clip(_safe_numeric(building_df.get("bid_height", 0.0)), a_min=0.0, a_max=None)).astype(np.float32)

    plot_centroids_m = v3._lonlat_to_local_m(v3._geometry_centroids_from_wkb(plot_df["geometry"]))
    building_centroids_m = v3._lonlat_to_local_m(v3._geometry_centroids_from_wkb(building_df["geometry"]))

    plot_junction = v3._project_memberships(plot_street, v3._street_endpoint_map(street_df))
    plot_street_count = np.asarray([len(set(v)) for v in plot_street], dtype=np.int32)
    plot_building_count = np.asarray([len(set(v)) for v in plot_building_idx], dtype=np.int32)
    plot_junction_count = np.asarray([len(set(v)) for v in plot_junction], dtype=np.int32)

    building_parent_plots = v3._build_inverse_index(plot_building_idx)
    street_adjacent_plot_count = np.asarray(
        [len(set(inv_street_to_plots.get(int(sid), []))) for sid in street_ids.tolist()],
        dtype=np.int32,
    )
    building_parent_plot_count = np.asarray(
        [len(set(building_parent_plots.get(int(bid), []))) for bid in building_ids.tolist()],
        dtype=np.int32,
    )
    intersection_incident_street_count = np.zeros((len(intersection_df),), dtype=np.int32)
    for sid, row in street_df[["u", "v"]].iterrows():
        for endpoint in [row["u"], row["v"]]:
            try:
                idx = osmid_to_idx[int(endpoint)]
            except Exception:
                continue
            intersection_incident_street_count[idx] += 1

    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else v3._adaptive_entity_cap(
        inv_street_to_plots,
        min_cap=12,
        max_cap=64,
        quantile=float(args.street_cap_quantile),
    )

    edge_plot_plot, edge_attr_plot_plot, plot_plot_stats = _edge_plot_plot_shared_street(
        plot_street=plot_street,
        street_len_map=street_len_map,
        street_cap=street_cap,
    )
    edge_plot_street, edge_attr_plot_street = _edge_plot_street(
        plot_street=plot_street,
        street_id_to_idx=street_id_to_idx,
        street_len_map=street_len_map,
    )
    edge_street_plot = np.vstack([edge_plot_street[1], edge_plot_street[0]]).astype(np.int64)
    edge_attr_street_plot = edge_attr_plot_street.copy()

    edge_plot_building, edge_attr_plot_building = _edge_plot_building(
        plot_building_idx=plot_building_idx,
        plot_area=plot_area,
        building_area=building_area,
        building_height_log=building_height_log,
    )
    edge_building_plot = np.vstack([edge_plot_building[1], edge_plot_building[0]]).astype(np.int64)
    edge_attr_building_plot = edge_attr_plot_building.copy()

    edge_street_intersection = _edge_street_intersection(
        street_df=street_df,
        street_id_to_idx=street_id_to_idx,
        osmid_to_idx=osmid_to_idx,
    )
    edge_intersection_street = np.vstack([edge_street_intersection[1], edge_street_intersection[0]]).astype(np.int64)

    edge_building_building, edge_attr_building_building = _edge_building_building_knn(
        building_df=building_df,
        building_centroids_m=building_centroids_m,
        k=int(args.building_knn_k),
    )

    # Node files.
    node_types_meta = {
        "plot": {
            "num_nodes": int(len(plot_df)),
            "feature_columns": PLOT_FEATURE_NAMES,
            "feature_file": _save_array(out_dir, "plot_feat", plot_x.astype(np.float32)),
            "id_file": _save_array(out_dir, "plot_node_ids", plot_ids),
            "label_file": _save_array(out_dir, "plot_label", plot_y.astype(np.int64)),
            "node_table_file": "plot_node_table.csv",
        },
        "street": {
            "num_nodes": int(len(street_df)),
            "feature_columns": STREET_FEATURE_NAMES,
            "feature_file": _save_array(out_dir, "street_feat", street_x.astype(np.float32)),
            "id_file": _save_array(out_dir, "street_node_ids", street_ids),
            "node_table_file": "street_node_table.csv",
        },
        "intersection": {
            "num_nodes": int(len(intersection_df)),
            "feature_columns": INTERSECTION_FEATURE_NAMES,
            "feature_file": _save_array(out_dir, "intersection_feat", intersection_x.astype(np.float32)),
            "id_file": _save_array(out_dir, "intersection_node_ids", osmid_ids),
            "node_table_file": "intersection_node_table.csv",
        },
        "building": {
            "num_nodes": int(len(building_df)),
            "feature_columns": BUILDING_FEATURE_NAMES,
            "feature_file": _save_array(out_dir, "building_feat", building_x.astype(np.float32)),
            "id_file": _save_array(out_dir, "building_node_ids", building_ids),
            "node_table_file": "building_node_table.csv",
        },
    }

    pd.DataFrame(
        {
            "node_idx": np.arange(len(plot_df), dtype=np.int64),
            "plot_id": plot_ids,
            "label": plot_y.astype(np.int64),
            "street_membership_count": plot_street_count,
            "junction_membership_count": plot_junction_count,
            "building_membership_count": plot_building_count,
        }
    ).to_csv(out_dir / "plot_node_table.csv", index=False)
    pd.DataFrame(
        {
            "node_idx": np.arange(len(street_df), dtype=np.int64),
            "street_id": street_ids,
            "u": _normalize_id_series(street_df["u"]),
            "v": _normalize_id_series(street_df["v"]),
            "length": _safe_numeric(street_df["length"]),
            "adjacent_plot_count": street_adjacent_plot_count,
        }
    ).to_csv(out_dir / "street_node_table.csv", index=False)
    pd.DataFrame(
        {
            "node_idx": np.arange(len(intersection_df), dtype=np.int64),
            "intersection_id": _normalize_id_series(pd.Series(intersection_df.get("intersection_id", np.arange(len(intersection_df))))),
            "osmid": osmid_ids,
            "incident_street_count": intersection_incident_street_count,
        }
    ).to_csv(out_dir / "intersection_node_table.csv", index=False)
    pd.DataFrame(
        {
            "node_idx": np.arange(len(building_df), dtype=np.int64),
            "bid": building_ids,
            "parent_plot_count": building_parent_plot_count,
            "bid_height": _safe_numeric(building_df.get("bid_height", 0.0)),
        }
    ).to_csv(out_dir / "building_node_table.csv", index=False)

    edge_types_meta = {}

    def _register_edge(name: str, src_type: str, rel: str, dst_type: str, edge_index: np.ndarray, edge_attr: np.ndarray | None, reverse_key: str | None) -> None:
        edge_index_file = _save_array(out_dir, f"edge_{name}", edge_index.astype(np.int64))
        edge_attr_file = None
        edge_attr_names = []
        if edge_attr is not None:
            edge_attr_file = _save_array(out_dir, f"edge_attr_{name}", edge_attr.astype(np.float32))
            edge_attr_names = EDGE_ATTR_NAMES.get(name, [])
        edge_types_meta[name] = {
            "src_type": src_type,
            "relation": rel,
            "dst_type": dst_type,
            "edge_index_file": edge_index_file,
            "edge_attr_file": edge_attr_file,
            "edge_attr_names": edge_attr_names,
            "num_edges": int(edge_index.shape[1]),
            "reverse_key": reverse_key,
        }

    _register_edge("plot__shares_street__plot", "plot", "shares_street", "plot", edge_plot_plot, edge_attr_plot_plot, "plot__shares_street__plot")
    _register_edge("plot__bounded_by__street", "plot", "bounded_by", "street", edge_plot_street, edge_attr_plot_street, "street__rev_bounded_by__plot")
    _register_edge("street__rev_bounded_by__plot", "street", "rev_bounded_by", "plot", edge_street_plot, edge_attr_street_plot, "plot__bounded_by__street")
    _register_edge("street__connects__intersection", "street", "connects", "intersection", edge_street_intersection, None, "intersection__rev_connects__street")
    _register_edge("intersection__rev_connects__street", "intersection", "rev_connects", "street", edge_intersection_street, None, "street__connects__intersection")
    _register_edge("plot__contains__building", "plot", "contains", "building", edge_plot_building, edge_attr_plot_building, "building__rev_contains__plot")
    _register_edge("building__rev_contains__plot", "building", "rev_contains", "plot", edge_building_plot, edge_attr_building_plot, "plot__contains__building")
    _register_edge("building__near__building", "building", "near", "building", edge_building_building, edge_attr_building_building, "building__near__building")

    label_counts = {str(k): int(v) for k, v in zip(*np.unique(plot_y, return_counts=True))}
    meta = {
        "dataset_name": dataset_name,
        "city": city,
        "dataset_family": "urban_plot_hetero_v1",
        "target_node_type": "plot",
        "target_task": "plot_label_for_evaluation",
        "notes": [
            "This dataset is plot-centric but keeps streets, intersections, and buildings as separate node types.",
            "Main graph relations are urban-object relations only; no feature-similarity graph is used in the main topology.",
            "The format is designed for future hetero GNN work and simple structure validation before model integration.",
        ],
        "node_type_order": NODE_TYPE_ORDER,
        "node_types": node_types_meta,
        "edge_types": edge_types_meta,
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "label_counts": label_counts,
        "builder_args": {
            "building_knn_k": int(args.building_knn_k),
            "street_entity_cap": int(street_cap),
            "street_cap_quantile": float(args.street_cap_quantile),
            "standardize_node_features": bool(args.standardize_node_features),
        },
        "relation_stats": {
            "plot_plot_shared_street": plot_plot_stats,
            "plot_building_reference_mode": plot_building_ref_stats,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] prepared {dataset_name}")
    print(f" city={city}")
    print(f" nodes plot/street/intersection/building = {len(plot_df)}/{len(street_df)}/{len(intersection_df)}/{len(building_df)}")
    print(
        " edges "
        f"plot-plot={edge_plot_plot.shape[1]} "
        f"plot-street={edge_plot_street.shape[1]} "
        f"street-intersection={edge_street_intersection.shape[1]} "
        f"plot-building={edge_plot_building.shape[1]} "
        f"building-building={edge_building_building.shape[1]}"
    )


if __name__ == "__main__":
    main()
