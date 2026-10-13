#!/usr/bin/env python3
"""
Prepare a richer plot-centric heterogeneous urban dataset.

V2 design principles:
- preserve the paper-style urban heterograph skeleton:
  plot / street / intersection / building
- keep relation types explicit instead of collapsing everything into one graph
- upgrade each relation with interpretable edge attributes that describe geometry,
  containment, adjacency, accessibility, or similarity
- keep semantic building-building edges separate from spatial building-building edges

Output folder example:
  data/urban_beijing_plot_hetero_v2/
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
import prepare_urban_plot_hetero_v1 as v1


NODE_TYPE_ORDER = ["plot", "street", "intersection", "building"]

EDGE_ATTR_NAMES = {
    "plot__shares_street__plot": [
        "shared_street_cnt",
        "jaccard_street",
        "shared_street_len_mean",
        "centroid_dist_m",
        "log_area_ratio_abs",
        "canopy_abs_diff",
        "pop_density_abs_diff",
        "built_density_abs_diff",
        "lcz_cosine",
    ],
    "plot__bounded_by__street": [
        "street_length",
        "plot_street_count",
        "plot_centroid_to_street_dist_m",
        "plot_street_angle_diff_deg",
        "street_adjacent_plot_count",
        "street_endpoint_degree_sum",
    ],
    "street__rev_bounded_by__plot": [
        "street_length",
        "plot_street_count",
        "plot_centroid_to_street_dist_m",
        "plot_street_angle_diff_deg",
        "street_adjacent_plot_count",
        "street_endpoint_degree_sum",
    ],
    "street__connects__intersection": [
        "street_length",
        "endpoint_angle_deg",
        "is_start_endpoint",
        "intersection_degree",
        "intersection_closeness",
        "intersection_betweenness",
        "street_dead_end",
    ],
    "intersection__rev_connects__street": [
        "street_length",
        "endpoint_angle_deg",
        "is_start_endpoint",
        "intersection_degree",
        "intersection_closeness",
        "intersection_betweenness",
        "street_dead_end",
    ],
    "plot__contains__building": [
        "building_area_ratio_to_plot",
        "building_centroid_to_plot_centroid_dist_m",
        "building_to_plot_boundary_dist_m",
        "plot_building_count",
        "is_largest_in_plot",
        "touches_plot_boundary",
        "log_building_height",
    ],
    "building__rev_contains__plot": [
        "building_area_ratio_to_plot",
        "building_centroid_to_plot_centroid_dist_m",
        "building_to_plot_boundary_dist_m",
        "plot_building_count",
        "is_largest_in_plot",
        "touches_plot_boundary",
        "log_building_height",
    ],
    "building__proximal_to__street": [
        "building_centroid_to_street_dist_m",
        "building_boundary_to_street_dist_m",
        "building_street_angle_diff_deg",
        "street_length",
        "parent_plot_street_count",
        "shared_plot_count",
        "is_closest_street",
        "frontage_proximity",
    ],
    "street__rev_proximal_to__building": [
        "building_centroid_to_street_dist_m",
        "building_boundary_to_street_dist_m",
        "building_street_angle_diff_deg",
        "street_length",
        "parent_plot_street_count",
        "shared_plot_count",
        "is_closest_street",
        "frontage_proximity",
    ],
    "building__near__building": [
        "centroid_dist_m",
        "mutual_knn",
        "same_plot",
        "shared_street_cnt",
        "orientation_diff_deg",
        "log_area_ratio_abs",
        "shape_l1",
        "height_log_abs_diff",
        "morphology_cosine",
    ],
    "building__semantic__building": [
        "gaussian_similarity",
        "semantic_cosine",
        "morphology_cosine",
        "same_plot",
        "shared_street_cnt",
        "visual_similarity",
        "visual_missing_mask",
    ],
}


def _project_xy(coords_lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    if coords_lonlat.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    coords = np.asarray(coords_lonlat, dtype=np.float64)
    cos_lat = math.cos(math.radians(float(lat0)))
    x = (coords[:, 0] - float(lon0)) * 111_320.0 * cos_lat
    y = (coords[:, 1] - float(lat0)) * 110_540.0
    out = np.stack([x, y], axis=1)
    out[~np.isfinite(out)] = 0.0
    return out


def _principal_ring_local(raw, lon0: float, lat0: float) -> np.ndarray:
    if not raw:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        geom, _ = v3._parse_wkb(bytes(raw))
    except Exception:
        return np.zeros((0, 2), dtype=np.float64)
    best = None
    best_area = -1.0
    if geom["type"] == "Polygon":
        rings = geom["rings"]
        if rings:
            best = rings[0]
    elif geom["type"] == "MultiPolygon":
        for rings in geom["polys"]:
            if not rings:
                continue
            _, _, area = v3._polygon_centroid_area(rings[0])
            if area > best_area:
                best_area = area
                best = rings[0]
    if best is None:
        return np.zeros((0, 2), dtype=np.float64)
    return _project_xy(np.asarray(best, dtype=np.float64), lon0=lon0, lat0=lat0)


def _ensure_closed(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return coords.reshape(0, 2)
    if np.allclose(coords[0], coords[-1]):
        return coords
    return np.vstack([coords, coords[0]])


def _safe_div(num: np.ndarray, den: np.ndarray | float) -> np.ndarray:
    den_arr = np.asarray(den, dtype=np.float64)
    return np.divide(num, np.maximum(den_arr, 1e-9))


def _wrap_abs_angle_diff(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    diff = np.abs(np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64))
    diff = np.minimum(diff, 360.0 - diff)
    diff = np.minimum(diff, 180.0)
    return diff.astype(np.float32)


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _segment_orientation_deg(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360.0
    return float(angle)


def _point_to_polyline_distance(p: np.ndarray, line: np.ndarray) -> float:
    if line.shape[0] == 0:
        return 0.0
    if line.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - line[0]))
    best = float("inf")
    for idx in range(line.shape[0] - 1):
        best = min(best, _point_to_segment_distance(p, line[idx], line[idx + 1]))
    return float(best)


def _point_to_ring_distance(p: np.ndarray, ring: np.ndarray) -> float:
    ring_c = _ensure_closed(ring)
    if ring_c.shape[0] == 0:
        return 0.0
    if ring_c.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - ring_c[0]))
    best = float("inf")
    for idx in range(ring_c.shape[0] - 1):
        best = min(best, _point_to_segment_distance(p, ring_c[idx], ring_c[idx + 1]))
    return float(best)


def _polyline_to_polyline_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    a_c = _ensure_closed(a) if a.shape[0] >= 2 else a
    b_c = _ensure_closed(b) if b.shape[0] >= 2 else b
    best = float("inf")
    for p in a_c:
        best = min(best, _point_to_ring_distance(p, b_c))
    for p in b_c:
        best = min(best, _point_to_ring_distance(p, a_c))
    return float(best)


def _vertices_to_polyline_distance(vertices: np.ndarray, line: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] == 0 or line.shape[0] == 0:
        return 0.0
    best = float("inf")
    for p in vertices:
        best = min(best, _point_to_polyline_distance(p, line))
    return float(best)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.divide(dot, np.maximum(na * nb, 1e-9))
    cos[~np.isfinite(cos)] = 0.0
    return cos.astype(np.float32)


def _lookup_edge_attr_dim(name: str) -> int:
    return len(EDGE_ATTR_NAMES.get(name, []))


def _register_edge(
    edge_types_meta: Dict[str, dict],
    out_dir: Path,
    name: str,
    src_type: str,
    rel: str,
    dst_type: str,
    edge_index: np.ndarray,
    edge_attr: np.ndarray | None,
    reverse_key: str | None,
) -> None:
    edge_index_file = v1._save_array(out_dir, f"edge_{name}", edge_index.astype(np.int64))
    edge_attr_file = None
    edge_attr_names = []
    if edge_attr is not None:
        edge_attr_file = v1._save_array(out_dir, f"edge_attr_{name}", edge_attr.astype(np.float32))
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


def _building_plot_relation(
    plot_building_idx: Sequence[Sequence[int]],
    plot_centroids_m: np.ndarray,
    plot_area: np.ndarray,
    building_centroids_m: np.ndarray,
    building_area: np.ndarray,
    building_height_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    plot_building_count = np.asarray([len(set(v)) for v in plot_building_idx], dtype=np.int32)
    building_to_plot = defaultdict(list)
    for p_idx, b_list in enumerate(plot_building_idx):
        for b_idx in sorted(set(b_list)):
            building_to_plot[int(b_idx)].append(int(p_idx))

    largest_building = {}
    for p_idx, b_list in enumerate(plot_building_idx):
        uniq = sorted(set(b_list))
        if not uniq:
            continue
        best = max(uniq, key=lambda b_idx: float(building_area[int(b_idx)]))
        largest_building[int(p_idx)] = int(best)

    touches = 0
    plot_eq_radius = np.sqrt(np.maximum(plot_area.astype(np.float64), 1.0) / math.pi)
    building_eq_radius = np.sqrt(np.maximum(building_area.astype(np.float64), 1.0) / math.pi)
    for p_idx, b_list in enumerate(plot_building_idx):
        p_area = max(float(plot_area[p_idx]), 1.0)
        p_centroid = plot_centroids_m[p_idx]
        count = float(plot_building_count[p_idx])
        for b_idx in sorted(set(b_list)):
            b_centroid = building_centroids_m[b_idx]
            ratio = float(building_area[b_idx] / p_area)
            centroid_dist = float(np.linalg.norm(b_centroid - p_centroid))
            boundary_dist = float(max(plot_eq_radius[p_idx] - centroid_dist, 0.0))
            touch = 1.0 if boundary_dist <= float(max(building_eq_radius[b_idx], 2.0)) else 0.0
            if touch > 0.0:
                touches += 1
            src.append(p_idx)
            dst.append(int(b_idx))
            attr.append(
                [
                    ratio,
                    centroid_dist,
                    boundary_dist,
                    count,
                    1.0 if int(largest_building.get(int(p_idx), -1)) == int(b_idx) else 0.0,
                    touch,
                    float(building_height_log[b_idx]),
                ]
            )

    if not src:
        e = np.zeros((2, 0), dtype=np.int64)
        a = np.zeros((0, _lookup_edge_attr_dim("plot__contains__building")), dtype=np.float32)
        return e, a, np.zeros((2, 0), dtype=np.int64), a.copy(), {"touch_ratio": 0.0}
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    rev_edge = np.vstack([edge_index[1], edge_index[0]]).astype(np.int64)
    rev_attr = edge_attr.copy()
    stats = {"touch_ratio": float(touches / max(len(src), 1))}
    return edge_index, edge_attr, rev_edge, rev_attr, stats


def _plot_street_relation(
    plot_street: Sequence[Sequence[int]],
    plot_centroids_m: np.ndarray,
    plot_orientation: np.ndarray,
    plot_street_count: np.ndarray,
    street_id_to_idx: Dict[int, int],
    street_length: np.ndarray,
    street_adjacent_plot_count: np.ndarray,
    street_endpoint_degree_sum: np.ndarray,
    street_orientation_deg: np.ndarray,
    street_segments: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    for p_idx, street_ids in enumerate(plot_street):
        p_point = plot_centroids_m[p_idx]
        p_ori = float(plot_orientation[p_idx])
        p_street_n = float(plot_street_count[p_idx])
        for sid in sorted(set(street_ids)):
            s_idx = street_id_to_idx.get(int(sid))
            if s_idx is None:
                continue
            dist = _point_to_polyline_distance(p_point, street_segments[s_idx])
            angle = float(_wrap_abs_angle_diff(p_ori, street_orientation_deg[s_idx]))
            src.append(p_idx)
            dst.append(int(s_idx))
            attr.append(
                [
                    float(street_length[s_idx]),
                    p_street_n,
                    dist,
                    angle,
                    float(street_adjacent_plot_count[s_idx]),
                    float(street_endpoint_degree_sum[s_idx]),
                ]
            )
    if not src:
        dim = _lookup_edge_attr_dim("plot__bounded_by__street")
        e = np.zeros((2, 0), dtype=np.int64)
        a = np.zeros((0, dim), dtype=np.float32)
        return e, a, np.zeros((2, 0), dtype=np.int64), a.copy()
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    rev_edge = np.vstack([edge_index[1], edge_index[0]]).astype(np.int64)
    rev_attr = edge_attr.copy()
    return edge_index, edge_attr, rev_edge, rev_attr


def _street_intersection_relation(
    street_df: pd.DataFrame,
    street_id_to_idx: Dict[int, int],
    osmid_to_idx: Dict[int, int],
    street_length: np.ndarray,
    street_start_angle_deg: np.ndarray,
    street_end_angle_deg: np.ndarray,
    street_dead_end: np.ndarray,
    intersection_degree: np.ndarray,
    intersection_closeness: np.ndarray,
    intersection_betweenness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    for row in street_df[["street_id", "u", "v"]].itertuples(index=False):
        try:
            sid = int(row.street_id)
        except Exception:
            continue
        s_idx = street_id_to_idx.get(sid)
        if s_idx is None:
            continue
        endpoints = [(row.u, 1.0, float(street_start_angle_deg[s_idx])), (row.v, 0.0, float(street_end_angle_deg[s_idx]))]
        for endpoint, is_start, angle in endpoints:
            try:
                i_idx = osmid_to_idx.get(int(endpoint))
            except Exception:
                i_idx = None
            if i_idx is None:
                continue
            src.append(int(s_idx))
            dst.append(int(i_idx))
            attr.append(
                [
                    float(street_length[s_idx]),
                    float(angle),
                    float(is_start),
                    float(intersection_degree[i_idx]),
                    float(intersection_closeness[i_idx]),
                    float(intersection_betweenness[i_idx]),
                    float(street_dead_end[s_idx]),
                ]
            )
    if not src:
        dim = _lookup_edge_attr_dim("street__connects__intersection")
        e = np.zeros((2, 0), dtype=np.int64)
        a = np.zeros((0, dim), dtype=np.float32)
        return e, a, np.zeros((2, 0), dtype=np.int64), a.copy()
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    rev_edge = np.vstack([edge_index[1], edge_index[0]]).astype(np.int64)
    rev_attr = edge_attr.copy()
    return edge_index, edge_attr, rev_edge, rev_attr


def _plot_plot_relation(
    plot_street: Sequence[Sequence[int]],
    street_len_map: Dict[int, float],
    street_cap: int,
    plot_centroids_m: np.ndarray,
    plot_area: np.ndarray,
    canopy_mean: np.ndarray,
    pop_density: np.ndarray,
    built_density: np.ndarray,
    lcz_group: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    edge_index, base_attr, stats = v1._edge_plot_plot_shared_street(
        plot_street=plot_street,
        street_len_map=street_len_map,
        street_cap=street_cap,
    )
    if edge_index.shape[1] == 0:
        return edge_index, np.zeros((0, _lookup_edge_attr_dim("plot__shares_street__plot")), dtype=np.float32), stats
    src = edge_index[0]
    dst = edge_index[1]
    centroid_dist = np.linalg.norm(plot_centroids_m[src] - plot_centroids_m[dst], axis=1).astype(np.float32)
    log_area_ratio_abs = np.abs(np.log1p(plot_area[src]) - np.log1p(plot_area[dst])).astype(np.float32)
    canopy_diff = np.abs(canopy_mean[src] - canopy_mean[dst]).astype(np.float32)
    pop_density_diff = np.abs(pop_density[src] - pop_density[dst]).astype(np.float32)
    built_density_diff = np.abs(built_density[src] - built_density[dst]).astype(np.float32)
    lcz_cosine = _cosine_rows(lcz_group[src], lcz_group[dst])
    edge_attr = np.concatenate(
        [
            base_attr.astype(np.float32),
            np.stack(
                [
                    centroid_dist,
                    log_area_ratio_abs,
                    canopy_diff,
                    pop_density_diff,
                    built_density_diff,
                    lcz_cosine,
                ],
                axis=1,
            ).astype(np.float32),
        ],
        axis=1,
    )
    return edge_index.astype(np.int64), edge_attr.astype(np.float32), stats


def _building_street_relation(
    building_parent_plots: Dict[int, List[int]],
    plot_street: Sequence[Sequence[int]],
    street_id_to_idx: Dict[int, int],
    street_length: np.ndarray,
    street_orientation_deg: np.ndarray,
    street_segments: Sequence[np.ndarray],
    building_centroids_m: np.ndarray,
    building_orientation_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    selected_edges = 0
    total_candidates = 0
    for b_idx, parent_plots in building_parent_plots.items():
        parent_plots = sorted(set(int(x) for x in parent_plots))
        if not parent_plots:
            continue
        candidate_streets: Dict[int, int] = defaultdict(int)
        for p_idx in parent_plots:
            for sid in set(plot_street[p_idx]):
                s_idx = street_id_to_idx.get(int(sid))
                if s_idx is not None:
                    candidate_streets[int(s_idx)] += 1
        if not candidate_streets:
            continue
        total_candidates += len(candidate_streets)
        b_centroid = building_centroids_m[b_idx]
        b_ori = float(building_orientation_deg[b_idx])
        best_s_idx = None
        best_centroid_dist = float("inf")
        best_boundary_dist = float("inf")
        for s_idx in candidate_streets.keys():
            centroid_dist = _point_to_polyline_distance(b_centroid, street_segments[s_idx])
            boundary_dist = centroid_dist
            if centroid_dist < best_centroid_dist:
                best_centroid_dist = centroid_dist
                best_boundary_dist = boundary_dist
                best_s_idx = int(s_idx)
        if best_s_idx is None:
            continue
        selected_edges += 1
        shared_plot_count = float(candidate_streets[best_s_idx])
        parent_plot_street_count = float(np.mean([len(set(plot_street[p_idx])) for p_idx in parent_plots]))
        angle = float(_wrap_abs_angle_diff(b_ori, street_orientation_deg[best_s_idx]))
        frontage_proximity = float(math.exp(-best_boundary_dist / 20.0))
        src.append(int(b_idx))
        dst.append(int(best_s_idx))
        attr.append(
            [
                float(best_centroid_dist),
                float(best_boundary_dist),
                angle,
                float(street_length[best_s_idx]),
                parent_plot_street_count,
                shared_plot_count,
                1.0,
                frontage_proximity,
            ]
        )
    if not src:
        dim = _lookup_edge_attr_dim("building__proximal_to__street")
        e = np.zeros((2, 0), dtype=np.int64)
        a = np.zeros((0, dim), dtype=np.float32)
        return e, a, np.zeros((2, 0), dtype=np.int64), a.copy(), {"selected_ratio": 0.0}
    edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    edge_attr = np.asarray(attr, dtype=np.float32)
    rev_edge = np.vstack([edge_index[1], edge_index[0]]).astype(np.int64)
    rev_attr = edge_attr.copy()
    stats = {"selected_ratio": float(selected_edges / max(total_candidates, 1))}
    return edge_index, edge_attr, rev_edge, rev_attr, stats


def _building_building_spatial_relation(
    building_centroids_m: np.ndarray,
    building_orientation_deg: np.ndarray,
    building_area: np.ndarray,
    building_height_log: np.ndarray,
    building_x: np.ndarray,
    building_parent_plots: Dict[int, List[int]],
    building_street_membership: Sequence[Sequence[int]],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(building_centroids_m.shape[0])
    if n < 2 or int(k) <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, _lookup_edge_attr_dim("building__near__building")), dtype=np.float32)
    k_eff = min(int(k) + 1, n)
    tree = cKDTree(building_centroids_m)
    dists, nbrs = tree.query(building_centroids_m, k=k_eff)
    dists = np.asarray(dists, dtype=np.float64)
    nbrs = np.asarray(nbrs, dtype=np.int64)
    pair_to_dist: Dict[Tuple[int, int], float] = {}
    directed_pairs = set()
    for i in range(n):
        for dist, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(nbrs[i])[1:]):
            j = int(j)
            if i == j:
                continue
            directed_pairs.add((int(i), j))
            key = (int(i), j) if int(i) < j else (j, int(i))
            cur = pair_to_dist.get(key)
            if cur is None or float(dist) < cur:
                pair_to_dist[key] = float(dist)
    if not pair_to_dist:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, _lookup_edge_attr_dim("building__near__building")), dtype=np.float32)
    pairs = np.asarray(list(pair_to_dist.keys()), dtype=np.int64)
    src_u = pairs[:, 0]
    dst_u = pairs[:, 1]
    dist = np.asarray(list(pair_to_dist.values()), dtype=np.float32)
    mutual = np.asarray([1.0 if ((int(j), int(i)) in directed_pairs and (int(i), int(j)) in directed_pairs) else 0.0 for i, j in zip(src_u, dst_u)], dtype=np.float32)
    same_plot = np.asarray(
        [1.0 if len(set(building_parent_plots.get(int(i), ())) & set(building_parent_plots.get(int(j), ()))) > 0 else 0.0 for i, j in zip(src_u, dst_u)],
        dtype=np.float32,
    )
    shared_street_cnt = np.asarray(
        [float(len(set(building_street_membership[int(i)]) & set(building_street_membership[int(j)]))) for i, j in zip(src_u, dst_u)],
        dtype=np.float32,
    )
    orient_diff = _wrap_abs_angle_diff(building_orientation_deg[src_u], building_orientation_deg[dst_u])
    log_area_ratio_abs = np.abs(np.log1p(building_area[src_u]) - np.log1p(building_area[dst_u])).astype(np.float32)
    shape_a = np.stack([np.log1p(building_area), building_x[:, 4], building_x[:, 7], building_x[:, 8]], axis=1).astype(np.float32)
    shape_l1 = np.mean(np.abs(shape_a[src_u] - shape_a[dst_u]), axis=1).astype(np.float32)
    height_gap = np.abs(building_height_log[src_u] - building_height_log[dst_u]).astype(np.float32)
    morph_cos = _cosine_rows(building_x[src_u], building_x[dst_u])
    edge_index_undirected = np.vstack([src_u, dst_u]).astype(np.int64)
    edge_attr_undirected = np.stack(
        [
            dist,
            mutual,
            same_plot,
            shared_street_cnt,
            orient_diff,
            log_area_ratio_abs,
            shape_l1,
            height_gap,
            morph_cos,
        ],
        axis=1,
    ).astype(np.float32)
    edge_index, edge_attr = v1._duplicate_directed(edge_index_undirected, edge_attr_undirected)
    return edge_index.astype(np.int64), edge_attr.astype(np.float32)


def _building_building_semantic_relation(
    semantic_feat: np.ndarray,
    building_x: np.ndarray,
    building_parent_plots: Dict[int, List[int]],
    building_street_membership: Sequence[Sequence[int]],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n = int(semantic_feat.shape[0])
    if n < 2 or int(k) <= 0:
        dim = _lookup_edge_attr_dim("building__semantic__building")
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, dim), dtype=np.float32), {"sigma": 1.0}
    feat = v3._zscore(semantic_feat.astype(np.float32))
    tree = cKDTree(feat)
    k_eff = min(int(k) + 1, n)
    dists, nbrs = tree.query(feat, k=k_eff)
    dists = np.asarray(dists, dtype=np.float64)
    nbrs = np.asarray(nbrs, dtype=np.int64)
    pair_to_dist: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for dist, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(nbrs[i])[1:]):
            j = int(j)
            if i == j:
                continue
            key = (int(i), j) if int(i) < j else (j, int(i))
            cur = pair_to_dist.get(key)
            if cur is None or float(dist) < cur:
                pair_to_dist[key] = float(dist)
    if not pair_to_dist:
        dim = _lookup_edge_attr_dim("building__semantic__building")
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, dim), dtype=np.float32), {"sigma": 1.0}
    pairs = np.asarray(list(pair_to_dist.keys()), dtype=np.int64)
    src_u = pairs[:, 0]
    dst_u = pairs[:, 1]
    dist = np.asarray(list(pair_to_dist.values()), dtype=np.float32)
    sigma = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    sigma = max(sigma, 1e-6)
    gaussian = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    semantic_cos = _cosine_rows(feat[src_u], feat[dst_u])
    morph_cos = _cosine_rows(building_x[src_u], building_x[dst_u])
    same_plot = np.asarray(
        [1.0 if len(set(building_parent_plots.get(int(i), ())) & set(building_parent_plots.get(int(j), ()))) > 0 else 0.0 for i, j in zip(src_u, dst_u)],
        dtype=np.float32,
    )
    shared_street_cnt = np.asarray(
        [float(len(set(building_street_membership[int(i)]) & set(building_street_membership[int(j)]))) for i, j in zip(src_u, dst_u)],
        dtype=np.float32,
    )
    visual_similarity = np.zeros_like(gaussian, dtype=np.float32)
    visual_missing_mask = np.ones_like(gaussian, dtype=np.float32)
    edge_index_undirected = np.vstack([src_u, dst_u]).astype(np.int64)
    edge_attr_undirected = np.stack(
        [
            gaussian,
            semantic_cos,
            morph_cos,
            same_plot,
            shared_street_cnt,
            visual_similarity,
            visual_missing_mask,
        ],
        axis=1,
    ).astype(np.float32)
    edge_index, edge_attr = v1._duplicate_directed(edge_index_undirected, edge_attr_undirected)
    return edge_index.astype(np.int64), edge_attr.astype(np.float32), {"sigma": sigma}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare plot-centric heterogeneous urban dataset v2 with richer typed edge attributes.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="", help="Default: urban_<city>_plot_hetero_v2")
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--building_spatial_knn_k", type=int, default=5)
    parser.add_argument("--building_semantic_knn_k", type=int, default=5)
    parser.add_argument("--disable_building_semantic", action="store_true")
    parser.add_argument("--proxy_building_centroid_from_plot", action="store_true")
    parser.add_argument("--street_entity_cap", type=int, default=0, help="0 = adaptive cap for plot-plot via shared streets")
    parser.add_argument("--street_cap_quantile", type=float, default=0.995)
    parser.add_argument("--standardize_node_features", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    city = args.city.strip()
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_hetero_v2"
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

    plot_df = v1._unique_by(pd.read_parquet(city_dir / "plot.parquet"), "plot_id")
    street_df = v1._unique_by(pd.read_parquet(city_dir / "street.parquet"), "street_id")
    building_df = v1._unique_by(pd.read_parquet(city_dir / "building.parquet"), "bid")
    intersection_df = v1._unique_by(pd.read_parquet(city_dir / "intersection.parquet"), "osmid")

    plot_ids = plot_df["plot_id"].to_numpy(dtype=np.int64, copy=True)
    street_ids = street_df["street_id"].to_numpy(dtype=np.int64, copy=True)
    building_ids = building_df["bid"].to_numpy(dtype=np.int64, copy=True)
    osmid_ids = intersection_df["osmid"].to_numpy(dtype=np.int64, copy=True)

    n_plot = len(plot_df)
    plot_street = v3._load_membership_array(city_dir / "plot_street_id.npy", n_plot, zero_scalar_empty=False)
    plot_bid_raw = v3._load_membership_array(city_dir / "plot_bid.npy", n_plot, zero_scalar_empty=True)

    plot_x = v1._maybe_standardize(v1._plot_features(plot_df), enable=bool(args.standardize_node_features))
    building_x = v1._maybe_standardize(v1._building_features(building_df), enable=bool(args.standardize_node_features))
    plot_y, label_mapping = v3._build_labels(plot_df, args.label_mode)

    street_id_to_idx = {int(sid): idx for idx, sid in enumerate(street_ids.tolist())}
    bid_to_idx = {int(bid): idx for idx, bid in enumerate(building_ids.tolist())}
    osmid_to_idx = {int(osmid): idx for idx, osmid in enumerate(osmid_ids.tolist())}

    plot_building_idx, plot_building_ref_stats = v1._resolve_membership_to_indices(
        memberships=plot_bid_raw,
        node_ids=building_ids,
        id_to_idx=bid_to_idx,
        relation_name="plot_bid",
    )
    building_parent_plots = v3._build_inverse_index(plot_building_idx)

    plot_centroids_lonlat = v3._geometry_centroids_from_wkb(plot_df["geometry"])
    plot_centroids_m = v3._lonlat_to_local_m(plot_centroids_lonlat)
    if bool(args.proxy_building_centroid_from_plot):
        building_centroids_m = np.zeros((len(building_df), 2), dtype=np.float64)
        building_proxy_valid = np.zeros((len(building_df),), dtype=np.float32)
        for b_idx in range(len(building_df)):
            parent_plots = sorted(set(building_parent_plots.get(int(b_idx), [])))
            if not parent_plots:
                continue
            building_centroids_m[b_idx] = np.mean(plot_centroids_m[parent_plots], axis=0)
            building_proxy_valid[b_idx] = 1.0
        building_centroids_lonlat = np.zeros((len(building_df), 2), dtype=np.float64)
    else:
        building_centroids_lonlat = v3._geometry_centroids_from_wkb(building_df["geometry"])
        building_centroids_m = v3._lonlat_to_local_m(building_centroids_lonlat)
        building_proxy_valid = np.ones((len(building_df),), dtype=np.float32)
    lon0 = float(np.nanmedian(plot_centroids_lonlat[:, 0]))
    lat0 = float(np.nanmedian(plot_centroids_lonlat[:, 1]))

    intersection_xy_lonlat = np.stack(
        [
            pd.to_numeric(intersection_df["x"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            pd.to_numeric(intersection_df["y"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    intersection_xy_m = _project_xy(intersection_xy_lonlat, lon0=lon0, lat0=lat0)

    street_u = v1._normalize_id_series(street_df["u"]).to_numpy(dtype=np.int64, copy=True)
    street_v = v1._normalize_id_series(street_df["v"]).to_numpy(dtype=np.int64, copy=True)
    street_segments: List[np.ndarray] = []
    street_orientation_deg = np.zeros((len(street_df),), dtype=np.float32)
    street_start_angle_deg = np.zeros((len(street_df),), dtype=np.float32)
    street_end_angle_deg = np.zeros((len(street_df),), dtype=np.float32)
    for idx, (u_id, v_id) in enumerate(zip(street_u.tolist(), street_v.tolist())):
        u_idx = osmid_to_idx.get(int(u_id))
        v_idx = osmid_to_idx.get(int(v_id))
        if u_idx is None or v_idx is None:
            seg = np.zeros((2, 2), dtype=np.float64)
        else:
            seg = np.stack([intersection_xy_m[u_idx], intersection_xy_m[v_idx]], axis=0).astype(np.float64)
        street_segments.append(seg)
        ori = _segment_orientation_deg(seg[0], seg[1])
        street_orientation_deg[idx] = ori
        street_start_angle_deg[idx] = ori
        street_end_angle_deg[idx] = float((ori + 180.0) % 360.0)

    street_len_map = v3._street_length_map(street_df)
    plot_area = v1._safe_numeric(plot_df.get("plot_area", 0.0))
    building_area = v1._safe_numeric(building_df.get("bid_area", 0.0))
    building_height_log = np.log1p(np.clip(v1._safe_numeric(building_df.get("bid_height", 0.0)), a_min=0.0, a_max=None)).astype(np.float32)
    plot_orientation = v1._safe_numeric(plot_df.get("plot_orientation", 0.0))
    building_orientation_deg = v1._safe_numeric(building_df.get("bid_orientation", 0.0))
    canopy_mean_col = v3._first_existing(plot_df.columns, v3.CANOPY_MEAN_CANDIDATES)
    pop_col = v3._first_existing(plot_df.columns, v3.POPULATION_CANDIDATES)
    built_col = v3._first_existing(plot_df.columns, v3.BUILT_CANDIDATES)
    canopy_mean = v1._safe_numeric(plot_df[canopy_mean_col], 0.0) if canopy_mean_col else np.zeros(len(plot_df), dtype=np.float32)
    population = v1._safe_numeric(plot_df[pop_col], 0.0) if pop_col else np.zeros(len(plot_df), dtype=np.float32)
    built = v1._safe_numeric(plot_df[built_col], 0.0) if built_col else np.zeros(len(plot_df), dtype=np.float32)
    plot_safe_area = np.maximum(plot_area.astype(np.float64), 1.0)
    pop_density = (population.astype(np.float64) / plot_safe_area).astype(np.float32)
    built_density = (built.astype(np.float64) / plot_safe_area).astype(np.float32)
    lcz_group_map = v3._lcz_group_values(plot_df)
    lcz_group = np.stack(
        [
            lcz_group_map["lcz_compact"],
            lcz_group_map["lcz_open"],
            lcz_group_map["lcz_sparse"],
            lcz_group_map["lcz_natural"],
            lcz_group_map["lcz_entropy"],
        ],
        axis=1,
    ).astype(np.float32)

    plot_street_count = np.asarray([len(set(v)) for v in plot_street], dtype=np.int32)
    plot_junction = v3._project_memberships(plot_street, v3._street_endpoint_map(street_df))
    plot_junction_count = np.asarray([len(set(v)) for v in plot_junction], dtype=np.int32)
    plot_building_count = np.asarray([len(set(v)) for v in plot_building_idx], dtype=np.int32)

    inv_street_to_plots = v3._build_inverse_index(plot_street)
    street_adjacent_plot_count = np.asarray([len(set(inv_street_to_plots.get(int(sid), []))) for sid in street_ids.tolist()], dtype=np.int32)
    street_length = v1._safe_numeric(street_df.get("length", 0.0))
    intersection_degree = v1._safe_numeric(intersection_df.get("intersection_degree", 0.0))
    intersection_closeness = v1._safe_numeric(intersection_df.get("Closeness Centrality", 0.0))
    intersection_betweenness = v1._safe_numeric(intersection_df.get("Betweenness Centrality", 0.0))
    street_endpoint_degree_sum = np.zeros((len(street_df),), dtype=np.float32)
    street_dead_end = np.zeros((len(street_df),), dtype=np.float32)
    intersection_incident_street_count = np.zeros((len(intersection_df),), dtype=np.int32)
    for s_idx, (u_id, v_id) in enumerate(zip(street_u.tolist(), street_v.tolist())):
        u_idx = osmid_to_idx.get(int(u_id))
        v_idx = osmid_to_idx.get(int(v_id))
        deg_u = float(intersection_degree[u_idx]) if u_idx is not None else 0.0
        deg_v = float(intersection_degree[v_idx]) if v_idx is not None else 0.0
        street_endpoint_degree_sum[s_idx] = deg_u + deg_v
        street_dead_end[s_idx] = 1.0 if min(deg_u, deg_v) <= 1.0 else 0.0
        if u_idx is not None:
            intersection_incident_street_count[u_idx] += 1
        if v_idx is not None:
            intersection_incident_street_count[v_idx] += 1

    street_x = v1._maybe_standardize(
        v1._street_features(street_df, intersection_df, inv_street_to_plots),
        enable=bool(args.standardize_node_features),
    )
    intersection_x = v1._maybe_standardize(v1._intersection_features(intersection_df), enable=bool(args.standardize_node_features))

    building_parent_plot_count = np.asarray([len(set(building_parent_plots.get(int(b_idx), []))) for b_idx in range(len(building_df))], dtype=np.int32)
    building_street_membership: List[List[int]] = []
    building_parent_plot_street_count = np.zeros((len(building_df),), dtype=np.float32)
    for b_idx in range(len(building_df)):
        parent_plots = sorted(set(building_parent_plots.get(int(b_idx), [])))
        streets = set()
        for p_idx in parent_plots:
            streets.update(plot_street[p_idx])
        building_street_membership.append(sorted(int(x) for x in streets))
        if parent_plots:
            building_parent_plot_street_count[b_idx] = float(np.mean([len(set(plot_street[p_idx])) for p_idx in parent_plots]))

    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else v3._adaptive_entity_cap(
        inv_street_to_plots,
        min_cap=12,
        max_cap=64,
        quantile=float(args.street_cap_quantile),
    )

    edge_plot_plot, edge_attr_plot_plot, plot_plot_stats = _plot_plot_relation(
        plot_street=plot_street,
        street_len_map=street_len_map,
        street_cap=street_cap,
        plot_centroids_m=plot_centroids_m,
        plot_area=plot_area,
        canopy_mean=canopy_mean,
        pop_density=pop_density,
        built_density=built_density,
        lcz_group=lcz_group,
    )
    edge_plot_street, edge_attr_plot_street, edge_street_plot, edge_attr_street_plot = _plot_street_relation(
        plot_street=plot_street,
        plot_centroids_m=plot_centroids_m,
        plot_orientation=plot_orientation,
        plot_street_count=plot_street_count,
        street_id_to_idx=street_id_to_idx,
        street_length=street_length,
        street_adjacent_plot_count=street_adjacent_plot_count,
        street_endpoint_degree_sum=street_endpoint_degree_sum,
        street_orientation_deg=street_orientation_deg,
        street_segments=street_segments,
    )
    edge_street_intersection, edge_attr_street_intersection, edge_intersection_street, edge_attr_intersection_street = _street_intersection_relation(
        street_df=street_df,
        street_id_to_idx=street_id_to_idx,
        osmid_to_idx=osmid_to_idx,
        street_length=street_length,
        street_start_angle_deg=street_start_angle_deg,
        street_end_angle_deg=street_end_angle_deg,
        street_dead_end=street_dead_end,
        intersection_degree=intersection_degree,
        intersection_closeness=intersection_closeness,
        intersection_betweenness=intersection_betweenness,
    )
    edge_plot_building, edge_attr_plot_building, edge_building_plot, edge_attr_building_plot, building_plot_stats = _building_plot_relation(
        plot_building_idx=plot_building_idx,
        plot_centroids_m=plot_centroids_m,
        plot_area=plot_area,
        building_centroids_m=building_centroids_m,
        building_area=building_area,
        building_height_log=building_height_log,
    )
    edge_building_street, edge_attr_building_street, edge_street_building, edge_attr_street_building, building_street_stats = _building_street_relation(
        building_parent_plots=building_parent_plots,
        plot_street=plot_street,
        street_id_to_idx=street_id_to_idx,
        street_length=street_length,
        street_orientation_deg=street_orientation_deg,
        street_segments=street_segments,
        building_centroids_m=building_centroids_m,
        building_orientation_deg=building_orientation_deg,
    )
    edge_building_building, edge_attr_building_building = _building_building_spatial_relation(
        building_centroids_m=building_centroids_m,
        building_orientation_deg=building_orientation_deg,
        building_area=building_area,
        building_height_log=building_height_log,
        building_x=building_x,
        building_parent_plots=building_parent_plots,
        building_street_membership=building_street_membership,
        k=int(args.building_spatial_knn_k),
    )

    edge_building_semantic = np.zeros((2, 0), dtype=np.int64)
    edge_attr_building_semantic = np.zeros((0, _lookup_edge_attr_dim("building__semantic__building")), dtype=np.float32)
    semantic_stats = {"enabled": False, "sigma": 1.0}
    if not bool(args.disable_building_semantic):
        plot_ctx_for_building = np.zeros((len(building_df), 4), dtype=np.float32)
        for b_idx in range(len(building_df)):
            parent_plots = sorted(set(building_parent_plots.get(int(b_idx), [])))
            if not parent_plots:
                continue
            plot_ctx_for_building[b_idx, 0] = float(np.mean(canopy_mean[parent_plots]))
            plot_ctx_for_building[b_idx, 1] = float(np.mean(pop_density[parent_plots]))
            plot_ctx_for_building[b_idx, 2] = float(np.mean(built_density[parent_plots]))
            plot_ctx_for_building[b_idx, 3] = float(np.mean(plot_street_count[parent_plots]))
        semantic_feat = np.concatenate(
            [
                building_x.astype(np.float32),
                plot_ctx_for_building.astype(np.float32),
                np.log1p(building_parent_plot_count.astype(np.float32)).reshape(-1, 1),
                np.log1p(building_parent_plot_street_count.astype(np.float32)).reshape(-1, 1),
            ],
            axis=1,
        )
        edge_building_semantic, edge_attr_building_semantic, semantic_extra = _building_building_semantic_relation(
            semantic_feat=semantic_feat,
            building_x=building_x,
            building_parent_plots=building_parent_plots,
            building_street_membership=building_street_membership,
            k=int(args.building_semantic_knn_k),
        )
        semantic_stats = {"enabled": True, **semantic_extra}

    node_types_meta = {
        "plot": {
            "num_nodes": int(len(plot_df)),
            "feature_columns": v1.PLOT_FEATURE_NAMES,
            "feature_file": v1._save_array(out_dir, "plot_feat", plot_x.astype(np.float32)),
            "id_file": v1._save_array(out_dir, "plot_node_ids", plot_ids),
            "label_file": v1._save_array(out_dir, "plot_label", plot_y.astype(np.int64)),
            "node_table_file": "plot_node_table.csv",
        },
        "street": {
            "num_nodes": int(len(street_df)),
            "feature_columns": v1.STREET_FEATURE_NAMES,
            "feature_file": v1._save_array(out_dir, "street_feat", street_x.astype(np.float32)),
            "id_file": v1._save_array(out_dir, "street_node_ids", street_ids),
            "node_table_file": "street_node_table.csv",
        },
        "intersection": {
            "num_nodes": int(len(intersection_df)),
            "feature_columns": v1.INTERSECTION_FEATURE_NAMES,
            "feature_file": v1._save_array(out_dir, "intersection_feat", intersection_x.astype(np.float32)),
            "id_file": v1._save_array(out_dir, "intersection_node_ids", osmid_ids),
            "node_table_file": "intersection_node_table.csv",
        },
        "building": {
            "num_nodes": int(len(building_df)),
            "feature_columns": v1.BUILDING_FEATURE_NAMES,
            "feature_file": v1._save_array(out_dir, "building_feat", building_x.astype(np.float32)),
            "id_file": v1._save_array(out_dir, "building_node_ids", building_ids),
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
            "u": street_u,
            "v": street_v,
            "length": street_length,
            "adjacent_plot_count": street_adjacent_plot_count,
            "endpoint_degree_sum": street_endpoint_degree_sum,
            "street_orientation_deg": street_orientation_deg,
        }
    ).to_csv(out_dir / "street_node_table.csv", index=False)
    pd.DataFrame(
        {
            "node_idx": np.arange(len(intersection_df), dtype=np.int64),
            "intersection_id": v1._normalize_id_series(pd.Series(intersection_df.get("intersection_id", np.arange(len(intersection_df))))),
            "osmid": osmid_ids,
            "incident_street_count": intersection_incident_street_count,
            "degree": intersection_degree,
        }
    ).to_csv(out_dir / "intersection_node_table.csv", index=False)
    pd.DataFrame(
        {
            "node_idx": np.arange(len(building_df), dtype=np.int64),
            "bid": building_ids,
            "parent_plot_count": building_parent_plot_count,
            "parent_plot_street_count_mean": building_parent_plot_street_count,
            "bid_height": v1._safe_numeric(building_df.get("bid_height", 0.0)),
        }
    ).to_csv(out_dir / "building_node_table.csv", index=False)

    edge_types_meta: Dict[str, dict] = {}
    _register_edge(edge_types_meta, out_dir, "plot__shares_street__plot", "plot", "shares_street", "plot", edge_plot_plot, edge_attr_plot_plot, "plot__shares_street__plot")
    _register_edge(edge_types_meta, out_dir, "plot__bounded_by__street", "plot", "bounded_by", "street", edge_plot_street, edge_attr_plot_street, "street__rev_bounded_by__plot")
    _register_edge(edge_types_meta, out_dir, "street__rev_bounded_by__plot", "street", "rev_bounded_by", "plot", edge_street_plot, edge_attr_street_plot, "plot__bounded_by__street")
    _register_edge(edge_types_meta, out_dir, "street__connects__intersection", "street", "connects", "intersection", edge_street_intersection, edge_attr_street_intersection, "intersection__rev_connects__street")
    _register_edge(edge_types_meta, out_dir, "intersection__rev_connects__street", "intersection", "rev_connects", "street", edge_intersection_street, edge_attr_intersection_street, "street__connects__intersection")
    _register_edge(edge_types_meta, out_dir, "plot__contains__building", "plot", "contains", "building", edge_plot_building, edge_attr_plot_building, "building__rev_contains__plot")
    _register_edge(edge_types_meta, out_dir, "building__rev_contains__plot", "building", "rev_contains", "plot", edge_building_plot, edge_attr_building_plot, "plot__contains__building")
    _register_edge(edge_types_meta, out_dir, "building__proximal_to__street", "building", "proximal_to", "street", edge_building_street, edge_attr_building_street, "street__rev_proximal_to__building")
    _register_edge(edge_types_meta, out_dir, "street__rev_proximal_to__building", "street", "rev_proximal_to", "building", edge_street_building, edge_attr_street_building, "building__proximal_to__street")
    _register_edge(edge_types_meta, out_dir, "building__near__building", "building", "near", "building", edge_building_building, edge_attr_building_building, "building__near__building")
    if edge_building_semantic.shape[1] > 0:
        _register_edge(edge_types_meta, out_dir, "building__semantic__building", "building", "semantic", "building", edge_building_semantic, edge_attr_building_semantic, "building__semantic__building")

    label_counts = {str(k): int(v) for k, v in zip(*np.unique(plot_y, return_counts=True))}
    meta = {
        "dataset_name": dataset_name,
        "city": city,
        "dataset_family": "urban_plot_hetero_v2",
        "target_node_type": "plot",
        "target_task": "plot_label_for_evaluation",
        "notes": [
            "V2 preserves the four-node-type urban heterograph and upgrades relations to typed edge_attr rather than only relation IDs.",
            "Edge features are relation-specific and focus on geometry, containment, accessibility, topology, or similarity.",
            "Building semantic edges remain separate from spatial building-building edges.",
            "VHR/SVI building embeddings are not available in the current raw city package, so semantic edges use morphology and parent-plot context; visual fields are emitted as explicit missing placeholders.",
        ],
        "node_type_order": NODE_TYPE_ORDER,
        "node_types": node_types_meta,
        "edge_types": edge_types_meta,
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "label_counts": label_counts,
        "builder_args": {
            "building_spatial_knn_k": int(args.building_spatial_knn_k),
            "building_semantic_knn_k": int(args.building_semantic_knn_k),
            "building_semantic_enabled": bool(not args.disable_building_semantic),
            "proxy_building_centroid_from_plot": bool(args.proxy_building_centroid_from_plot),
            "street_entity_cap": int(street_cap),
            "street_cap_quantile": float(args.street_cap_quantile),
            "standardize_node_features": bool(args.standardize_node_features),
        },
        "relation_stats": {
            "plot_plot_shared_street": plot_plot_stats,
            "plot_building_reference_mode": plot_building_ref_stats,
            "plot_building_touch_ratio": building_plot_stats,
            "building_street_selection": building_street_stats,
            "building_semantic": semantic_stats,
            "building_centroid_proxy_coverage": {
                "enabled": bool(args.proxy_building_centroid_from_plot),
                "valid_ratio": float(building_proxy_valid.mean()) if len(building_proxy_valid) > 0 else 0.0,
            },
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
        f"building-street={edge_building_street.shape[1]} "
        f"building-building-spatial={edge_building_building.shape[1]} "
        f"building-building-semantic={edge_building_semantic.shape[1]}"
    )


if __name__ == "__main__":
    main()
