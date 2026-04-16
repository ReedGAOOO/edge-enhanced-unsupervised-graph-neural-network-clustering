#!/usr/bin/env python3
"""
Build a semantics-first plot graph from raw urban city data.

V3 design principles:
- edge existence must come from interpretable urban relations
- feature similarity does not create edges
- building overlap is retained as an auxiliary edge attribute, not a primary edge source
- geometric proximity is used only as a fallback for structurally under-connected plots

Input folder example (per city):
  urban_network_datasets/<city>/
    plot.parquet
    street.parquet
    plot_street_id.npy
    plot_bid.npy

Output dataset folder:
  <out_root>/<dataset_name>/
    <dataset_name>_feat.npy
    <dataset_name>_label.npy
    <dataset_name>_edge_index.npy
    <dataset_name>_edge_attr.npy
    <dataset_name>_edge_weight.npy
    <dataset_name>_meta.json
    <dataset_name>_node_table.csv
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


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

POPULATION_CANDIDATES = ["2025_pop", "PopSum", "population"]
BUILT_CANDIDATES = ["2025_built"]
CANOPY_MEAN_CANDIDATES = ["canopy_mean", "Canopy height_mean"]
CANOPY_STD_CANDIDATES = ["canopy_stdev", "Canopy height_std"]

EDGE_FEATURE_NAMES = [
    "rel_shared_street",
    "rel_shared_junction",
    "rel_geom_fallback",
    "shared_street_cnt",
    "shared_junction_cnt",
    "shared_building_cnt",
    "shared_street_len_mean",
    "jaccard_street",
    "jaccard_junction",
    "jaccard_building",
    "centroid_dist_m",
    "geom_score",
    "log_area_gap",
    "population_density_gap",
    "built_density_gap",
    "canopy_gap",
    "street_count_i",
    "street_count_j",
    "junction_count_i",
    "junction_count_j",
    "building_count_i",
    "building_count_j",
]

PAIR_REL_STREET = 0
PAIR_REL_JUNCTION = 1
PAIR_REL_GEOM = 2
PAIR_STREET_CNT = 3
PAIR_STREET_LEN_SUM = 4
PAIR_JUNCTION_CNT = 5
PAIR_BUILDING_CNT = 6
PAIR_GEOM_DIST_SUM = 7
PAIR_GEOM_DIST_CNT = 8
PAIR_STAT_DIM = 9

V3_VARIANTS = {
    "v3s": {
        "description": "Street-boundary graph only. Edges connect plots sharing at least one street segment.",
        "use_street": True,
        "use_junction": False,
        "use_geom_fallback": False,
    },
    "v3sj": {
        "description": "Street-boundary graph plus same-junction context edges, with relation-aware pruning.",
        "use_street": True,
        "use_junction": True,
        "use_geom_fallback": False,
    },
    "v3sjg": {
        "description": "Street-boundary graph plus same-junction edges, with geometric fallback edges only for under-connected plots.",
        "use_street": True,
        "use_junction": True,
        "use_geom_fallback": True,
    },
}


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return ""


def _safe_id_list(v, zero_scalar_empty: bool = False) -> List[int]:
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if zero_scalar_empty and isinstance(v, (int, np.integer)) and int(v) == 0:
        return []
    if isinstance(v, np.ndarray):
        raw = v.tolist()
    elif isinstance(v, (list, tuple, set)):
        raw = list(v)
    else:
        raw = [v]

    out: List[int] = []
    stack = list(raw)
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, float) and np.isnan(cur):
            continue
        if zero_scalar_empty and isinstance(cur, (int, np.integer)) and int(cur) == 0:
            continue
        if isinstance(cur, (list, tuple, set, np.ndarray)):
            stack.extend(list(cur))
            continue
        try:
            out.append(int(cur))
        except Exception:
            continue
    out.reverse()
    return out


def _load_membership_array(fp: Path, n: int, zero_scalar_empty: bool = False) -> List[List[int]]:
    arr = np.load(fp, allow_pickle=True)
    if arr.ndim != 1:
        raise ValueError(f"{fp} must be 1D object array, got shape {arr.shape}")
    if len(arr) != n:
        raise ValueError(f"{fp} length mismatch: len={len(arr)} vs expected {n}")
    return [_safe_id_list(x, zero_scalar_empty=zero_scalar_empty) for x in arr]


def _build_inverse_index(memberships: Sequence[Sequence[int]]) -> Dict[int, List[int]]:
    inv: Dict[int, List[int]] = defaultdict(list)
    for node_idx, ids in enumerate(memberships):
        for x in ids:
            inv[x].append(node_idx)
    return inv


def _street_length_map(street_df: pd.DataFrame) -> Dict[int, float]:
    if not {"street_id", "length"}.issubset(set(street_df.columns)):
        return {}
    g = street_df.groupby("street_id", as_index=False)["length"].mean()
    out: Dict[int, float] = {}
    for _, row in g.iterrows():
        try:
            out[int(row["street_id"])] = float(row["length"])
        except Exception:
            continue
    return out


def _street_endpoint_map(street_df: pd.DataFrame) -> Dict[int, List[int]]:
    if not {"street_id", "u", "v"}.issubset(set(street_df.columns)):
        return {}
    endpoints: Dict[int, set[int]] = defaultdict(set)
    for row in street_df[["street_id", "u", "v"]].dropna().itertuples(index=False):
        try:
            sid = int(row.street_id)
            endpoints[sid].add(int(row.u))
            endpoints[sid].add(int(row.v))
        except Exception:
            continue
    return {k: sorted(v) for k, v in endpoints.items()}


def _project_memberships(
    src_memberships: Sequence[Sequence[int]],
    relation_map: Dict[int, Sequence[int]],
) -> List[List[int]]:
    out: List[List[int]] = []
    for ids in src_memberships:
        agg: set[int] = set()
        for entity_id in ids:
            agg.update(relation_map.get(entity_id, ()))
        out.append(sorted(agg))
    return out


def _adaptive_entity_cap(
    inv_index: Dict[int, List[int]],
    min_cap: int,
    max_cap: int,
    quantile: float,
) -> int:
    sizes = np.asarray([len(set(nodes)) for nodes in inv_index.values() if len(set(nodes)) >= 2], dtype=np.int32)
    if sizes.size == 0:
        return int(min_cap)
    cap = int(math.ceil(float(np.quantile(sizes, quantile))))
    cap = max(cap, int(np.quantile(sizes, 0.95)))
    return int(np.clip(cap, min_cap, max_cap))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x.copy()
    x[~np.isfinite(x)] = np.nan
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu).astype(np.float64)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(mu, inds[1])
    std = np.std(x, axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    z = (x - mu) / std
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _read_u32(buf: bytes, offset: int, endian: str) -> Tuple[int, int]:
    return int(struct.unpack_from(f"{endian}I", buf, offset)[0]), offset + 4


def _read_xy_block(buf: bytes, offset: int, endian: str, n_points: int) -> Tuple[np.ndarray, int]:
    if n_points <= 0:
        return np.zeros((0, 2), dtype=np.float64), offset
    arr = np.frombuffer(buf, dtype=np.dtype(f"{endian}f8"), count=2 * n_points, offset=offset)
    arr = np.asarray(arr, dtype=np.float64).reshape(n_points, 2).copy()
    return arr, offset + 16 * n_points


def _parse_wkb(buf: bytes, offset: int = 0) -> Tuple[Dict[str, object], int]:
    byte_order = buf[offset]
    if byte_order == 1:
        endian = "<"
    elif byte_order == 0:
        endian = ">"
    else:
        raise ValueError(f"Unsupported WKB byte order: {byte_order}")
    offset += 1

    geom_type, offset = _read_u32(buf, offset, endian)
    if geom_type == 3:  # Polygon
        n_rings, offset = _read_u32(buf, offset, endian)
        rings: List[np.ndarray] = []
        for _ in range(n_rings):
            n_points, offset = _read_u32(buf, offset, endian)
            coords, offset = _read_xy_block(buf, offset, endian, n_points)
            rings.append(coords)
        return {"type": "Polygon", "rings": rings}, offset

    if geom_type == 6:  # MultiPolygon
        n_geoms, offset = _read_u32(buf, offset, endian)
        polys: List[List[np.ndarray]] = []
        for _ in range(n_geoms):
            child, offset = _parse_wkb(buf, offset)
            if child["type"] == "Polygon":
                polys.append(child["rings"])
        return {"type": "MultiPolygon", "polys": polys}, offset

    raise ValueError(f"Unsupported WKB geometry type: {geom_type}")


def _polygon_centroid_area(coords: np.ndarray) -> Tuple[float, float, float]:
    if coords.shape[0] < 3:
        if coords.shape[0] == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])), 0.0
    pts = coords
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    x1 = pts[:-1, 0]
    y1 = pts[:-1, 1]
    x2 = pts[1:, 0]
    y2 = pts[1:, 1]
    cross = x1 * y2 - x2 * y1
    area2 = float(np.sum(cross))
    area = area2 / 2.0
    if abs(area) < 1e-12:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), 0.0
    cx = float(np.sum((x1 + x2) * cross) / (3.0 * area2))
    cy = float(np.sum((y1 + y2) * cross) / (3.0 * area2))
    return cx, cy, abs(area)


def _geometry_centroids_from_wkb(series: pd.Series) -> np.ndarray:
    out = np.zeros((len(series), 2), dtype=np.float64)
    for idx, raw in enumerate(series):
        if not raw:
            continue
        try:
            geom, _ = _parse_wkb(bytes(raw))
        except Exception:
            continue
        if geom["type"] == "Polygon":
            rings = geom["rings"]
            if not rings:
                continue
            cx, cy, _ = _polygon_centroid_area(rings[0])
            out[idx] = [cx, cy]
        elif geom["type"] == "MultiPolygon":
            accum_area = 0.0
            accum_x = 0.0
            accum_y = 0.0
            for rings in geom["polys"]:
                if not rings:
                    continue
                cx, cy, area = _polygon_centroid_area(rings[0])
                w = max(area, 1e-9)
                accum_x += w * cx
                accum_y += w * cy
                accum_area += w
            if accum_area > 0:
                out[idx] = [accum_x / accum_area, accum_y / accum_area]
    return out


def _lonlat_to_local_m(xy_lonlat: np.ndarray) -> np.ndarray:
    lon = xy_lonlat[:, 0].astype(np.float64, copy=False)
    lat = xy_lonlat[:, 1].astype(np.float64, copy=False)
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    cos_lat = math.cos(math.radians(lat0))
    x = (lon - lon0) * 111_320.0 * cos_lat
    y = (lat - lat0) * 110_540.0
    out = np.stack([x, y], axis=1)
    out[~np.isfinite(out)] = 0.0
    return out


def _lcz_group_values(plot_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    cols = [c for c in plot_df.columns if c.startswith("lcz_")]
    if not cols:
        z = np.zeros((len(plot_df),), dtype=np.float32)
        return {
            "lcz_compact": z.copy(),
            "lcz_open": z.copy(),
            "lcz_sparse": z.copy(),
            "lcz_natural": z.copy(),
            "lcz_entropy": z.copy(),
        }
    numeric = plot_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def _sum(group_cols: Sequence[str]) -> np.ndarray:
        good = [c for c in group_cols if c in numeric.columns]
        if not good:
            return np.zeros((len(plot_df),), dtype=np.float32)
        return numeric[good].sum(axis=1).to_numpy(dtype=np.float32)

    compact = _sum(["lcz_1", "lcz_2", "lcz_3"])
    open_built = _sum(["lcz_4", "lcz_5", "lcz_6"])
    sparse = _sum(["lcz_7", "lcz_8", "lcz_9", "lcz_10"])
    natural = _sum(["lcz_11", "lcz_12", "lcz_13", "lcz_14", "lcz_15", "lcz_16", "lcz_17"])

    probs = numeric.to_numpy(dtype=np.float32)
    probs = np.clip(probs, a_min=0.0, a_max=None)
    total = probs.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    probs = probs / total
    entropy = -np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0), axis=1).astype(np.float32)
    return {
        "lcz_compact": compact,
        "lcz_open": open_built,
        "lcz_sparse": sparse,
        "lcz_natural": natural,
        "lcz_entropy": entropy,
    }


def _build_node_features(
    plot_df: pd.DataFrame,
    plot_street: Sequence[Sequence[int]],
    plot_bid: Sequence[Sequence[int]],
    plot_junction: Sequence[Sequence[int]],
    street_len_map: Dict[int, float],
    standardize: bool,
    feature_clip: float,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    n = len(plot_df)
    if n == 0:
        raise ValueError("plot.parquet has no rows.")

    population_col = _first_existing(plot_df.columns, POPULATION_CANDIDATES)
    built_col = _first_existing(plot_df.columns, BUILT_CANDIDATES)
    canopy_mean_col = _first_existing(plot_df.columns, CANOPY_MEAN_CANDIDATES)
    canopy_std_col = _first_existing(plot_df.columns, CANOPY_STD_CANDIDATES)

    area = pd.to_numeric(plot_df.get("plot_area", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    perimeter = pd.to_numeric(plot_df.get("plot_perimeter", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    convexity = pd.to_numeric(plot_df.get("plot_convexity", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    corners = pd.to_numeric(plot_df.get("plot_corners", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    elongation = pd.to_numeric(plot_df.get("plot_elongation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    orientation = pd.to_numeric(plot_df.get("plot_orientation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    rectangularity = pd.to_numeric(plot_df.get("plot_rectangularity", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    complexity = pd.to_numeric(plot_df.get("plot_complexity", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    canopy_mean = (
        pd.to_numeric(plot_df[canopy_mean_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if canopy_mean_col
        else np.zeros((n,), dtype=np.float32)
    )
    canopy_std = (
        pd.to_numeric(plot_df[canopy_std_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if canopy_std_col
        else np.zeros((n,), dtype=np.float32)
    )
    population = (
        pd.to_numeric(plot_df[population_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if population_col
        else np.zeros((n,), dtype=np.float32)
    )
    built = (
        pd.to_numeric(plot_df[built_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if built_col
        else np.zeros((n,), dtype=np.float32)
    )

    safe_area = np.maximum(area, 1.0)
    pop_density = population / safe_area
    built_density = built / safe_area

    street_count = np.asarray([len(set(v)) for v in plot_street], dtype=np.float32)
    building_count = np.asarray([len(set(v)) for v in plot_bid], dtype=np.float32)
    junction_count = np.asarray([len(set(v)) for v in plot_junction], dtype=np.float32)

    attached_street_len_mean = np.zeros((n,), dtype=np.float32)
    for idx, ids in enumerate(plot_street):
        if not ids:
            continue
        vals = [float(street_len_map.get(int(sid), 0.0)) for sid in ids]
        if vals:
            attached_street_len_mean[idx] = float(np.mean(vals))

    lcz = _lcz_group_values(plot_df)
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
        "log_street_count": np.log1p(np.clip(street_count, a_min=0.0, a_max=None)),
        "log_junction_count": np.log1p(np.clip(junction_count, a_min=0.0, a_max=None)),
        "log_building_count": np.log1p(np.clip(building_count, a_min=0.0, a_max=None)),
        "log_attached_street_len_mean": np.log1p(np.clip(attached_street_len_mean, a_min=0.0, a_max=None)),
    }

    feature_cols = list(feature_map.keys())
    x = np.stack([feature_map[name].astype(np.float32) for name in feature_cols], axis=1)
    if standardize:
        x = _zscore(x)
    if feature_clip > 0:
        x = np.clip(x, -float(feature_clip), float(feature_clip))
    x[~np.isfinite(x)] = 0.0

    aux = {
        "area_log": feature_map["log_plot_area"].astype(np.float32),
        "pop_density_log": feature_map["log_population_density"].astype(np.float32),
        "built_density_log": feature_map["log_built_density"].astype(np.float32),
        "canopy_mean": canopy_mean.astype(np.float32),
        "street_count": street_count.astype(np.float32),
        "junction_count": junction_count.astype(np.float32),
        "building_count": building_count.astype(np.float32),
        "population_col": np.asarray([population_col] * n, dtype=object),
        "built_col": np.asarray([built_col] * n, dtype=object),
        "canopy_mean_col": np.asarray([canopy_mean_col] * n, dtype=object),
        "canopy_std_col": np.asarray([canopy_std_col] * n, dtype=object),
    }
    return x.astype(np.float32), feature_cols, aux


def _build_labels(plot_df: pd.DataFrame, label_mode: str) -> Tuple[np.ndarray, Dict[str, int] | None]:
    if label_mode == "zeros":
        return np.zeros(len(plot_df), dtype=np.int64), None
    available = [c for c in LAND_USE_CANDIDATES if c in plot_df.columns]
    if not available:
        return np.zeros(len(plot_df), dtype=np.int64), None
    scores = plot_df[available].fillna(0).to_numpy(dtype=np.float32)
    unknown_idx = len(available)
    y = np.argmax(scores, axis=1).astype(np.int64)
    y[np.max(scores, axis=1) <= 0] = unknown_idx
    mapping = {name: idx for idx, name in enumerate(available)}
    mapping["unknown"] = unknown_idx
    return y, mapping


def _accumulate_relation_pairs(
    inv_index: Dict[int, List[int]],
    max_nodes_per_entity: int,
    pair_stats: Dict[Tuple[int, int], List[float]],
    relation: str,
    entity_value: Dict[int, float] | None = None,
) -> Dict[str, int]:
    stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}
    for entity_id, nodes in inv_index.items():
        stats["entities_total"] += 1
        uniq = sorted(set(nodes))
        m = len(uniq)
        if m < 2:
            continue
        if m > max_nodes_per_entity:
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        value = float(entity_value.get(entity_id, 0.0)) if entity_value is not None else 0.0
        for i, j in combinations(uniq, 2):
            key = (i, j)
            rec = pair_stats.setdefault(key, [0.0] * PAIR_STAT_DIM)
            if relation == "street":
                rec[PAIR_REL_STREET] = 1.0
                rec[PAIR_STREET_CNT] += 1.0
                rec[PAIR_STREET_LEN_SUM] += value
            elif relation == "junction":
                rec[PAIR_REL_JUNCTION] = 1.0
                rec[PAIR_JUNCTION_CNT] += 1.0
            else:
                raise ValueError(f"Unknown relation: {relation}")
    return stats


def _structural_degree(num_nodes: int, pair_stats: Dict[Tuple[int, int], List[float]]) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.int32)
    for i, j in pair_stats.keys():
        deg[i] += 1
        deg[j] += 1
    return deg


def _estimate_geom_radius(centroids_m: np.ndarray) -> float:
    if centroids_m.shape[0] < 2:
        return 0.0
    tree = cKDTree(centroids_m)
    dists, _ = tree.query(centroids_m, k=min(2, centroids_m.shape[0]))
    if dists.ndim == 1 or dists.shape[1] < 2:
        return 0.0
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0
    return float(np.quantile(nn, 0.95) * 2.0)


def _add_geom_fallback_pairs(
    centroids_m: np.ndarray,
    structural_deg: np.ndarray,
    pair_stats: Dict[Tuple[int, int], List[float]],
    geom_k: int,
    geom_max_struct_deg: int,
    geom_radius_m: float,
) -> Dict[str, int]:
    stats = {"geom_candidate_nodes": 0, "geom_pairs_inserted": 0, "geom_radius_m": float(geom_radius_m)}
    n = centroids_m.shape[0]
    if n < 2 or geom_k <= 0:
        return stats
    tree = cKDTree(centroids_m)
    candidate_nodes = np.where(structural_deg <= int(geom_max_struct_deg))[0]
    stats["geom_candidate_nodes"] = int(candidate_nodes.shape[0])
    if candidate_nodes.size == 0:
        return stats

    max_k = min(int(geom_k) + 8, n)
    for i in candidate_nodes.tolist():
        dists, nbrs = tree.query(centroids_m[i], k=max_k)
        dists = np.atleast_1d(dists)
        nbrs = np.atleast_1d(nbrs)
        added = 0
        for dist, j in zip(dists[1:], nbrs[1:]):
            j = int(j)
            if i == j:
                continue
            if geom_radius_m > 0 and float(dist) > float(geom_radius_m):
                continue
            key = (i, j) if i < j else (j, i)
            if key in pair_stats:
                continue
            rec = pair_stats.setdefault(key, [0.0] * PAIR_STAT_DIM)
            rec[PAIR_REL_GEOM] = 1.0
            rec[PAIR_GEOM_DIST_SUM] += float(dist)
            rec[PAIR_GEOM_DIST_CNT] += 1.0
            stats["geom_pairs_inserted"] += 1
            added += 1
            if added >= int(geom_k):
                break
    return stats


def _augment_shared_building_attr(
    inv_building: Dict[int, List[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    max_nodes_per_entity: int,
) -> Dict[str, int]:
    stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0, "pairs_augmented": 0}
    pair_keys = set(pair_stats.keys())
    if not pair_keys:
        return stats
    for _, nodes in inv_building.items():
        stats["entities_total"] += 1
        uniq = sorted(set(nodes))
        m = len(uniq)
        if m < 2:
            continue
        if m > max_nodes_per_entity:
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        for i, j in combinations(uniq, 2):
            key = (i, j)
            if key not in pair_keys:
                continue
            pair_stats[key][PAIR_BUILDING_CNT] += 1.0
            stats["pairs_augmented"] += 1
    return stats


def _junction_only_keep_mask(
    src: np.ndarray,
    dst: np.ndarray,
    rel_street: np.ndarray,
    rel_geom: np.ndarray,
    junction_cnt: np.ndarray,
    centroid_dist: np.ndarray,
    junction_topk: int,
) -> np.ndarray:
    keep = np.zeros((src.shape[0],), dtype=bool)
    always = (rel_street > 0) | (rel_geom > 0)
    keep[always] = True
    if junction_topk <= 0:
        keep |= ~always
        return keep

    junction_only = (~always) & (junction_cnt > 0)
    if not junction_only.any():
        return keep

    incident: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for eid in np.where(junction_only)[0].tolist():
        score = float(junction_cnt[eid])
        dist = float(centroid_dist[eid])
        incident[int(src[eid])].append((-score, dist, eid))
        incident[int(dst[eid])].append((-score, dist, eid))
    for items in incident.values():
        items.sort()
        for _, _, eid in items[: int(junction_topk)]:
            keep[eid] = True
    return keep


def _finalize_edges(
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    aux: Dict[str, np.ndarray],
    geom_radius_m: float,
    junction_topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if not pair_stats:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {"street": 0, "junction": 0, "geom": 0},
        )

    pairs = np.asarray(list(pair_stats.keys()), dtype=np.int64)
    vals = np.asarray(list(pair_stats.values()), dtype=np.float32)
    src = pairs[:, 0]
    dst = pairs[:, 1]

    rel_street = vals[:, PAIR_REL_STREET]
    rel_junction = vals[:, PAIR_REL_JUNCTION]
    rel_geom = vals[:, PAIR_REL_GEOM]
    street_cnt = vals[:, PAIR_STREET_CNT]
    street_len_mean = np.divide(vals[:, PAIR_STREET_LEN_SUM], np.maximum(street_cnt, 1.0))
    junction_cnt = vals[:, PAIR_JUNCTION_CNT]
    building_cnt = vals[:, PAIR_BUILDING_CNT]
    geom_dist = np.divide(vals[:, PAIR_GEOM_DIST_SUM], np.maximum(vals[:, PAIR_GEOM_DIST_CNT], 1.0))

    street_deg = aux["street_count"].astype(np.float32)
    junction_deg = aux["junction_count"].astype(np.float32)
    building_deg = aux["building_count"].astype(np.float32)

    jaccard_street = np.divide(street_cnt, np.maximum(street_deg[src] + street_deg[dst] - street_cnt, 1.0))
    jaccard_junction = np.divide(junction_cnt, np.maximum(junction_deg[src] + junction_deg[dst] - junction_cnt, 1.0))
    jaccard_building = np.divide(building_cnt, np.maximum(building_deg[src] + building_deg[dst] - building_cnt, 1.0))

    diff_area = np.abs(aux["area_log"][src] - aux["area_log"][dst]).astype(np.float32)
    diff_pop = np.abs(aux["pop_density_log"][src] - aux["pop_density_log"][dst]).astype(np.float32)
    diff_built = np.abs(aux["built_density_log"][src] - aux["built_density_log"][dst]).astype(np.float32)
    diff_canopy = np.abs(aux["canopy_mean"][src] - aux["canopy_mean"][dst]).astype(np.float32)

    if not np.any(geom_dist > 0):
        dx = centroids_m[src, 0] - centroids_m[dst, 0]
        dy = centroids_m[src, 1] - centroids_m[dst, 1]
        centroid_dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    else:
        centroid_dist = geom_dist.copy().astype(np.float32)
        missing = centroid_dist <= 0
        if np.any(missing):
            dx = centroids_m[src[missing], 0] - centroids_m[dst[missing], 0]
            dy = centroids_m[src[missing], 1] - centroids_m[dst[missing], 1]
            centroid_dist[missing] = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    scale = max(float(geom_radius_m), float(np.quantile(centroid_dist, 0.5)) if centroid_dist.size else 1.0, 1.0)
    geom_score = np.exp(-centroid_dist / scale).astype(np.float32)

    keep = _junction_only_keep_mask(
        src=src,
        dst=dst,
        rel_street=rel_street,
        rel_geom=rel_geom,
        junction_cnt=junction_cnt,
        centroid_dist=centroid_dist,
        junction_topk=junction_topk,
    )
    src = src[keep]
    dst = dst[keep]
    rel_street = rel_street[keep]
    rel_junction = rel_junction[keep]
    rel_geom = rel_geom[keep]
    street_cnt = street_cnt[keep]
    street_len_mean = street_len_mean[keep]
    junction_cnt = junction_cnt[keep]
    building_cnt = building_cnt[keep]
    jaccard_street = jaccard_street[keep]
    jaccard_junction = jaccard_junction[keep]
    jaccard_building = jaccard_building[keep]
    diff_area = diff_area[keep]
    diff_pop = diff_pop[keep]
    diff_built = diff_built[keep]
    diff_canopy = diff_canopy[keep]
    centroid_dist = centroid_dist[keep]
    geom_score = geom_score[keep]

    edge_attr = np.stack(
        [
            rel_street,
            rel_junction,
            rel_geom,
            street_cnt,
            junction_cnt,
            building_cnt,
            street_len_mean,
            jaccard_street,
            jaccard_junction,
            jaccard_building,
            centroid_dist,
            geom_score,
            diff_area,
            diff_pop,
            diff_built,
            diff_canopy,
            street_deg[src],
            street_deg[dst],
            junction_deg[src],
            junction_deg[dst],
            building_deg[src],
            building_deg[dst],
        ],
        axis=1,
    ).astype(np.float32)

    sc_norm = street_cnt / max(float(np.max(street_cnt)), 1.0) if street_cnt.size else street_cnt
    jc_norm = junction_cnt / max(float(np.max(junction_cnt)), 1.0) if junction_cnt.size else junction_cnt
    sl_norm = street_len_mean / max(float(np.max(street_len_mean)), 1.0) if street_len_mean.size else street_len_mean

    weight = (
        rel_street * (0.72 + 0.18 * sc_norm + 0.10 * sl_norm)
        + rel_junction * (0.34 + 0.26 * jc_norm)
        + rel_geom * (0.12 + 0.18 * geom_score)
    ).astype(np.float32)
    weight = np.clip(weight, 1e-6, None)

    relation_counts = {
        "street": int(np.sum(rel_street > 0)),
        "junction": int(np.sum((rel_junction > 0) & (rel_street <= 0))),
        "geom": int(np.sum(rel_geom > 0)),
    }

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    weight = np.concatenate([weight, weight], axis=0).astype(np.float32)
    return edge_index, edge_attr, weight, relation_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare semantics-first urban plot graph V3.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="", help="Default: urban_<city>_plot_<variant>")
    parser.add_argument("--variant", type=str, default="v3sjg", choices=sorted(V3_VARIANTS.keys()))
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--street_entity_cap", type=int, default=0, help="0 = adaptive")
    parser.add_argument("--junction_entity_cap", type=int, default=0, help="0 = adaptive")
    parser.add_argument("--building_attr_entity_cap", type=int, default=0, help="0 = adaptive")
    parser.add_argument("--street_cap_quantile", type=float, default=0.995)
    parser.add_argument("--junction_cap_quantile", type=float, default=0.995)
    parser.add_argument("--building_cap_quantile", type=float, default=0.99)
    parser.add_argument("--geom_k", type=int, default=3)
    parser.add_argument("--geom_max_struct_deg", type=int, default=1)
    parser.add_argument("--geom_radius_scale", type=float, default=1.0)
    parser.add_argument("--junction_topk", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    city = args.city.strip()
    variant = str(args.variant).strip().lower()
    spec = V3_VARIANTS[variant]
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_{variant}"

    urban_root = Path(args.urban_root).resolve()
    city_dir = urban_root / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")

    required = ["plot.parquet", "street.parquet", "plot_street_id.npy", "plot_bid.npy"]
    for fn in required:
        fp = city_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    out_dir = Path(args.out_root).resolve() / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / dataset_name
    out_files = [
        base.with_name(f"{dataset_name}_feat.npy"),
        base.with_name(f"{dataset_name}_label.npy"),
        base.with_name(f"{dataset_name}_edge_index.npy"),
        base.with_name(f"{dataset_name}_edge_attr.npy"),
        base.with_name(f"{dataset_name}_edge_weight.npy"),
        base.with_name(f"{dataset_name}_meta.json"),
        base.with_name(f"{dataset_name}_node_table.csv"),
    ]
    if (not args.force) and any(p.exists() for p in out_files):
        exists = [str(p) for p in out_files if p.exists()]
        raise FileExistsError(f"Output files already exist. Use --force. Existing: {exists}")

    plot_df = pd.read_parquet(city_dir / "plot.parquet")
    street_df = pd.read_parquet(city_dir / "street.parquet")
    n = len(plot_df)
    if n <= 0:
        raise ValueError("plot.parquet has no rows.")

    plot_street = _load_membership_array(city_dir / "plot_street_id.npy", n, zero_scalar_empty=False)
    plot_bid = _load_membership_array(city_dir / "plot_bid.npy", n, zero_scalar_empty=True)

    street_len_map = _street_length_map(street_df)
    street_endpoints = _street_endpoint_map(street_df)
    plot_junction = _project_memberships(plot_street, street_endpoints)

    x, feat_cols, aux = _build_node_features(
        plot_df=plot_df,
        plot_street=plot_street,
        plot_bid=plot_bid,
        plot_junction=plot_junction,
        street_len_map=street_len_map,
        standardize=not args.no_standardize,
        feature_clip=float(args.feature_clip),
    )
    y, label_mapping = _build_labels(plot_df, args.label_mode)

    centroids_lonlat = _geometry_centroids_from_wkb(plot_df["geometry"])
    centroids_m = _lonlat_to_local_m(centroids_lonlat)

    inv_street = _build_inverse_index(plot_street)
    inv_junction = _build_inverse_index(plot_junction)
    inv_building = _build_inverse_index(plot_bid)

    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else _adaptive_entity_cap(inv_street, min_cap=12, max_cap=64, quantile=float(args.street_cap_quantile))
    junction_cap = int(args.junction_entity_cap) if int(args.junction_entity_cap) > 0 else _adaptive_entity_cap(inv_junction, min_cap=12, max_cap=64, quantile=float(args.junction_cap_quantile))
    building_cap = int(args.building_attr_entity_cap) if int(args.building_attr_entity_cap) > 0 else _adaptive_entity_cap(inv_building, min_cap=8, max_cap=48, quantile=float(args.building_cap_quantile))

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}
    junction_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}

    if spec["use_street"]:
        street_stats = _accumulate_relation_pairs(
            inv_index=inv_street,
            max_nodes_per_entity=street_cap,
            pair_stats=pair_stats,
            relation="street",
            entity_value=street_len_map,
        )

    if spec["use_junction"]:
        junction_stats = _accumulate_relation_pairs(
            inv_index=inv_junction,
            max_nodes_per_entity=junction_cap,
            pair_stats=pair_stats,
            relation="junction",
            entity_value=None,
        )

    geom_stats = {"geom_candidate_nodes": 0, "geom_pairs_inserted": 0, "geom_radius_m": 0.0}
    geom_radius_m = 0.0
    if spec["use_geom_fallback"]:
        structural_deg = _structural_degree(n, pair_stats)
        geom_radius_m = _estimate_geom_radius(centroids_m) * max(float(args.geom_radius_scale), 0.0)
        geom_stats = _add_geom_fallback_pairs(
            centroids_m=centroids_m,
            structural_deg=structural_deg,
            pair_stats=pair_stats,
            geom_k=int(args.geom_k),
            geom_max_struct_deg=int(args.geom_max_struct_deg),
            geom_radius_m=geom_radius_m,
        )

    building_attr_stats = _augment_shared_building_attr(
        inv_building=inv_building,
        pair_stats=pair_stats,
        max_nodes_per_entity=building_cap,
    )

    edge_index, edge_attr, edge_weight, relation_counts = _finalize_edges(
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        aux=aux,
        geom_radius_m=max(float(geom_radius_m), 1.0),
        junction_topk=int(args.junction_topk),
    )

    np.save(base.with_name(f"{dataset_name}_feat.npy"), x.astype(np.float32))
    np.save(base.with_name(f"{dataset_name}_label.npy"), y.astype(np.int64))
    np.save(base.with_name(f"{dataset_name}_edge_index.npy"), edge_index.astype(np.int64))
    np.save(base.with_name(f"{dataset_name}_edge_attr.npy"), edge_attr.astype(np.float32))
    np.save(base.with_name(f"{dataset_name}_edge_weight.npy"), edge_weight.astype(np.float32))

    node_table = pd.DataFrame(
        {
            "node_idx": np.arange(n, dtype=np.int64),
            "plot_id": plot_df["plot_id"].values if "plot_id" in plot_df.columns else np.arange(n, dtype=np.int64),
            "street_membership_count": aux["street_count"].astype(np.int32),
            "junction_membership_count": aux["junction_count"].astype(np.int32),
            "building_membership_count": aux["building_count"].astype(np.int32),
            "label": y.astype(np.int64),
        }
    )
    node_table.to_csv(base.with_name(f"{dataset_name}_node_table.csv"), index=False)

    meta = {
        "city": city,
        "dataset_name": dataset_name,
        "variant": variant,
        "variant_spec": spec,
        "n_nodes": int(n),
        "n_features": int(x.shape[1]),
        "n_edges_directed": int(edge_index.shape[1]),
        "n_edges_undirected": int(edge_index.shape[1] // 2),
        "feature_columns": feat_cols,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "edge_feature_dim": int(edge_attr.shape[1]),
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "schema_mapping": {
            "population": str(aux["population_col"][0]) if len(aux["population_col"]) else "",
            "built": str(aux["built_col"][0]) if len(aux["built_col"]) else "",
            "canopy_mean": str(aux["canopy_mean_col"][0]) if len(aux["canopy_mean_col"]) else "",
            "canopy_std": str(aux["canopy_std_col"][0]) if len(aux["canopy_std_col"]) else "",
        },
        "adaptive_caps": {
            "street_entity_cap": int(street_cap),
            "junction_entity_cap": int(junction_cap),
            "building_attr_entity_cap": int(building_cap),
        },
        "geom_fallback": {
            "geom_k": int(args.geom_k),
            "geom_max_struct_deg": int(args.geom_max_struct_deg),
            "geom_radius_scale": float(args.geom_radius_scale),
            "geom_radius_m": float(geom_stats["geom_radius_m"]),
        },
        "relation_counts_undirected": relation_counts,
        "street_index_stats": street_stats,
        "junction_index_stats": junction_stats,
        "geom_stats": geom_stats,
        "building_attr_stats": building_attr_stats,
        "notes": [
            "Topology is semantics-first: street, junction, and optional geometric fallback only.",
            "Feature similarity does not create or prune edges in V3.",
            "Shared-building overlap is retained as an interpretable edge attribute only.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] prepared {dataset_name}")
    print(f" city={city} variant={variant}")
    print(f" nodes={n} edges_undirected={edge_index.shape[1] // 2}")
    print(f" relation_counts={relation_counts}")


if __name__ == "__main__":
    main()
