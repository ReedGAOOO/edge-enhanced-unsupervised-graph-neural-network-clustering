#!/usr/bin/env python3
"""
Build plot-level Urban graphs with more local, semantic edge construction.

V4 is a data-construction search target. It keeps plot nodes, but changes the
projection from the raw Urban heterogeneous graph:

- shared street and shared junction relations are local top-k projections,
  not full cliques over every entity membership;
- building-neighbour edges are induced by nearby building footprints that
  belong to different plots;
- geometric kNN is optional and should be treated as a fallback/control.

The edge features intentionally avoid the land-use score columns used to derive
labels. They use morphology, population/environment proxies, topology counts,
and building-form context.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_urban_plot_graph_v3 as v3  # noqa: E402


EDGE_FEATURE_NAMES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
    "is_building_neighbor",
    "shared_street_cnt",
    "shared_junction_cnt",
    "building_neighbor_cnt",
    "shared_street_len_mean",
    "centroid_dist_m",
    "building_dist_mean",
    "building_dist_min",
    "jaccard_street",
    "jaccard_junction",
    "street_count_sum",
    "street_count_gap",
    "junction_count_sum",
    "junction_count_gap",
    "building_count_sum",
    "building_count_gap",
    "log_area_gap",
    "orientation_diff_deg",
    "population_density_gap",
    "built_density_gap",
    "canopy_gap",
    "building_height_mean_gap",
    "building_area_mean_gap",
    "node_feat_cosine",
    "node_feat_l2",
]


V4_VARIANTS = {
    "v4sl": {
        "description": "Local shared-street projection only.",
        "use_street": True,
        "use_junction": False,
        "use_building": False,
        "use_geom": False,
    },
    "v4slj": {
        "description": "Local shared-street plus local shared-junction projection.",
        "use_street": True,
        "use_junction": True,
        "use_building": False,
        "use_geom": False,
    },
    "v4sljb": {
        "description": "Local street/junction projection plus building-neighbour projected edges.",
        "use_street": True,
        "use_junction": True,
        "use_building": True,
        "use_geom": False,
    },
    "v4sljbg": {
        "description": "Local street/junction/building graph plus geometric fallback for low-degree plots.",
        "use_street": True,
        "use_junction": True,
        "use_building": True,
        "use_geom": True,
    },
    "v4g": {
        "description": "Geometric plot kNN control graph.",
        "use_street": False,
        "use_junction": False,
        "use_building": False,
        "use_geom": True,
    },
}


P_REL_STREET = 0
P_REL_JUNCTION = 1
P_REL_BUILDING = 2
P_REL_GEOM = 3
P_STREET_CNT = 4
P_STREET_LEN_SUM = 5
P_JUNCTION_CNT = 6
P_BUILDING_CNT = 7
P_BUILDING_DIST_SUM = 8
P_BUILDING_DIST_MIN = 9
P_GEOM_DIST_SUM = 10
P_GEOM_DIST_CNT = 11
PAIR_DIM = 12


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _orientation_diff_deg(orientation_deg: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    diff = np.abs(orientation_deg[src] - orientation_deg[dst]).astype(np.float32)
    diff = np.minimum(diff, 360.0 - diff)
    return np.minimum(diff, 180.0).astype(np.float32)


def _feature_relation(node_x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xi = node_x[src].astype(np.float32, copy=False)
    xj = node_x[dst].astype(np.float32, copy=False)
    dot = np.sum(xi * xj, axis=1)
    ni = np.linalg.norm(xi, axis=1)
    nj = np.linalg.norm(xj, axis=1)
    cosine = np.divide(dot, np.clip(ni * nj, 1e-8, None)).astype(np.float32)
    l2 = np.sqrt(np.mean(np.square(xi - xj), axis=1)).astype(np.float32)
    cosine[~np.isfinite(cosine)] = 0.0
    l2[~np.isfinite(l2)] = 0.0
    return cosine, l2


def _building_plot_assignment(plot_bid: Sequence[Sequence[int]], n_buildings: int) -> np.ndarray:
    out = np.full((int(n_buildings),), -1, dtype=np.int64)
    for plot_idx, ids in enumerate(plot_bid):
        for bid in ids:
            b = int(bid)
            if 0 <= b < n_buildings and out[b] < 0:
                out[b] = int(plot_idx)
    return out


def _plot_building_means(
    plot_bid: Sequence[Sequence[int]],
    building_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    n = len(plot_bid)
    area = pd.to_numeric(building_df.get("bid_area", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    height = pd.to_numeric(building_df.get("bid_height", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    area_mean = np.zeros((n,), dtype=np.float32)
    height_mean = np.zeros((n,), dtype=np.float32)
    for idx, ids in enumerate(plot_bid):
        valid = [int(b) for b in ids if 0 <= int(b) < len(building_df)]
        if not valid:
            continue
        area_mean[idx] = _safe_mean(area[valid])
        height_mean[idx] = _safe_mean(height[valid])
    return {
        "building_area_mean": area_mean,
        "building_height_mean": height_mean,
    }


def _estimate_plot_radius(centroids_m: np.ndarray, q: float = 0.95, scale: float = 4.0) -> float:
    if centroids_m.shape[0] < 2:
        return 0.0
    tree = cKDTree(centroids_m)
    dists, _ = tree.query(centroids_m, k=min(2, centroids_m.shape[0]))
    nn = dists[:, 1] if dists.ndim == 2 else dists
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0
    return float(np.quantile(nn, float(q)) * float(scale))


def _add_pair(pair_stats: Dict[Tuple[int, int], List[float]], i: int, j: int) -> List[float]:
    if i == j:
        raise ValueError("Self-pair is not valid.")
    key = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
    return pair_stats.setdefault(key, [0.0] * PAIR_DIM)


def _local_pairs_for_entity(nodes: Sequence[int], centroids_m: np.ndarray, topk: int) -> Iterable[Tuple[int, int, float]]:
    uniq = np.asarray(sorted(set(int(x) for x in nodes)), dtype=np.int64)
    m = int(uniq.shape[0])
    if m < 2:
        return []
    if topk <= 0 or m <= int(topk) + 1:
        out = []
        xy = centroids_m[uniq]
        for a, b in combinations(range(m), 2):
            dist = float(np.linalg.norm(xy[a] - xy[b]))
            out.append((int(uniq[a]), int(uniq[b]), dist))
        return out

    xy = centroids_m[uniq]
    out_set: dict[Tuple[int, int], float] = {}
    k = min(int(topk) + 1, m)
    for pos, node in enumerate(uniq):
        diff = xy - xy[pos]
        dist2 = np.sum(diff * diff, axis=1)
        nbr_pos = np.argpartition(dist2, kth=k - 1)[:k]
        nbr_pos = [int(p) for p in nbr_pos if int(p) != pos]
        nbr_pos.sort(key=lambda p: float(dist2[p]))
        for p in nbr_pos[: int(topk)]:
            i = int(node)
            j = int(uniq[p])
            key = (i, j) if i < j else (j, i)
            out_set[key] = min(float(np.sqrt(dist2[p])), out_set.get(key, float("inf")))
    return [(i, j, d) for (i, j), d in out_set.items()]


def _accumulate_local_entity_pairs(
    inv_index: Dict[int, List[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    relation: str,
    max_nodes_per_entity: int,
    topk: int,
    entity_value: Dict[int, float] | None = None,
) -> Dict[str, int]:
    stats = {
        "entities_total": 0,
        "entities_used": 0,
        "entities_skipped_large": 0,
        "pairs_touched": 0,
        "topk": int(topk),
        "max_nodes_per_entity": int(max_nodes_per_entity),
    }
    for entity_id, nodes in inv_index.items():
        stats["entities_total"] += 1
        uniq = sorted(set(int(x) for x in nodes))
        m = len(uniq)
        if m < 2:
            continue
        if m > int(max_nodes_per_entity):
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        val = float(entity_value.get(int(entity_id), 0.0)) if entity_value is not None else 0.0
        for i, j, _ in _local_pairs_for_entity(uniq, centroids_m=centroids_m, topk=int(topk)):
            rec = _add_pair(pair_stats, i, j)
            if relation == "street":
                rec[P_REL_STREET] = 1.0
                rec[P_STREET_CNT] += 1.0
                rec[P_STREET_LEN_SUM] += val
            elif relation == "junction":
                rec[P_REL_JUNCTION] = 1.0
                rec[P_JUNCTION_CNT] += 1.0
            else:
                raise ValueError(f"Unknown relation: {relation}")
            stats["pairs_touched"] += 1
    return stats


def _accumulate_building_neighbor_pairs(
    building_df: pd.DataFrame,
    plot_bid: Sequence[Sequence[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    building_knn: int,
    building_max_dist_m: float,
) -> Dict[str, int]:
    stats = {
        "buildings_total": int(len(building_df)),
        "buildings_assigned": 0,
        "pairs_touched": 0,
        "building_knn": int(building_knn),
        "building_max_dist_m": float(building_max_dist_m),
    }
    if len(building_df) < 2 or int(building_knn) <= 0:
        return stats

    b_to_plot = _building_plot_assignment(plot_bid=plot_bid, n_buildings=len(building_df))
    valid = np.where(b_to_plot >= 0)[0]
    stats["buildings_assigned"] = int(valid.shape[0])
    if valid.shape[0] < 2:
        return stats

    centroids_lonlat = v3._geometry_centroids_from_wkb(building_df["geometry"])
    centroids_m = v3._lonlat_to_local_m(centroids_lonlat)
    xy = centroids_m[valid]
    tree = cKDTree(xy)
    k = min(int(building_knn) + 1, valid.shape[0])
    dists, nbrs = tree.query(xy, k=k)
    dists = np.atleast_2d(dists)
    nbrs = np.atleast_2d(nbrs)

    for row_idx, src_building in enumerate(valid):
        src_plot = int(b_to_plot[int(src_building)])
        for dist, nbr_local in zip(dists[row_idx, 1:], nbrs[row_idx, 1:]):
            if not np.isfinite(float(dist)):
                continue
            if float(building_max_dist_m) > 0 and float(dist) > float(building_max_dist_m):
                continue
            dst_building = int(valid[int(nbr_local)])
            dst_plot = int(b_to_plot[dst_building])
            if src_plot == dst_plot or dst_plot < 0:
                continue
            rec = _add_pair(pair_stats, src_plot, dst_plot)
            rec[P_REL_BUILDING] = 1.0
            rec[P_BUILDING_CNT] += 1.0
            rec[P_BUILDING_DIST_SUM] += float(dist)
            prev = rec[P_BUILDING_DIST_MIN]
            rec[P_BUILDING_DIST_MIN] = float(dist) if prev <= 0 else min(float(prev), float(dist))
            stats["pairs_touched"] += 1
    return stats


def _add_geom_pairs(
    centroids_m: np.ndarray,
    pair_stats: Dict[Tuple[int, int], List[float]],
    geom_k: int,
    geom_radius_m: float,
    low_degree_only: bool,
    max_existing_degree: int,
) -> Dict[str, int | float]:
    stats: Dict[str, int | float] = {
        "geom_pairs_touched": 0,
        "geom_k": int(geom_k),
        "geom_radius_m": float(geom_radius_m),
        "low_degree_only": bool(low_degree_only),
    }
    n = int(centroids_m.shape[0])
    if n < 2 or int(geom_k) <= 0:
        return stats

    deg = np.zeros((n,), dtype=np.int32)
    for i, j in pair_stats.keys():
        deg[int(i)] += 1
        deg[int(j)] += 1
    nodes = np.where(deg <= int(max_existing_degree))[0] if low_degree_only else np.arange(n, dtype=np.int64)
    stats["geom_candidate_nodes"] = int(nodes.shape[0])
    if nodes.size == 0:
        return stats

    tree = cKDTree(centroids_m)
    k = min(int(geom_k) + 8, n)
    for i in nodes.tolist():
        dists, nbrs = tree.query(centroids_m[int(i)], k=k)
        dists = np.atleast_1d(dists)
        nbrs = np.atleast_1d(nbrs)
        added = 0
        for dist, j in zip(dists[1:], nbrs[1:]):
            j = int(j)
            if int(i) == j:
                continue
            if float(geom_radius_m) > 0 and float(dist) > float(geom_radius_m):
                continue
            rec = _add_pair(pair_stats, int(i), j)
            if rec[P_REL_STREET] > 0 or rec[P_REL_JUNCTION] > 0 or rec[P_REL_BUILDING] > 0:
                continue
            rec[P_REL_GEOM] = 1.0
            rec[P_GEOM_DIST_SUM] += float(dist)
            rec[P_GEOM_DIST_CNT] += 1.0
            stats["geom_pairs_touched"] = int(stats["geom_pairs_touched"]) + 1
            added += 1
            if added >= int(geom_k):
                break
    return stats


def _finalize_edges(
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    aux: Dict[str, np.ndarray],
    building_aux: Dict[str, np.ndarray],
    node_x: np.ndarray,
    plot_df: pd.DataFrame,
    geom_radius_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if not pair_stats:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {"street": 0, "junction": 0, "building": 0, "geom": 0},
        )

    pairs = np.asarray(list(pair_stats.keys()), dtype=np.int64)
    vals = np.asarray(list(pair_stats.values()), dtype=np.float32)
    src = pairs[:, 0]
    dst = pairs[:, 1]

    rel_street = vals[:, P_REL_STREET]
    rel_junction = vals[:, P_REL_JUNCTION]
    rel_building = vals[:, P_REL_BUILDING]
    rel_geom = vals[:, P_REL_GEOM]
    street_cnt = vals[:, P_STREET_CNT]
    junction_cnt = vals[:, P_JUNCTION_CNT]
    building_cnt = vals[:, P_BUILDING_CNT]
    street_len_mean = np.divide(vals[:, P_STREET_LEN_SUM], np.maximum(street_cnt, 1.0))
    building_dist_mean = np.divide(vals[:, P_BUILDING_DIST_SUM], np.maximum(building_cnt, 1.0))
    building_dist_min = vals[:, P_BUILDING_DIST_MIN]
    geom_dist = np.divide(vals[:, P_GEOM_DIST_SUM], np.maximum(vals[:, P_GEOM_DIST_CNT], 1.0))

    dx = centroids_m[src, 0] - centroids_m[dst, 0]
    dy = centroids_m[src, 1] - centroids_m[dst, 1]
    centroid_dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    geom_missing = (geom_dist <= 0) | (~np.isfinite(geom_dist))
    geom_dist[geom_missing] = centroid_dist[geom_missing]

    street_deg = aux["street_count"].astype(np.float32)
    junction_deg = aux["junction_count"].astype(np.float32)
    building_deg = aux["building_count"].astype(np.float32)
    jaccard_street = np.divide(street_cnt, np.maximum(street_deg[src] + street_deg[dst] - street_cnt, 1.0))
    jaccard_junction = np.divide(junction_cnt, np.maximum(junction_deg[src] + junction_deg[dst] - junction_cnt, 1.0))

    orientation = pd.to_numeric(plot_df.get("plot_orientation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    orientation_diff = _orientation_diff_deg(orientation, src, dst)
    node_feat_cosine, node_feat_l2 = _feature_relation(node_x, src, dst)

    is_street = (rel_street > 0).astype(np.float32)
    is_building = (rel_building > 0).astype(np.float32)
    is_geom = ((rel_geom > 0) & (is_street <= 0) & (is_building <= 0)).astype(np.float32)
    is_junction_only = ((rel_junction > 0) & (is_street <= 0) & (is_building <= 0) & (is_geom <= 0)).astype(np.float32)
    is_shared_building = is_building.copy()

    street_count_sum = (street_deg[src] + street_deg[dst]).astype(np.float32)
    street_count_gap = np.abs(street_deg[src] - street_deg[dst]).astype(np.float32)
    junction_count_sum = (junction_deg[src] + junction_deg[dst]).astype(np.float32)
    junction_count_gap = np.abs(junction_deg[src] - junction_deg[dst]).astype(np.float32)
    building_count_sum = (building_deg[src] + building_deg[dst]).astype(np.float32)
    building_count_gap = np.abs(building_deg[src] - building_deg[dst]).astype(np.float32)

    diff_area = np.abs(aux["area_log"][src] - aux["area_log"][dst]).astype(np.float32)
    diff_pop = np.abs(aux["pop_density_log"][src] - aux["pop_density_log"][dst]).astype(np.float32)
    diff_built = np.abs(aux["built_density_log"][src] - aux["built_density_log"][dst]).astype(np.float32)
    diff_canopy = np.abs(aux["canopy_mean"][src] - aux["canopy_mean"][dst]).astype(np.float32)
    diff_b_height = np.abs(building_aux["building_height_mean"][src] - building_aux["building_height_mean"][dst]).astype(np.float32)
    diff_b_area = np.abs(np.log1p(building_aux["building_area_mean"][src]) - np.log1p(building_aux["building_area_mean"][dst])).astype(np.float32)

    edge_attr = np.stack(
        [
            is_street,
            is_junction_only,
            is_geom,
            is_shared_building,
            is_building,
            street_cnt,
            junction_cnt,
            building_cnt,
            street_len_mean,
            centroid_dist,
            building_dist_mean,
            building_dist_min,
            jaccard_street,
            jaccard_junction,
            street_count_sum,
            street_count_gap,
            junction_count_sum,
            junction_count_gap,
            building_count_sum,
            building_count_gap,
            diff_area,
            orientation_diff,
            diff_pop,
            diff_built,
            diff_canopy,
            diff_b_height,
            diff_b_area,
            node_feat_cosine,
            node_feat_l2,
        ],
        axis=1,
    ).astype(np.float32)
    edge_attr[~np.isfinite(edge_attr)] = 0.0

    sc_norm = street_cnt / max(float(np.max(street_cnt)), 1.0) if street_cnt.size else street_cnt
    jc_norm = junction_cnt / max(float(np.max(junction_cnt)), 1.0) if junction_cnt.size else junction_cnt
    bc_norm = building_cnt / max(float(np.max(building_cnt)), 1.0) if building_cnt.size else building_cnt
    geom_score = np.exp(-centroid_dist / max(float(geom_radius_m), 1.0)).astype(np.float32)
    weight = (
        is_street * (0.62 + 0.18 * sc_norm)
        + is_building * (0.50 + 0.22 * bc_norm)
        + is_junction_only * (0.30 + 0.18 * jc_norm)
        + is_geom * (0.12 + 0.18 * geom_score)
    ).astype(np.float32)
    weight = np.clip(weight, 1e-6, None)

    relation_counts = {
        "street": int(np.sum(is_street > 0)),
        "junction": int(np.sum(is_junction_only > 0)),
        "building": int(np.sum(is_building > 0)),
        "geom": int(np.sum(is_geom > 0)),
    }
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    weight = np.concatenate([weight, weight], axis=0).astype(np.float32)
    return edge_index, edge_attr, weight, relation_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Urban plot graph V4 semantic variants.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--variant", type=str, default="v4sljb", choices=sorted(V4_VARIANTS.keys()))
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--street_entity_cap", type=int, default=0)
    parser.add_argument("--junction_entity_cap", type=int, default=0)
    parser.add_argument("--street_cap_quantile", type=float, default=0.995)
    parser.add_argument("--junction_cap_quantile", type=float, default=0.995)
    parser.add_argument("--street_topk", type=int, default=4)
    parser.add_argument("--junction_topk", type=int, default=6)
    parser.add_argument("--building_knn", type=int, default=4)
    parser.add_argument("--building_max_dist_m", type=float, default=80.0)
    parser.add_argument("--geom_k", type=int, default=3)
    parser.add_argument("--geom_radius_m", type=float, default=0.0, help="0 = estimate from plot nearest-neighbour distances.")
    parser.add_argument("--geom_radius_scale", type=float, default=4.0)
    parser.add_argument("--geom_max_existing_degree", type=int, default=1)
    parser.add_argument("--geom_all_nodes", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    city = args.city.strip()
    variant = args.variant.strip().lower()
    spec = V4_VARIANTS[variant]
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_{variant}"

    city_dir = Path(args.urban_root).expanduser().resolve() / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")
    for fn in ["plot.parquet", "street.parquet", "building.parquet", "plot_street_id.npy", "plot_bid.npy"]:
        if not (city_dir / fn).exists():
            raise FileNotFoundError(f"Required file not found: {city_dir / fn}")

    out_dir = Path(args.out_root).expanduser().resolve() / dataset_name
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
        raise FileExistsError(f"Output exists. Use --force. Existing: {exists}")

    plot_df = pd.read_parquet(city_dir / "plot.parquet")
    street_df = pd.read_parquet(city_dir / "street.parquet")
    building_df = pd.read_parquet(city_dir / "building.parquet")
    n = len(plot_df)

    plot_street = v3._load_membership_array(city_dir / "plot_street_id.npy", n, zero_scalar_empty=False)
    plot_bid = v3._load_membership_array(city_dir / "plot_bid.npy", n, zero_scalar_empty=True)
    street_len_map = v3._street_length_map(street_df)
    street_endpoints = v3._street_endpoint_map(street_df)
    plot_junction = v3._project_memberships(plot_street, street_endpoints)

    x, feat_cols, aux = v3._build_node_features(
        plot_df=plot_df,
        plot_street=plot_street,
        plot_bid=plot_bid,
        plot_junction=plot_junction,
        street_len_map=street_len_map,
        standardize=not args.no_standardize,
        feature_clip=float(args.feature_clip),
    )
    y, label_mapping = v3._build_labels(plot_df, args.label_mode)

    centroids_lonlat = v3._geometry_centroids_from_wkb(plot_df["geometry"])
    centroids_m = v3._lonlat_to_local_m(centroids_lonlat)
    inv_street = v3._build_inverse_index(plot_street)
    inv_junction = v3._build_inverse_index(plot_junction)

    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else v3._adaptive_entity_cap(
        inv_street, min_cap=12, max_cap=64, quantile=float(args.street_cap_quantile)
    )
    junction_cap = int(args.junction_entity_cap) if int(args.junction_entity_cap) > 0 else v3._adaptive_entity_cap(
        inv_junction, min_cap=12, max_cap=64, quantile=float(args.junction_cap_quantile)
    )

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0, "pairs_touched": 0}
    junction_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0, "pairs_touched": 0}
    building_stats = {"buildings_total": int(len(building_df)), "buildings_assigned": 0, "pairs_touched": 0}

    if spec["use_street"]:
        street_stats = _accumulate_local_entity_pairs(
            inv_index=inv_street,
            pair_stats=pair_stats,
            centroids_m=centroids_m,
            relation="street",
            max_nodes_per_entity=int(street_cap),
            topk=int(args.street_topk),
            entity_value=street_len_map,
        )
    if spec["use_junction"]:
        junction_stats = _accumulate_local_entity_pairs(
            inv_index=inv_junction,
            pair_stats=pair_stats,
            centroids_m=centroids_m,
            relation="junction",
            max_nodes_per_entity=int(junction_cap),
            topk=int(args.junction_topk),
            entity_value=None,
        )
    if spec["use_building"]:
        building_stats = _accumulate_building_neighbor_pairs(
            building_df=building_df,
            plot_bid=plot_bid,
            pair_stats=pair_stats,
            building_knn=int(args.building_knn),
            building_max_dist_m=float(args.building_max_dist_m),
        )

    geom_radius_m = float(args.geom_radius_m)
    if geom_radius_m <= 0:
        geom_radius_m = _estimate_plot_radius(centroids_m, q=0.95, scale=float(args.geom_radius_scale))
    geom_stats: Dict[str, int | float] = {"geom_pairs_touched": 0, "geom_radius_m": float(geom_radius_m)}
    if spec["use_geom"]:
        geom_stats = _add_geom_pairs(
            centroids_m=centroids_m,
            pair_stats=pair_stats,
            geom_k=int(args.geom_k),
            geom_radius_m=float(geom_radius_m),
            low_degree_only=not bool(args.geom_all_nodes),
            max_existing_degree=int(args.geom_max_existing_degree),
        )

    building_aux = _plot_building_means(plot_bid=plot_bid, building_df=building_df)
    edge_index, edge_attr, edge_weight, relation_counts = _finalize_edges(
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        aux=aux,
        building_aux=building_aux,
        node_x=x,
        plot_df=plot_df,
        geom_radius_m=max(float(geom_radius_m), 1.0),
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
            "building_height_mean": building_aux["building_height_mean"].astype(np.float32),
            "building_area_mean": building_aux["building_area_mean"].astype(np.float32),
            "label": y.astype(np.int64),
        }
    )
    node_table.to_csv(base.with_name(f"{dataset_name}_node_table.csv"), index=False)

    meta = {
        "city": city,
        "dataset_name": dataset_name,
        "variant": variant,
        "variant_spec": spec,
        "edge_schema": "v4_semantic_local_29",
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
        "construction_params": {
            "street_entity_cap": int(street_cap),
            "junction_entity_cap": int(junction_cap),
            "street_topk": int(args.street_topk),
            "junction_topk": int(args.junction_topk),
            "building_knn": int(args.building_knn),
            "building_max_dist_m": float(args.building_max_dist_m),
            "geom_k": int(args.geom_k),
            "geom_radius_m": float(geom_radius_m),
            "geom_all_nodes": bool(args.geom_all_nodes),
            "geom_max_existing_degree": int(args.geom_max_existing_degree),
        },
        "relation_counts_undirected": relation_counts,
        "street_index_stats": street_stats,
        "junction_index_stats": junction_stats,
        "building_neighbor_stats": building_stats,
        "geom_stats": geom_stats,
        "notes": [
            "V4 is intended for data construction search, not as a finalized benchmark.",
            "Street and junction projection is local top-k, avoiding full entity cliques.",
            "Building-neighbour edges are induced by nearby building footprint centroids across plot boundaries.",
            "Land-use score columns are not used as node or edge features because they define the diagnostic labels.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] prepared {dataset_name}")
    print(f" city={city} variant={variant}")
    print(f" nodes={n} edges_undirected={edge_index.shape[1] // 2}")
    print(f" relation_counts={relation_counts}")


if __name__ == "__main__":
    main()
