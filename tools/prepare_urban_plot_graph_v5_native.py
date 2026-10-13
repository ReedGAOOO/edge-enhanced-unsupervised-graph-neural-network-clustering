#!/usr/bin/env python3
"""Build Urban plot graph V5 from native object relations.

V5 is a compatibility view over the Urbanity-style native object graph. It keeps
the analysis node as an urban plot so the existing tile detectors can run, but
edge existence is induced only by native object relations and typed meta-paths:

- plot-street-plot: plots sharing a boundary street segment;
- plot-street-intersection-street-plot: plots linked through incident streets;
- plot-building-building-plot: plots linked by neighbouring building footprints.

Unlike V3/V4, V5 has no geometric fallback edges. The `is_geom_fallback` feature
is still emitted for schema compatibility and is always zero.
"""

from __future__ import annotations

import argparse
import json
import math
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
import prepare_urban_plot_graph_v4_semantic as v4  # noqa: E402
import prepare_urban_plot_hetero_v1 as hetero_v1  # noqa: E402


EDGE_FEATURE_NAMES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
    "is_native_plot_adjacency",
    "is_street_intersection_path",
    "is_building_neighbor",
    "native_relation_count",
    "shared_street_cnt",
    "street_intersection_path_cnt",
    "building_neighbor_cnt",
    "shared_street_len_mean",
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
    "centroid_dist_m",
    "log_area_gap",
    "orientation_diff_deg",
    "population_density_gap",
    "built_density_gap",
    "canopy_gap",
    "building_height_mean_gap",
    "building_area_mean_gap",
    "node_feat_cosine",
    "node_feat_l2",
    "native_edge_confidence",
]

P_STREET = 0
P_JUNCTION_PATH = 1
P_BUILDING_NEIGHBOR = 2
P_STREET_CNT = 3
P_STREET_LEN_SUM = 4
P_JUNCTION_PATH_CNT = 5
P_BUILDING_CNT = 6
P_BUILDING_DIST_SUM = 7
P_BUILDING_DIST_MIN = 8
PAIR_DIM = 9


def _add_pair(pair_stats: Dict[Tuple[int, int], List[float]], i: int, j: int) -> List[float]:
    if int(i) == int(j):
        raise ValueError("Self-pair is not valid.")
    key = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
    return pair_stats.setdefault(key, [0.0] * PAIR_DIM)


def _local_pairs_for_entity(nodes: Sequence[int], centroids_m: np.ndarray, topk: int) -> Iterable[Tuple[int, int]]:
    uniq = np.asarray(sorted(set(int(x) for x in nodes)), dtype=np.int64)
    m = int(uniq.shape[0])
    if m < 2:
        return []
    if int(topk) <= 0 or m <= int(topk) + 1:
        return [(int(a), int(b)) for a, b in combinations(uniq.tolist(), 2)]

    xy = centroids_m[uniq]
    out: set[Tuple[int, int]] = set()
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
            out.add((i, j) if i < j else (j, i))
    return sorted(out)


def _adaptive_cap(inv_index: Dict[int, List[int]], min_cap: int, max_cap: int, quantile: float) -> int:
    return v3._adaptive_entity_cap(inv_index, min_cap=int(min_cap), max_cap=int(max_cap), quantile=float(quantile))


def _accumulate_street_adjacency(
    inv_street: Dict[int, List[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    street_length: Dict[int, float],
    max_plots_per_street: int,
    topk: int,
) -> dict:
    stats = {
        "entities_total": 0,
        "entities_used": 0,
        "entities_skipped_large": 0,
        "pairs_touched": 0,
        "max_plots_per_street": int(max_plots_per_street),
        "topk": int(topk),
    }
    for sid, plots in inv_street.items():
        stats["entities_total"] += 1
        uniq = sorted(set(int(x) for x in plots))
        if len(uniq) < 2:
            continue
        if len(uniq) > int(max_plots_per_street):
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        slen = float(street_length.get(int(sid), 0.0))
        for i, j in _local_pairs_for_entity(uniq, centroids_m=centroids_m, topk=int(topk)):
            rec = _add_pair(pair_stats, i, j)
            rec[P_STREET] = 1.0
            rec[P_STREET_CNT] += 1.0
            rec[P_STREET_LEN_SUM] += slen
            stats["pairs_touched"] += 1
    return stats


def _accumulate_intersection_paths(
    inv_junction: Dict[int, List[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    max_plots_per_junction: int,
    topk: int,
) -> dict:
    stats = {
        "entities_total": 0,
        "entities_used": 0,
        "entities_skipped_large": 0,
        "pairs_touched": 0,
        "max_plots_per_junction": int(max_plots_per_junction),
        "topk": int(topk),
    }
    for junction_id, plots in inv_junction.items():
        stats["entities_total"] += 1
        uniq = sorted(set(int(x) for x in plots))
        if len(uniq) < 2:
            continue
        if len(uniq) > int(max_plots_per_junction):
            stats["entities_skipped_large"] += 1
            continue
        stats["entities_used"] += 1
        for i, j in _local_pairs_for_entity(uniq, centroids_m=centroids_m, topk=int(topk)):
            rec = _add_pair(pair_stats, i, j)
            rec[P_JUNCTION_PATH] = 1.0
            rec[P_JUNCTION_PATH_CNT] += 1.0
            stats["pairs_touched"] += 1
    return stats


def _building_centroids_m(
    building_df: pd.DataFrame,
    plot_centroids_m: np.ndarray,
    building_parent_plots: Dict[int, List[int]],
    max_exact_buildings: int,
) -> tuple[np.ndarray, dict]:
    n = int(len(building_df))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), {"mode": "empty", "proxy_valid_ratio": 0.0}
    if n <= int(max_exact_buildings):
        lonlat = v3._geometry_centroids_from_wkb(building_df["geometry"])
        return v3._lonlat_to_local_m(lonlat), {"mode": "exact_wkb_centroid", "proxy_valid_ratio": 0.0}

    centroids = np.zeros((n, 2), dtype=np.float64)
    valid = np.zeros((n,), dtype=bool)
    for b_idx, parents in building_parent_plots.items():
        p = [int(x) for x in set(parents) if 0 <= int(x) < plot_centroids_m.shape[0]]
        if not p:
            continue
        centroids[int(b_idx)] = np.mean(plot_centroids_m[p], axis=0)
        valid[int(b_idx)] = True
    return centroids, {
        "mode": "parent_plot_centroid_proxy",
        "proxy_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "max_exact_buildings": int(max_exact_buildings),
    }


def _accumulate_building_neighbor_paths(
    building_df: pd.DataFrame,
    plot_building_idx: Sequence[Sequence[int]],
    pair_stats: Dict[Tuple[int, int], List[float]],
    plot_centroids_m: np.ndarray,
    building_knn: int,
    building_max_dist_m: float,
    max_exact_buildings: int,
) -> dict:
    stats = {
        "buildings_total": int(len(building_df)),
        "buildings_assigned": 0,
        "pairs_touched": 0,
        "building_knn": int(building_knn),
        "building_max_dist_m": float(building_max_dist_m),
    }
    if len(building_df) < 2 or int(building_knn) <= 0:
        return stats

    building_parent_plots = v3._build_inverse_index(plot_building_idx)
    b_to_plots = {int(b): sorted(set(int(p) for p in parents)) for b, parents in building_parent_plots.items()}
    valid = np.asarray(sorted(b_to_plots.keys()), dtype=np.int64)
    stats["buildings_assigned"] = int(valid.shape[0])
    if valid.shape[0] < 2:
        return stats

    centroids_m, centroid_stats = _building_centroids_m(
        building_df=building_df,
        plot_centroids_m=plot_centroids_m,
        building_parent_plots=building_parent_plots,
        max_exact_buildings=int(max_exact_buildings),
    )
    stats.update({f"centroid_{k}": v for k, v in centroid_stats.items()})

    xy = centroids_m[valid]
    finite = np.isfinite(xy).all(axis=1)
    valid = valid[finite]
    xy = xy[finite]
    if valid.shape[0] < 2:
        return stats

    tree = cKDTree(xy)
    k = min(int(building_knn) + 1, valid.shape[0])
    dists, nbrs = tree.query(xy, k=k)
    dists = np.atleast_2d(dists)
    nbrs = np.atleast_2d(nbrs)
    for row_idx, src_building in enumerate(valid):
        src_plots = b_to_plots.get(int(src_building), [])
        for dist, nbr_local in zip(dists[row_idx, 1:], nbrs[row_idx, 1:]):
            if not np.isfinite(float(dist)):
                continue
            if float(building_max_dist_m) > 0 and float(dist) > float(building_max_dist_m):
                continue
            dst_building = int(valid[int(nbr_local)])
            dst_plots = b_to_plots.get(dst_building, [])
            for src_plot in src_plots:
                for dst_plot in dst_plots:
                    if int(src_plot) == int(dst_plot):
                        continue
                    rec = _add_pair(pair_stats, int(src_plot), int(dst_plot))
                    rec[P_BUILDING_NEIGHBOR] = 1.0
                    rec[P_BUILDING_CNT] += 1.0
                    rec[P_BUILDING_DIST_SUM] += float(dist)
                    prev = rec[P_BUILDING_DIST_MIN]
                    rec[P_BUILDING_DIST_MIN] = float(dist) if prev <= 0 else min(float(prev), float(dist))
                    stats["pairs_touched"] += 1
    return stats


def _plot_building_means(plot_building_idx: Sequence[Sequence[int]], building_df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(plot_building_idx)
    area = pd.to_numeric(building_df.get("bid_area", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    height = pd.to_numeric(building_df.get("bid_height", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    area_mean = np.zeros((n,), dtype=np.float32)
    height_mean = np.zeros((n,), dtype=np.float32)
    for idx, ids in enumerate(plot_building_idx):
        valid = [int(b) for b in ids if 0 <= int(b) < len(building_df)]
        if not valid:
            continue
        area_mean[idx] = float(np.mean(area[valid]))
        height_mean[idx] = float(np.mean(height[valid]))
    return {"building_area_mean": area_mean, "building_height_mean": height_mean}


def _finalize_edges(
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    aux: dict[str, np.ndarray],
    building_aux: dict[str, np.ndarray],
    node_x: np.ndarray,
    plot_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
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

    street_flag = vals[:, P_STREET]
    junction_flag = vals[:, P_JUNCTION_PATH]
    building_flag = vals[:, P_BUILDING_NEIGHBOR]
    street_cnt = vals[:, P_STREET_CNT]
    junction_cnt = vals[:, P_JUNCTION_PATH_CNT]
    building_cnt = vals[:, P_BUILDING_CNT]
    street_len_mean = np.divide(vals[:, P_STREET_LEN_SUM], np.maximum(street_cnt, 1.0))
    building_dist_mean = np.divide(vals[:, P_BUILDING_DIST_SUM], np.maximum(building_cnt, 1.0))
    building_dist_min = vals[:, P_BUILDING_DIST_MIN]

    dx = centroids_m[src, 0] - centroids_m[dst, 0]
    dy = centroids_m[src, 1] - centroids_m[dst, 1]
    centroid_dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    street_deg = aux["street_count"].astype(np.float32)
    junction_deg = aux["junction_count"].astype(np.float32)
    building_deg = aux["building_count"].astype(np.float32)
    jaccard_street = np.divide(street_cnt, np.maximum(street_deg[src] + street_deg[dst] - street_cnt, 1.0))
    jaccard_junction = np.divide(junction_cnt, np.maximum(junction_deg[src] + junction_deg[dst] - junction_cnt, 1.0))

    is_street = (street_flag > 0).astype(np.float32)
    is_building = (building_flag > 0).astype(np.float32)
    is_geom = np.zeros_like(is_street, dtype=np.float32)
    is_junction_only = ((junction_flag > 0) & (is_street <= 0) & (is_building <= 0)).astype(np.float32)
    native_relation_count = (is_street + (junction_flag > 0).astype(np.float32) + is_building).astype(np.float32)

    orientation = pd.to_numeric(plot_df.get("plot_orientation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    orientation_diff = v4._orientation_diff_deg(orientation, src, dst)
    node_feat_cosine, node_feat_l2 = v4._feature_relation(node_x, src, dst)

    diff_area = np.abs(aux["area_log"][src] - aux["area_log"][dst]).astype(np.float32)
    diff_pop = np.abs(aux["pop_density_log"][src] - aux["pop_density_log"][dst]).astype(np.float32)
    diff_built = np.abs(aux["built_density_log"][src] - aux["built_density_log"][dst]).astype(np.float32)
    diff_canopy = np.abs(aux["canopy_mean"][src] - aux["canopy_mean"][dst]).astype(np.float32)
    diff_b_height = np.abs(building_aux["building_height_mean"][src] - building_aux["building_height_mean"][dst]).astype(np.float32)
    diff_b_area = np.abs(np.log1p(building_aux["building_area_mean"][src]) - np.log1p(building_aux["building_area_mean"][dst])).astype(np.float32)

    sc_norm = street_cnt / max(float(np.max(street_cnt)), 1.0) if street_cnt.size else street_cnt
    jc_norm = junction_cnt / max(float(np.max(junction_cnt)), 1.0) if junction_cnt.size else junction_cnt
    bc_norm = building_cnt / max(float(np.max(building_cnt)), 1.0) if building_cnt.size else building_cnt
    confidence = (
        is_street * (0.68 + 0.16 * sc_norm)
        + is_building * (0.58 + 0.18 * bc_norm)
        + is_junction_only * (0.34 + 0.14 * jc_norm)
    ).astype(np.float32)
    confidence = np.clip(confidence, 1e-6, 1.0)

    edge_attr = np.stack(
        [
            is_street,
            is_junction_only,
            is_geom,
            is_building,
            is_street,
            (junction_flag > 0).astype(np.float32),
            is_building,
            native_relation_count,
            street_cnt,
            junction_cnt,
            building_cnt,
            street_len_mean,
            building_dist_mean,
            building_dist_min,
            jaccard_street,
            jaccard_junction,
            (street_deg[src] + street_deg[dst]).astype(np.float32),
            np.abs(street_deg[src] - street_deg[dst]).astype(np.float32),
            (junction_deg[src] + junction_deg[dst]).astype(np.float32),
            np.abs(junction_deg[src] - junction_deg[dst]).astype(np.float32),
            (building_deg[src] + building_deg[dst]).astype(np.float32),
            np.abs(building_deg[src] - building_deg[dst]).astype(np.float32),
            centroid_dist,
            diff_area,
            orientation_diff,
            diff_pop,
            diff_built,
            diff_canopy,
            diff_b_height,
            diff_b_area,
            node_feat_cosine,
            node_feat_l2,
            confidence,
        ],
        axis=1,
    ).astype(np.float32)
    edge_attr[~np.isfinite(edge_attr)] = 0.0

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    edge_weight = np.concatenate([confidence, confidence], axis=0).astype(np.float32)
    relation_counts = {
        "street": int(np.sum(is_street > 0)),
        "junction": int(np.sum(is_junction_only > 0)),
        "building": int(np.sum(is_building > 0)),
        "geom": 0,
    }
    return edge_index, edge_attr, edge_weight, relation_counts


def _required(city_dir: Path) -> None:
    for fn in ["plot.parquet", "street.parquet", "building.parquet", "plot_street_id.npy", "plot_bid.npy"]:
        if not (city_dir / fn).exists():
            raise FileNotFoundError(f"Required file not found: {city_dir / fn}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Urban plot graph V5 from native object relations.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--street_entity_cap", type=int, default=0)
    parser.add_argument("--junction_entity_cap", type=int, default=0)
    parser.add_argument("--street_cap_quantile", type=float, default=0.995)
    parser.add_argument("--junction_cap_quantile", type=float, default=0.995)
    parser.add_argument("--street_topk", type=int, default=4)
    parser.add_argument("--junction_topk", type=int, default=6)
    parser.add_argument("--building_knn", type=int, default=3)
    parser.add_argument("--building_max_dist_m", type=float, default=80.0)
    parser.add_argument("--max_exact_buildings", type=int, default=250000)
    parser.add_argument("--disable_building_paths", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    city = args.city.strip().lower()
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_v5"
    city_dir = Path(args.urban_root).expanduser().resolve() / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")
    _required(city_dir)

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
    plot_bid_raw = v3._load_membership_array(city_dir / "plot_bid.npy", n, zero_scalar_empty=True)
    building_ids = pd.to_numeric(building_df.get("bid", pd.Series(np.arange(len(building_df)))), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    bid_to_idx = {int(bid): idx for idx, bid in enumerate(building_ids.tolist())}
    plot_bid, plot_bid_ref_stats = hetero_v1._resolve_membership_to_indices(
        memberships=plot_bid_raw,
        node_ids=building_ids,
        id_to_idx=bid_to_idx,
        relation_name="plot_bid",
    )

    street_len_map = v3._street_length_map(street_df)
    street_endpoint = v3._street_endpoint_map(street_df)
    plot_junction = v3._project_memberships(plot_street, street_endpoint)
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
    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else _adaptive_cap(
        inv_street, min_cap=8, max_cap=48, quantile=float(args.street_cap_quantile)
    )
    junction_cap = int(args.junction_entity_cap) if int(args.junction_entity_cap) > 0 else _adaptive_cap(
        inv_junction, min_cap=8, max_cap=64, quantile=float(args.junction_cap_quantile)
    )

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_stats = _accumulate_street_adjacency(
        inv_street=inv_street,
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        street_length=street_len_map,
        max_plots_per_street=int(street_cap),
        topk=int(args.street_topk),
    )
    junction_stats = _accumulate_intersection_paths(
        inv_junction=inv_junction,
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        max_plots_per_junction=int(junction_cap),
        topk=int(args.junction_topk),
    )
    building_stats = {"buildings_total": int(len(building_df)), "buildings_assigned": 0, "pairs_touched": 0, "disabled": True}
    if not bool(args.disable_building_paths):
        building_stats = _accumulate_building_neighbor_paths(
            building_df=building_df,
            plot_building_idx=plot_bid,
            pair_stats=pair_stats,
            plot_centroids_m=centroids_m,
            building_knn=int(args.building_knn),
            building_max_dist_m=float(args.building_max_dist_m),
            max_exact_buildings=int(args.max_exact_buildings),
        )
        building_stats["disabled"] = False

    building_aux = _plot_building_means(plot_bid, building_df)
    edge_index, edge_attr, edge_weight, relation_counts = _finalize_edges(
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        aux=aux,
        building_aux=building_aux,
        node_x=x,
        plot_df=plot_df,
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
        "variant": "v5",
        "edge_schema": "v5_native_object_relation_33",
        "n_nodes": int(n),
        "n_features": int(x.shape[1]),
        "n_edges_directed": int(edge_index.shape[1]),
        "n_edges_undirected": int(edge_index.shape[1] // 2),
        "feature_columns": feat_cols,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "edge_feature_dim": int(edge_attr.shape[1]),
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "relation_counts_undirected": relation_counts,
        "construction_params": {
            "street_entity_cap": int(street_cap),
            "junction_entity_cap": int(junction_cap),
            "street_topk": int(args.street_topk),
            "junction_topk": int(args.junction_topk),
            "building_knn": int(args.building_knn),
            "building_max_dist_m": float(args.building_max_dist_m),
            "max_exact_buildings": int(args.max_exact_buildings),
            "disable_building_paths": bool(args.disable_building_paths),
        },
        "plot_bid_reference_stats": plot_bid_ref_stats,
        "street_native_relation_stats": street_stats,
        "junction_native_metapath_stats": junction_stats,
        "building_native_metapath_stats": building_stats,
        "notes": [
            "V5 is a plot-level compatibility view over native Urbanity object relations.",
            "Edges are induced only by plot-street-plot, plot-street-intersection-street-plot, or plot-building-building-plot paths.",
            "No geometry fallback is used; is_geom_fallback is always zero.",
            "Street and junction entity projections are local top-k to avoid turning high-degree urban objects into dense cliques.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] prepared {dataset_name}")
    print(f" city={city} variant=v5")
    print(f" nodes={n} edges_undirected={edge_index.shape[1] // 2}")
    print(f" relation_counts={relation_counts}")


if __name__ == "__main__":
    main()
