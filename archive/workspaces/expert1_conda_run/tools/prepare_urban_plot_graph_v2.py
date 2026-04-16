#!/usr/bin/env python3
"""
Build a more semantically grounded plot-level graph from urban city data.

Design goals:
- avoid feature/label leakage by excluding land-use score columns from node features by default
- make edges represent urban interaction mechanisms, not only raw co-membership
- keep the output format compatible with the existing training pipeline

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
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - optional import for runtime environment only
    NearestNeighbors = None


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

DEMOGRAPHIC_CANDIDATES = [
    "PopSum",
    "Men",
    "Women",
    "Children",
    "Youth",
    "Elderly",
]

EDGE_FEATURE_NAMES = [
    "shared_street_cnt",
    "shared_building_cnt",
    "shared_junction_cnt",
    "shared_street_len_mean",
    "jaccard_street",
    "jaccard_building",
    "jaccard_junction",
    "street_deg_i",
    "street_deg_j",
    "building_deg_i",
    "building_deg_j",
    "junction_deg_i",
    "junction_deg_j",
    "feat_knn_sim",
    "node_feat_cosine",
    "node_feat_l2",
    "node_feat_l1_mean",
]

STAT_STREET_CNT = 0
STAT_STREET_LEN_SUM = 1
STAT_BUILDING_CNT = 2
STAT_JUNCTION_CNT = 3
STAT_FEAT_SIM_SUM = 4
STAT_FEAT_SIM_CNT = 5
PAIR_STAT_DIM = 6


def _safe_id_list(v) -> List[int]:
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
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
        if isinstance(cur, (list, tuple, set, np.ndarray)):
            stack.extend(list(cur))
            continue
        try:
            out.append(int(cur))
        except Exception:
            continue
    out.reverse()
    return out


def _load_membership_array(fp: Path, n: int) -> List[List[int]]:
    arr = np.load(fp, allow_pickle=True)
    if arr.ndim != 1:
        raise ValueError(f"{fp} must be 1D object array, got shape {arr.shape}")
    if len(arr) != n:
        raise ValueError(f"{fp} length mismatch: len={len(arr)} vs expected {n}")
    return [_safe_id_list(x) for x in arr]


def _build_inverse_index(memberships: Sequence[Sequence[int]]) -> Dict[int, List[int]]:
    inv: Dict[int, List[int]] = defaultdict(list)
    for node_idx, ids in enumerate(memberships):
        for x in ids:
            inv[x].append(node_idx)
    return inv


def _street_length_map(street_df: pd.DataFrame) -> Dict[int, float]:
    if not {"street_id", "length"}.issubset(set(street_df.columns)):
        return {}
    out: Dict[int, float] = {}
    g = street_df.groupby("street_id", as_index=False)["length"].mean()
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
    cols = street_df[["street_id", "u", "v"]].dropna()
    for row in cols.itertuples(index=False):
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


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x.copy()
    x[~np.isfinite(x)] = np.nan
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu).astype(np.float32)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(mu, inds[1])
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    z = (x - np.mean(x, axis=0)) / std
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _categorize_numeric_columns(plot_df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
    landuse_cols = [c for c in LAND_USE_CANDIDATES if c in numeric_cols]
    blocked = set(landuse_cols)
    blocked.add("plot_id")
    blocked.update([c for c in numeric_cols if c.lower().endswith("_id")])

    morph_cols = [c for c in numeric_cols if c.startswith("plot_") and c not in blocked]
    env_cols = [
        c
        for c in numeric_cols
        if (
            c.startswith("canopy_")
            or c.startswith("Canopy height")
            or c.startswith("lcz_")
            or c.startswith("2025_")
            or c in DEMOGRAPHIC_CANDIDATES
        )
        and c not in blocked
    ]
    other_cols = [
        c
        for c in numeric_cols
        if c not in blocked and c not in morph_cols and c not in env_cols
    ]
    return {
        "numeric_all": numeric_cols,
        "landuse": landuse_cols,
        "morph": morph_cols,
        "env": env_cols,
        "other": other_cols,
    }


def _unique_keep_order(cols: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _build_node_features(
    plot_df: pd.DataFrame,
    feature_profile: str,
    standardize: bool,
    feature_clip: float,
    include_landuse_features: bool,
    include_sparse_city_specific: bool,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    feature_groups = _categorize_numeric_columns(plot_df)
    landuse_cols = feature_groups["landuse"]

    if feature_profile == "morph_only":
        feat_cols = list(feature_groups["morph"])
    elif feature_profile == "morph_env":
        feat_cols = list(feature_groups["morph"]) + list(feature_groups["env"])
    elif feature_profile == "all_numeric_no_landuse":
        blocked = set(landuse_cols)
        blocked.add("plot_id")
        blocked.update([c for c in feature_groups["numeric_all"] if c.lower().endswith("_id")])
        feat_cols = [c for c in feature_groups["numeric_all"] if c not in blocked]
    elif feature_profile == "all_numeric":
        blocked = {"plot_id"}
        blocked.update([c for c in feature_groups["numeric_all"] if c.lower().endswith("_id")])
        feat_cols = [c for c in feature_groups["numeric_all"] if c not in blocked]
    else:
        raise ValueError(f"Unknown feature_profile: {feature_profile}")

    if include_sparse_city_specific:
        feat_cols.extend(feature_groups["other"])
    if include_landuse_features:
        feat_cols.extend(landuse_cols)

    feat_cols = _unique_keep_order(feat_cols)
    if not feat_cols:
        raise ValueError("No node feature columns selected.")

    x = plot_df[feat_cols].to_numpy(dtype=np.float32)
    if standardize:
        x = _zscore(x)
    if feature_clip > 0:
        x = np.clip(x, -float(feature_clip), float(feature_clip))
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32), feat_cols, feature_groups


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

        value = 0.0
        if entity_value is not None:
            value = float(entity_value.get(entity_id, 0.0))

        for i, j in combinations(uniq, 2):
            key = (i, j)
            rec = pair_stats.setdefault(key, [0.0] * PAIR_STAT_DIM)
            if relation == "street":
                rec[STAT_STREET_CNT] += 1.0
                rec[STAT_STREET_LEN_SUM] += value
            elif relation == "building":
                rec[STAT_BUILDING_CNT] += 1.0
            elif relation == "junction":
                rec[STAT_JUNCTION_CNT] += 1.0
            else:
                raise ValueError(f"Unknown relation: {relation}")
    return stats


def _accumulate_feature_knn_pairs(
    x: np.ndarray,
    pair_stats: Dict[Tuple[int, int], List[float]],
    feature_knn_k: int,
    feature_knn_temp: float,
) -> Dict[str, int]:
    stats = {"knn_pairs_total": 0, "knn_pairs_inserted": 0}
    if feature_knn_k <= 0 or x.shape[0] < 2:
        return stats
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for feature KNN edges.")

    n_neighbors = min(int(feature_knn_k) + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(x)
    dists, nbrs = nn.kneighbors(x, return_distance=True)
    temp = max(float(feature_knn_temp), 1e-6)
    for i in range(x.shape[0]):
        for dist, j in zip(dists[i, 1:], nbrs[i, 1:]):
            stats["knn_pairs_total"] += 1
            j = int(j)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            sim = float(np.exp(-float(dist) / temp))
            rec = pair_stats.setdefault(key, [0.0] * PAIR_STAT_DIM)
            rec[STAT_FEAT_SIM_SUM] += sim
            rec[STAT_FEAT_SIM_CNT] += 1.0
            stats["knn_pairs_inserted"] += 1
    return stats


def _pair_node_relation_features(
    x: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    chunk_size: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = src.shape[0]
    if e == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    chunk_size = max(10000, int(chunk_size))
    out_cos = np.zeros((e,), dtype=np.float32)
    out_l2 = np.zeros((e,), dtype=np.float32)
    out_l1 = np.zeros((e,), dtype=np.float32)

    for st in range(0, e, chunk_size):
        ed = min(e, st + chunk_size)
        xi = x[src[st:ed]]
        xj = x[dst[st:ed]]
        dot = np.sum(xi * xj, axis=1)
        ni = np.linalg.norm(xi, axis=1)
        nj = np.linalg.norm(xj, axis=1)
        out_cos[st:ed] = (dot / np.maximum(ni * nj, 1e-8)).astype(np.float32)
        diff = xi - xj
        out_l2[st:ed] = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
        out_l1[st:ed] = np.mean(np.abs(diff), axis=1).astype(np.float32)

    return out_cos, out_l2, out_l1


def _finalize_edges(
    pair_stats: Dict[Tuple[int, int], List[float]],
    street_deg: np.ndarray,
    building_deg: np.ndarray,
    junction_deg: np.ndarray,
    topk_per_node: int,
    node_x: np.ndarray,
    node_rel_chunk: int,
    feature_edge_blend: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not pair_stats:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            EDGE_FEATURE_NAMES,
        )

    pairs = np.array(list(pair_stats.keys()), dtype=np.int64)
    vals = np.array(list(pair_stats.values()), dtype=np.float32)
    src = pairs[:, 0]
    dst = pairs[:, 1]

    sc = vals[:, STAT_STREET_CNT]
    sl_sum = vals[:, STAT_STREET_LEN_SUM]
    bc = vals[:, STAT_BUILDING_CNT]
    jc = vals[:, STAT_JUNCTION_CNT]
    fs_sum = vals[:, STAT_FEAT_SIM_SUM]
    fs_cnt = vals[:, STAT_FEAT_SIM_CNT]

    feat_knn_sim = np.divide(fs_sum, np.maximum(fs_cnt, 1.0))
    mean_slen = np.divide(sl_sum, np.maximum(sc, 1.0))

    js = np.divide(sc, np.maximum(street_deg[src] + street_deg[dst] - sc, 1.0))
    jb = np.divide(bc, np.maximum(building_deg[src] + building_deg[dst] - bc, 1.0))
    jj = np.divide(jc, np.maximum(junction_deg[src] + junction_deg[dst] - jc, 1.0))

    sc_norm = sc / max(float(np.max(sc)), 1.0)
    bc_norm = bc / max(float(np.max(bc)), 1.0)
    jc_norm = jc / max(float(np.max(jc)), 1.0)
    sl_norm = mean_slen / max(float(np.max(mean_slen)), 1.0)

    rel_cos, rel_l2, rel_l1 = _pair_node_relation_features(
        x=node_x.astype(np.float32, copy=False),
        src=src,
        dst=dst,
        chunk_size=node_rel_chunk,
    )
    rel_cos_01 = np.clip((rel_cos + 1.0) / 2.0, 0.0, 1.0)
    rel_l2_norm = rel_l2 / max(float(np.max(rel_l2)), 1.0)
    rel_l1_norm = rel_l1 / max(float(np.max(rel_l1)), 1.0)
    feature_affinity = np.clip(
        0.65 * feat_knn_sim + 0.25 * rel_cos_01 + 0.10 * (1.0 - 0.5 * (rel_l2_norm + rel_l1_norm)),
        0.0,
        1.0,
    )

    structure_prior = (
        0.26 * js
        + 0.14 * jb
        + 0.18 * jj
        + 0.18 * sc_norm
        + 0.10 * bc_norm
        + 0.08 * jc_norm
        + 0.06 * sl_norm
    ).astype(np.float32)

    blend = min(max(float(feature_edge_blend), 0.0), 1.0)
    weight = ((1.0 - blend) * structure_prior + blend * feature_affinity).astype(np.float32)
    weight = np.clip(weight, 1e-6, None)

    if topk_per_node > 0:
        incident: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for eid, (u, v) in enumerate(pairs):
            w = float(weight[eid])
            incident[int(u)].append((w, eid))
            incident[int(v)].append((w, eid))
        keep = np.zeros(len(pairs), dtype=bool)
        for items in incident.values():
            items.sort(key=lambda x: x[0], reverse=True)
            for _, eid in items[:topk_per_node]:
                keep[eid] = True
        src = src[keep]
        dst = dst[keep]
        sc = sc[keep]
        bc = bc[keep]
        jc = jc[keep]
        mean_slen = mean_slen[keep]
        js = js[keep]
        jb = jb[keep]
        jj = jj[keep]
        feat_knn_sim = feat_knn_sim[keep]
        rel_cos = rel_cos[keep]
        rel_l2_norm = rel_l2_norm[keep]
        rel_l1_norm = rel_l1_norm[keep]
        weight = weight[keep]

    edge_attr = np.stack(
        [
            sc,
            bc,
            jc,
            mean_slen,
            js,
            jb,
            jj,
            street_deg[src],
            street_deg[dst],
            building_deg[src],
            building_deg[dst],
            junction_deg[src],
            junction_deg[dst],
            feat_knn_sim,
            rel_cos,
            rel_l2_norm,
            rel_l1_norm,
        ],
        axis=1,
    ).astype(np.float32)

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    weight = np.concatenate([weight, weight], axis=0).astype(np.float32)
    return edge_index, edge_attr, weight, EDGE_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare urban plot graph with v2 semantics.")
    parser.add_argument("--city", type=str, required=True, help="City folder name, e.g. beijing")
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="", help="Default: urban_<city>_plot_v2")
    parser.add_argument(
        "--feature_profile",
        type=str,
        default="morph_env",
        choices=["morph_only", "morph_env", "all_numeric_no_landuse", "all_numeric"],
    )
    parser.add_argument("--include_landuse_features", action="store_true")
    parser.add_argument("--include_sparse_city_specific", action="store_true")
    parser.add_argument("--max_plots_per_street", type=int, default=60)
    parser.add_argument("--max_plots_per_building", type=int, default=40)
    parser.add_argument("--max_plots_per_junction", type=int, default=48)
    parser.add_argument("--feature_knn_k", type=int, default=8)
    parser.add_argument("--feature_knn_temp", type=float, default=3.0)
    parser.add_argument("--feature_edge_blend", type=float, default=0.18)
    parser.add_argument("--topk_per_node", type=int, default=24)
    parser.add_argument("--node_rel_chunk", type=int, default=200000)
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    city = args.city.strip()
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot_v2"
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

    plot_street = _load_membership_array(city_dir / "plot_street_id.npy", n)
    plot_bid = _load_membership_array(city_dir / "plot_bid.npy", n)

    x, feat_cols, feature_groups = _build_node_features(
        plot_df=plot_df,
        feature_profile=args.feature_profile,
        standardize=not args.no_standardize,
        feature_clip=float(args.feature_clip),
        include_landuse_features=bool(args.include_landuse_features),
        include_sparse_city_specific=bool(args.include_sparse_city_specific),
    )
    y, label_mapping = _build_labels(plot_df, args.label_mode)

    street_endpoints = _street_endpoint_map(street_df)
    plot_junction = _project_memberships(plot_street, street_endpoints)

    inv_street = _build_inverse_index(plot_street)
    inv_building = _build_inverse_index(plot_bid)
    inv_junction = _build_inverse_index(plot_junction)
    street_len = _street_length_map(street_df)

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_stats = _accumulate_relation_pairs(
        inv_index=inv_street,
        max_nodes_per_entity=int(args.max_plots_per_street),
        pair_stats=pair_stats,
        relation="street",
        entity_value=street_len,
    )
    building_stats = _accumulate_relation_pairs(
        inv_index=inv_building,
        max_nodes_per_entity=int(args.max_plots_per_building),
        pair_stats=pair_stats,
        relation="building",
        entity_value=None,
    )
    junction_stats = _accumulate_relation_pairs(
        inv_index=inv_junction,
        max_nodes_per_entity=int(args.max_plots_per_junction),
        pair_stats=pair_stats,
        relation="junction",
        entity_value=None,
    )
    feature_knn_stats = _accumulate_feature_knn_pairs(
        x=x,
        pair_stats=pair_stats,
        feature_knn_k=int(args.feature_knn_k),
        feature_knn_temp=float(args.feature_knn_temp),
    )

    street_deg = np.array([len(set(v)) for v in plot_street], dtype=np.float32)
    building_deg = np.array([len(set(v)) for v in plot_bid], dtype=np.float32)
    junction_deg = np.array([len(set(v)) for v in plot_junction], dtype=np.float32)

    edge_index, edge_attr, edge_weight, edge_feature_names = _finalize_edges(
        pair_stats=pair_stats,
        street_deg=street_deg,
        building_deg=building_deg,
        junction_deg=junction_deg,
        topk_per_node=int(args.topk_per_node),
        node_x=x,
        node_rel_chunk=int(args.node_rel_chunk),
        feature_edge_blend=float(args.feature_edge_blend),
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
            "street_membership_count": street_deg.astype(np.int32),
            "building_membership_count": building_deg.astype(np.int32),
            "junction_membership_count": junction_deg.astype(np.int32),
            "label": y.astype(np.int64),
        }
    )
    node_table.to_csv(base.with_name(f"{dataset_name}_node_table.csv"), index=False)

    meta = {
        "city": city,
        "dataset_name": dataset_name,
        "n_nodes": int(n),
        "n_features": int(x.shape[1]),
        "n_edges_directed": int(edge_index.shape[1]),
        "n_edges_undirected": int(edge_index.shape[1] // 2),
        "feature_profile": args.feature_profile,
        "feature_columns": feat_cols,
        "feature_groups_available": feature_groups,
        "include_landuse_features": bool(args.include_landuse_features),
        "include_sparse_city_specific": bool(args.include_sparse_city_specific),
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "max_plots_per_street": int(args.max_plots_per_street),
        "max_plots_per_building": int(args.max_plots_per_building),
        "max_plots_per_junction": int(args.max_plots_per_junction),
        "feature_knn_k": int(args.feature_knn_k),
        "feature_knn_temp": float(args.feature_knn_temp),
        "feature_edge_blend": float(args.feature_edge_blend),
        "topk_per_node": int(args.topk_per_node),
        "node_rel_chunk": int(args.node_rel_chunk),
        "feature_clip": float(args.feature_clip),
        "edge_feature_names": edge_feature_names,
        "edge_feature_dim": int(edge_attr.shape[1]),
        "street_index_stats": street_stats,
        "building_index_stats": building_stats,
        "junction_index_stats": junction_stats,
        "feature_knn_stats": feature_knn_stats,
        "notes": [
            "Default v2 features exclude land-use score columns to reduce label leakage.",
            "Edges combine shared street, shared building, shared street-junction, and feature-KNN affinity.",
            "Edge weights remain scalar priors so the existing DSE/G20 pipeline can use the dataset directly.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[ok] urban v2 dataset prepared")
    print(f" city={city}")
    print(f" out_dir={out_dir}")
    print(f" feature_profile={args.feature_profile}")
    print(f" include_landuse_features={bool(args.include_landuse_features)}")
    print(f" nodes={n}, features={x.shape[1]}")
    print(f" edges_undirected={edge_index.shape[1] // 2}, edges_directed={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
