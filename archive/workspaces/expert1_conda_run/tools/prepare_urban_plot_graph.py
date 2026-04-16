#!/usr/bin/env python3
"""
Build a plot-level graph dataset from urban network city data.

Input folder example (per city):
  urban_network_datasets/<city>/
    plot.parquet
    street.parquet
    plot_street_id.npy
    plot_bid.npy

Output dataset folder:
  <out_root>/<dataset_name>/
    <dataset_name>_feat.npy        [N, F] float32
    <dataset_name>_label.npy       [N] int64 (pseudo labels or zeros)
    <dataset_name>_edge_index.npy  [2, E] int64
    <dataset_name>_edge_attr.npy   [E, D] float32
    <dataset_name>_edge_weight.npy [E] float32
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

EDGE_FEATURE_NAMES = [
    "shared_street_cnt",
    "shared_building_cnt",
    "shared_street_len_mean",
    "jaccard_street",
    "jaccard_building",
    "street_deg_i",
    "street_deg_j",
    "building_deg_i",
    "building_deg_j",
    "node_feat_cosine",
    "node_feat_l2",
    "node_feat_l1_mean",
]


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
    required = {"street_id", "length"}
    if not required.issubset(set(street_df.columns)):
        return {}
    g = street_df.groupby("street_id", as_index=False)["length"].mean()
    out = {}
    for _, row in g.iterrows():
        try:
            out[int(row["street_id"])] = float(row["length"])
        except Exception:
            continue
    return out


def _accumulate_pairs(
    inv_index: Dict[int, List[int]],
    max_nodes_per_entity: int,
    pair_stats: Dict[Tuple[int, int], List[float]],
    mode: str,
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

        e_val = 0.0
        if entity_value is not None:
            e_val = float(entity_value.get(entity_id, 0.0))

        for i, j in combinations(uniq, 2):
            key = (i, j)
            rec = pair_stats.setdefault(key, [0.0, 0.0, 0.0])  # street_cnt, street_len_sum, building_cnt
            if mode == "street":
                rec[0] += 1.0
                rec[1] += e_val
            elif mode == "building":
                rec[2] += 1.0
            else:
                raise ValueError(f"Unknown mode: {mode}")
    return stats


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x.copy()
    x[~np.isfinite(x)] = np.nan
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu).astype(np.float32)
    x = x.copy()
    inds = np.where(np.isnan(x))
    x[inds] = np.take(mu, inds[1])
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    z = (x - np.mean(x, axis=0)) / std
    z[~np.isfinite(z)] = 0.0
    return z


def _build_node_features(plot_df: pd.DataFrame, standardize: bool, feature_clip: float) -> Tuple[np.ndarray, List[str]]:
    numeric_cols = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
    drop_cols = {c for c in numeric_cols if c.lower().endswith("_id")}
    drop_cols.add("plot_id")  # keep explicit for clarity
    feat_cols = [c for c in numeric_cols if c not in drop_cols]
    x = plot_df[feat_cols].to_numpy(dtype=np.float32)
    if standardize:
        x = _zscore(x)
    if feature_clip > 0:
        x = np.clip(x, -float(feature_clip), float(feature_clip))
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32), feat_cols


def _build_labels(plot_df: pd.DataFrame, label_mode: str) -> Tuple[np.ndarray, Dict[str, int] | None]:
    if label_mode == "zeros":
        y = np.zeros(len(plot_df), dtype=np.int64)
        return y, None

    available = [c for c in LAND_USE_CANDIDATES if c in plot_df.columns]
    if not available:
        y = np.zeros(len(plot_df), dtype=np.int64)
        return y, None
    scores = plot_df[available].fillna(0).to_numpy(dtype=np.float32)
    unknown_idx = len(available)
    y = np.argmax(scores, axis=1).astype(np.int64)
    y[np.max(scores, axis=1) <= 0] = unknown_idx
    mapping = {name: idx for idx, name in enumerate(available)}
    mapping["unknown"] = unknown_idx
    return y, mapping


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
        cos = dot / np.maximum(ni * nj, 1e-8)
        diff = xi - xj
        l2 = np.sqrt(np.sum(diff * diff, axis=1))
        l1 = np.mean(np.abs(diff), axis=1)
        out_cos[st:ed] = cos.astype(np.float32)
        out_l2[st:ed] = l2.astype(np.float32)
        out_l1[st:ed] = l1.astype(np.float32)

    return out_cos, out_l2, out_l1


def _finalize_edges(
    pair_stats: Dict[Tuple[int, int], List[float]],
    street_deg: np.ndarray,
    building_deg: np.ndarray,
    topk_per_node: int,
    node_x: np.ndarray | None = None,
    node_rel_chunk: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not pair_stats:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            EDGE_FEATURE_NAMES,
        )

    pairs = np.array(list(pair_stats.keys()), dtype=np.int64)  # [M, 2]
    vals = np.array(list(pair_stats.values()), dtype=np.float32)  # [M, 3]
    sc = vals[:, 0]
    sl_sum = vals[:, 1]
    bc = vals[:, 2]

    i = pairs[:, 0]
    j = pairs[:, 1]

    mean_slen = np.divide(sl_sum, np.maximum(sc, 1.0))
    js = np.divide(sc, np.maximum(street_deg[i] + street_deg[j] - sc, 1.0))
    jb = np.divide(bc, np.maximum(building_deg[i] + building_deg[j] - bc, 1.0))

    sc_norm = sc / max(float(np.max(sc)), 1.0)
    bc_norm = bc / max(float(np.max(bc)), 1.0)
    sl_norm = mean_slen / max(float(np.max(mean_slen)), 1.0)

    # Scalar edge weight used by current model (can be blended in data loader)
    weight = (
        0.45 * js
        + 0.15 * jb
        + 0.25 * sc_norm
        + 0.10 * bc_norm
        + 0.05 * sl_norm
    ).astype(np.float32)
    weight = np.clip(weight, 1e-6, None)

    # Top-k pruning on undirected pairs
    if topk_per_node > 0:
        incident: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for eid, (u, v) in enumerate(pairs):
            w = float(weight[eid])
            incident[int(u)].append((w, eid))
            incident[int(v)].append((w, eid))
        keep = np.zeros(len(pairs), dtype=bool)
        for node, items in incident.items():
            items.sort(key=lambda x: x[0], reverse=True)
            for _, eid in items[:topk_per_node]:
                keep[eid] = True
        pairs = pairs[keep]
        weight = weight[keep]
        sc = sc[keep]
        bc = bc[keep]
        mean_slen = mean_slen[keep]
        js = js[keep]
        jb = jb[keep]

    # Make undirected by duplicating reverse edges
    src = pairs[:, 0]
    dst = pairs[:, 1]
    if node_x is None:
        rel_cos = np.zeros_like(weight, dtype=np.float32)
        rel_l2 = np.zeros_like(weight, dtype=np.float32)
        rel_l1 = np.zeros_like(weight, dtype=np.float32)
    else:
        rel_cos, rel_l2, rel_l1 = _pair_node_relation_features(
            x=node_x.astype(np.float32, copy=False),
            src=src,
            dst=dst,
            chunk_size=node_rel_chunk,
        )

    rel_l2 = rel_l2 / max(float(np.max(rel_l2)), 1.0)
    rel_l1 = rel_l1 / max(float(np.max(rel_l1)), 1.0)

    edge_attr = np.stack(
        [
            sc,
            bc,
            mean_slen,
            js,
            jb,
            street_deg[src],
            street_deg[dst],
            building_deg[src],
            building_deg[dst],
            rel_cos,
            rel_l2,
            rel_l1,
        ],
        axis=1,
    ).astype(np.float32)

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    weight = np.concatenate([weight, weight], axis=0).astype(np.float32)
    return edge_index, edge_attr, weight, EDGE_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare plot-level graph from urban network dataset.")
    parser.add_argument("--city", type=str, required=True, help="City folder name, e.g., beijing")
    parser.add_argument(
        "--urban_root",
        type=str,
        default="data/urban_network_datasets",
        help="Root directory containing city subfolders.",
    )
    parser.add_argument("--out_root", type=str, default="data", help="Output root path.")
    parser.add_argument("--dataset_name", type=str, default="", help="Output dataset name.")
    parser.add_argument("--max_plots_per_street", type=int, default=80)
    parser.add_argument("--max_plots_per_building", type=int, default=80)
    parser.add_argument("--topk_per_node", type=int, default=32)
    parser.add_argument("--node_rel_chunk", type=int, default=200000)
    parser.add_argument("--feature_clip", type=float, default=8.0,
                        help="Clip node features to [-feature_clip, feature_clip] after standardization. <=0 disables clipping.")
    parser.add_argument("--label_mode", type=str, default="landuse", choices=["landuse", "zeros"])
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if exist.")
    args = parser.parse_args()

    city = args.city.strip()
    dataset_name = args.dataset_name.strip() or f"urban_{city}_plot"
    urban_root = Path(args.urban_root).resolve()
    city_dir = urban_root / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")

    required = ["plot.parquet", "street.parquet", "plot_street_id.npy", "plot_bid.npy"]
    for fn in required:
        fp = city_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")

    out_dir = (Path(args.out_root).resolve() / dataset_name)
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

    x, feat_cols = _build_node_features(
        plot_df,
        standardize=not args.no_standardize,
        feature_clip=float(args.feature_clip),
    )
    y, label_mapping = _build_labels(plot_df, args.label_mode)

    inv_street = _build_inverse_index(plot_street)
    inv_building = _build_inverse_index(plot_bid)
    s_len = _street_length_map(street_df)

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_acc_stats = _accumulate_pairs(
        inv_index=inv_street,
        max_nodes_per_entity=args.max_plots_per_street,
        pair_stats=pair_stats,
        mode="street",
        entity_value=s_len,
    )
    building_acc_stats = _accumulate_pairs(
        inv_index=inv_building,
        max_nodes_per_entity=args.max_plots_per_building,
        pair_stats=pair_stats,
        mode="building",
        entity_value=None,
    )

    street_deg = np.array([len(set(v)) for v in plot_street], dtype=np.float32)
    building_deg = np.array([len(set(v)) for v in plot_bid], dtype=np.float32)

    edge_index, edge_attr, edge_weight, edge_feature_names = _finalize_edges(
        pair_stats=pair_stats,
        street_deg=street_deg,
        building_deg=building_deg,
        topk_per_node=args.topk_per_node,
        node_x=x,
        node_rel_chunk=args.node_rel_chunk,
    )

    # Save files
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
        "feature_columns": feat_cols,
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
        "max_plots_per_street": int(args.max_plots_per_street),
        "max_plots_per_building": int(args.max_plots_per_building),
        "topk_per_node": int(args.topk_per_node),
        "node_rel_chunk": int(args.node_rel_chunk),
        "feature_clip": float(args.feature_clip),
        "edge_feature_names": edge_feature_names,
        "edge_feature_dim": int(edge_attr.shape[1]),
        "street_index_stats": street_acc_stats,
        "building_index_stats": building_acc_stats,
        "notes": [
            "Edge construction uses shared street IDs and shared building IDs.",
            "Edge attributes are saved for downstream edge-feature usage.",
            "Edge weights are scalar priors for current pipeline compatibility.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[ok] dataset prepared")
    print(f" city={city}")
    print(f" out_dir={out_dir}")
    print(f" nodes={n}, features={x.shape[1]}")
    print(f" edges_undirected={edge_index.shape[1] // 2}, edges_directed={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
