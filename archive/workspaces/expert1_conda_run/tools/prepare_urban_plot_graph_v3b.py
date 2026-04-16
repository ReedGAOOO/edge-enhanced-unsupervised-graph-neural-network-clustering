#!/usr/bin/env python3
"""
Build a semantics-first plot graph from raw urban city data with the V3b edge schema.

V3b keeps the V3 topology rules, but tightens edge attributes:
- relation bits are semantically cleaner
- endpoint context is permutation-invariant for undirected edges
- geometry keeps interpretable gap features
- feature relations are descriptive only and never create edges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import prepare_urban_plot_graph_v3 as v3


EDGE_FEATURE_NAMES = [
    "is_street_backed",
    "is_junction_only",
    "is_geom_fallback",
    "is_shared_building",
    "shared_street_cnt",
    "shared_junction_cnt",
    "jaccard_street",
    "jaccard_junction",
    "shared_street_len_mean",
    "street_count_sum",
    "street_count_gap",
    "junction_count_sum",
    "junction_count_gap",
    "building_count_sum",
    "building_count_gap",
    "centroid_dist_m",
    "log_area_gap",
    "orientation_diff_deg",
    "node_feat_cosine",
    "node_feat_l2",
]

V3B_VARIANTS = {
    "v3bs": {
        "description": "V3b street-boundary graph only. Edges connect plots sharing at least one street segment.",
        "use_street": True,
        "use_junction": False,
        "use_geom_fallback": False,
    },
    "v3bsj": {
        "description": "V3b street-boundary graph plus same-junction context edges, with relation-aware pruning.",
        "use_street": True,
        "use_junction": True,
        "use_geom_fallback": False,
    },
    "v3bsjg": {
        "description": "V3b street-boundary graph plus same-junction edges, with geometric fallback edges only for under-connected plots.",
        "use_street": True,
        "use_junction": True,
        "use_geom_fallback": True,
    },
}


def _orientation_array(plot_df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(plot_df.get("plot_orientation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _orientation_diff_deg(orientation_deg: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    diff = np.abs(orientation_deg[src] - orientation_deg[dst]).astype(np.float32)
    diff = np.minimum(diff, 360.0 - diff)
    diff = np.minimum(diff, 180.0)
    return diff.astype(np.float32)


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


def _finalize_edges_v3b(
    pair_stats: Dict[Tuple[int, int], List[float]],
    centroids_m: np.ndarray,
    aux: Dict[str, np.ndarray],
    node_x: np.ndarray,
    plot_df: pd.DataFrame,
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

    rel_street = vals[:, v3.PAIR_REL_STREET]
    rel_junction = vals[:, v3.PAIR_REL_JUNCTION]
    rel_geom = vals[:, v3.PAIR_REL_GEOM]
    street_cnt = vals[:, v3.PAIR_STREET_CNT]
    street_len_mean = np.divide(vals[:, v3.PAIR_STREET_LEN_SUM], np.maximum(street_cnt, 1.0))
    junction_cnt = vals[:, v3.PAIR_JUNCTION_CNT]
    building_cnt = vals[:, v3.PAIR_BUILDING_CNT]
    geom_dist = np.divide(vals[:, v3.PAIR_GEOM_DIST_SUM], np.maximum(vals[:, v3.PAIR_GEOM_DIST_CNT], 1.0))

    street_deg = aux["street_count"].astype(np.float32)
    junction_deg = aux["junction_count"].astype(np.float32)
    building_deg = aux["building_count"].astype(np.float32)

    jaccard_street = np.divide(street_cnt, np.maximum(street_deg[src] + street_deg[dst] - street_cnt, 1.0))
    jaccard_junction = np.divide(junction_cnt, np.maximum(junction_deg[src] + junction_deg[dst] - junction_cnt, 1.0))
    diff_area = np.abs(aux["area_log"][src] - aux["area_log"][dst]).astype(np.float32)

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

    keep = v3._junction_only_keep_mask(
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
    diff_area = diff_area[keep]
    centroid_dist = centroid_dist[keep]

    orientation_deg = _orientation_array(plot_df)
    orientation_diff = _orientation_diff_deg(orientation_deg, src, dst)
    node_feat_cosine, node_feat_l2 = _feature_relation(node_x, src, dst)

    is_street_backed = (rel_street > 0).astype(np.float32)
    is_junction_only = ((rel_junction > 0) & (rel_street <= 0) & (rel_geom <= 0)).astype(np.float32)
    is_geom_fallback = (rel_geom > 0).astype(np.float32)
    is_shared_building = (building_cnt > 0).astype(np.float32)

    street_count_sum = (street_deg[src] + street_deg[dst]).astype(np.float32)
    street_count_gap = np.abs(street_deg[src] - street_deg[dst]).astype(np.float32)
    junction_count_sum = (junction_deg[src] + junction_deg[dst]).astype(np.float32)
    junction_count_gap = np.abs(junction_deg[src] - junction_deg[dst]).astype(np.float32)
    building_count_sum = (building_deg[src] + building_deg[dst]).astype(np.float32)
    building_count_gap = np.abs(building_deg[src] - building_deg[dst]).astype(np.float32)

    edge_attr = np.stack(
        [
            is_street_backed,
            is_junction_only,
            is_geom_fallback,
            is_shared_building,
            street_cnt,
            junction_cnt,
            jaccard_street,
            jaccard_junction,
            street_len_mean,
            street_count_sum,
            street_count_gap,
            junction_count_sum,
            junction_count_gap,
            building_count_sum,
            building_count_gap,
            centroid_dist,
            diff_area,
            orientation_diff,
            node_feat_cosine,
            node_feat_l2,
        ],
        axis=1,
    ).astype(np.float32)

    geom_score = np.exp(-centroid_dist / max(float(geom_radius_m), 1.0)).astype(np.float32)
    sc_norm = street_cnt / max(float(np.max(street_cnt)), 1.0) if street_cnt.size else street_cnt
    jc_norm = junction_cnt / max(float(np.max(junction_cnt)), 1.0) if junction_cnt.size else junction_cnt
    sl_norm = street_len_mean / max(float(np.max(street_len_mean)), 1.0) if street_len_mean.size else street_len_mean

    weight = (
        is_street_backed * (0.72 + 0.18 * sc_norm + 0.10 * sl_norm)
        + is_junction_only * (0.34 + 0.26 * jc_norm)
        + is_geom_fallback * (0.12 + 0.18 * geom_score)
    ).astype(np.float32)
    weight = np.clip(weight, 1e-6, None)

    relation_counts = {
        "street": int(np.sum(is_street_backed > 0)),
        "junction": int(np.sum(is_junction_only > 0)),
        "geom": int(np.sum(is_geom_fallback > 0)),
    }

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0).astype(np.float32)
    weight = np.concatenate([weight, weight], axis=0).astype(np.float32)
    return edge_index, edge_attr, weight, relation_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare semantics-first urban plot graph V3b.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--urban_root", type=str, default="data/urban_network_datasets")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="", help="Default: urban_<city>_plot_<variant>")
    parser.add_argument("--variant", type=str, default="v3bsjg", choices=sorted(V3B_VARIANTS.keys()))
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
    spec = V3B_VARIANTS[variant]
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
    inv_building = v3._build_inverse_index(plot_bid)

    street_cap = int(args.street_entity_cap) if int(args.street_entity_cap) > 0 else v3._adaptive_entity_cap(inv_street, min_cap=12, max_cap=64, quantile=float(args.street_cap_quantile))
    junction_cap = int(args.junction_entity_cap) if int(args.junction_entity_cap) > 0 else v3._adaptive_entity_cap(inv_junction, min_cap=12, max_cap=64, quantile=float(args.junction_cap_quantile))
    building_cap = int(args.building_attr_entity_cap) if int(args.building_attr_entity_cap) > 0 else v3._adaptive_entity_cap(inv_building, min_cap=8, max_cap=48, quantile=float(args.building_cap_quantile))

    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    street_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}
    junction_stats = {"entities_total": 0, "entities_used": 0, "entities_skipped_large": 0}

    if spec["use_street"]:
        street_stats = v3._accumulate_relation_pairs(
            inv_index=inv_street,
            max_nodes_per_entity=street_cap,
            pair_stats=pair_stats,
            relation="street",
            entity_value=street_len_map,
        )

    if spec["use_junction"]:
        junction_stats = v3._accumulate_relation_pairs(
            inv_index=inv_junction,
            max_nodes_per_entity=junction_cap,
            pair_stats=pair_stats,
            relation="junction",
            entity_value=None,
        )

    geom_stats = {"geom_candidate_nodes": 0, "geom_pairs_inserted": 0, "geom_radius_m": 0.0}
    geom_radius_m = 0.0
    if spec["use_geom_fallback"]:
        structural_deg = v3._structural_degree(n, pair_stats)
        geom_radius_m = v3._estimate_geom_radius(centroids_m) * max(float(args.geom_radius_scale), 0.0)
        geom_stats = v3._add_geom_fallback_pairs(
            centroids_m=centroids_m,
            structural_deg=structural_deg,
            pair_stats=pair_stats,
            geom_k=int(args.geom_k),
            geom_max_struct_deg=int(args.geom_max_struct_deg),
            geom_radius_m=geom_radius_m,
        )

    building_attr_stats = v3._augment_shared_building_attr(
        inv_building=inv_building,
        pair_stats=pair_stats,
        max_nodes_per_entity=building_cap,
    )

    edge_index, edge_attr, edge_weight, relation_counts = _finalize_edges_v3b(
        pair_stats=pair_stats,
        centroids_m=centroids_m,
        aux=aux,
        node_x=x,
        plot_df=plot_df,
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
        "edge_schema": "v3b_full_desc_20",
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
        "building_attr_stats": building_attr_stats,
        "geom_stats": geom_stats,
        "notes": [
            "V3b keeps V3 topology but tightens the edge schema.",
            "Junction relation is encoded as junction-only rather than a heavily overlapping multi-hot bit.",
            "Endpoint context is symmetric for undirected edges via sum/gap features.",
            "Feature relations are descriptive only and do not create or prune edges.",
        ],
    }
    base.with_name(f"{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] prepared {dataset_name}")
    print(f" city={city} variant={variant}")
    print(f" nodes={n} edges_undirected={edge_index.shape[1] // 2}")
    print(f" relation_counts={relation_counts}")


if __name__ == "__main__":
    main()
