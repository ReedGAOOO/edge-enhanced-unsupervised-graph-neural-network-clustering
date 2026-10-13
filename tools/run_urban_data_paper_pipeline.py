#!/usr/bin/env python3
"""Urban data-paper pipeline for tile-level graph characteristic discovery.

This is a data/mechanism pipeline, not the main GNN clustering model. It builds
interpretable tile-level detectors from Urban plot graphs:

- relation-mix outlier
- spatial/degree corrected residual-edge outlier
- geometry-confound detector
- graph-form mismatch detector

Land-use labels are used only for weak-label validation and interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_urban_plot_graph_v3 as v3  # noqa: E402
from urban_factor_experiment_utils import (  # noqa: E402
    RELATION_FEATURES,
    city_label,
    city_to_dataset,
    coarse_labels,
    ensure_dir,
    feature_index,
    known_mask,
    load_sparse_graph,
    parse_cities,
    undirected_unique,
    write_readme,
)


DEFAULT_BASE_GRAPH_ROOT = "/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data"
DEFAULT_URBAN_ROOT = "/mnt/e/urban network datasets"

GRAPH_FEATURES_FOR_MISMATCH = [
    "n_nodes",
    "n_edges_within",
    "mean_degree",
    "degree_var",
    "induced_edge_density",
    "street_edge_share",
    "junction_edge_share",
    "building_edge_share",
    "geom_edge_share",
    "boundary_crossing_edge_ratio",
]

FORM_FEATURES_FOR_MISMATCH = [
    "plot_area_mean",
    "plot_area_std",
    "building_count_mean",
    "street_count_mean",
    "building_area_mean",
    "density_proxy_nodes_per_km2",
]


def _safe_numeric(series: pd.Series | Any, n: int, default: float = np.nan) -> np.ndarray:
    if isinstance(series, pd.Series):
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    else:
        arr = np.full((n,), default, dtype=np.float64)
    if arr.shape[0] != n:
        out = np.full((n,), default, dtype=np.float64)
        out[: min(n, arr.shape[0])] = arr[: min(n, arr.shape[0])]
        arr = out
    return arr


def _node_feature(graph, name: str, default: float = np.nan) -> np.ndarray:
    feature_names = list(graph.meta.get("feature_columns", []))
    if name in feature_names:
        return graph.x[:, feature_names.index(name)].astype(np.float64, copy=False)
    return np.full((graph.x.shape[0],), default, dtype=np.float64)


def _edge_feature(edge_attr: np.ndarray, feature_names: list[str], name: str, default: float = np.nan) -> np.ndarray:
    idx = feature_index(feature_names, name)
    if idx is None:
        return np.full((edge_attr.shape[0],), default, dtype=np.float64)
    return edge_attr[:, idx].astype(np.float64, copy=False)


def _read_node_table(base_root: Path, dataset: str, n: int) -> pd.DataFrame:
    path = base_root / dataset / f"{dataset}_node_table.csv"
    if not path.exists():
        return pd.DataFrame({"node_idx": np.arange(n)})
    df = pd.read_csv(path)
    if len(df) != n:
        df = df.reindex(range(n)).reset_index(drop=True)
        df["node_idx"] = np.arange(n)
    return df


def _read_plot_df(urban_root: Path, city: str, n: int) -> pd.DataFrame:
    path = urban_root / city / "plot.parquet"
    if not path.exists():
        return pd.DataFrame(index=range(n))
    df = pd.read_parquet(path)
    if len(df) != n:
        df = df.reindex(range(n)).reset_index(drop=True)
    return df


def _building_area_mean(urban_root: Path, city: str, n: int) -> np.ndarray:
    city_dir = urban_root / city
    building_path = city_dir / "building.parquet"
    membership_path = city_dir / "plot_bid.npy"
    out = np.full((n,), np.nan, dtype=np.float64)
    if not building_path.exists() or not membership_path.exists():
        return out
    try:
        building_df = pd.read_parquet(building_path, columns=["bid_area"])
    except Exception:
        building_df = pd.read_parquet(building_path)
    area = pd.to_numeric(building_df.get("bid_area", 0.0), errors="coerce").to_numpy(dtype=np.float64)
    memberships = v3._load_membership_array(membership_path, n, zero_scalar_empty=True)
    for idx, bids in enumerate(memberships):
        valid = [int(b) for b in bids if 0 <= int(b) < area.shape[0]]
        if valid:
            vals = area[valid]
            vals = vals[np.isfinite(vals)]
            out[idx] = float(vals.mean()) if vals.size else np.nan
    return out


def _centroids(plot_df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    if "geometry" not in plot_df.columns:
        z = np.full((n, 2), np.nan, dtype=np.float64)
        return z, z
    try:
        lonlat = v3._geometry_centroids_from_wkb(plot_df["geometry"])
        if lonlat.shape[0] != n:
            out = np.full((n, 2), np.nan, dtype=np.float64)
            out[: min(n, lonlat.shape[0])] = lonlat[: min(n, lonlat.shape[0])]
            lonlat = out
        local_m = v3._lonlat_to_local_m(lonlat)
        return lonlat, local_m
    except Exception:
        z = np.full((n, 2), np.nan, dtype=np.float64)
        return z, z


def _degree(num_nodes: int, edge_index_u: np.ndarray) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.float64)
    if edge_index_u.shape[1] == 0:
        return deg
    np.add.at(deg, edge_index_u[0].astype(np.int64), 1.0)
    np.add.at(deg, edge_index_u[1].astype(np.int64), 1.0)
    return deg


def _entropy_norm(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    counts = np.asarray(list(Counter(vals.astype(int)).values()), dtype=np.float64)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def _robust_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-9:
        sd = x.std(skipna=True)
        if not np.isfinite(sd) or sd < 1e-9:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return ((x - x.mean(skipna=True)) / sd).clip(-10.0, 10.0)
    return (0.6745 * (x - med) / mad).clip(-10.0, 10.0)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def _standardize_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64).copy()
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(arr))
    if inds[0].size:
        arr[inds] = np.take(med, inds[1])
    sd = np.nanstd(arr, axis=0)
    sd[sd < 1e-9] = 1.0
    return ((arr - np.nanmean(arr, axis=0)) / sd).astype(np.float64)


def _union_find_components(nodes: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray) -> int:
    if nodes.size == 0:
        return 0
    idx = {int(node): pos for pos, node in enumerate(nodes)}
    parent = np.arange(nodes.size, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for a, b in zip(edge_src, edge_dst):
        ia = idx.get(int(a))
        ib = idx.get(int(b))
        if ia is None or ib is None:
            continue
        ra = find(ia)
        rb = find(ib)
        if ra != rb:
            parent[rb] = ra
    roots = {find(i) for i in range(nodes.size)}
    return int(len(roots))


def _node_context(base_root: Path, urban_root: Path, city: str, graph) -> pd.DataFrame:
    n = graph.x.shape[0]
    dataset = graph.dataset
    node_table = _read_node_table(base_root, dataset, n)
    plot_df = _read_plot_df(urban_root, city, n)
    lonlat, local_m = _centroids(plot_df, n)
    plot_area = _safe_numeric(plot_df.get("plot_area", None), n)
    if not np.isfinite(plot_area).any():
        plot_area = np.expm1(_node_feature(graph, "log_plot_area", default=np.nan))
    building_count = _safe_numeric(
        node_table.get("building_membership_count", None),
        n,
        default=np.nan,
    )
    if not np.isfinite(building_count).any():
        building_count = np.expm1(_node_feature(graph, "log_building_count", default=np.nan))
    street_count = _safe_numeric(node_table.get("street_membership_count", None), n, default=np.nan)
    if not np.isfinite(street_count).any():
        street_count = np.expm1(_node_feature(graph, "log_street_count", default=np.nan))
    building_area = _building_area_mean(urban_root, city, n)
    coarse, coarse_mapping = coarse_labels(graph.labels.astype(np.int64, copy=False), graph.meta)
    known = known_mask(graph.labels, graph.meta)
    return pd.DataFrame(
        {
            "node_idx": np.arange(n, dtype=np.int64),
            "city": city,
            "dataset": dataset,
            "lon": lonlat[:, 0],
            "lat": lonlat[:, 1],
            "x_m": local_m[:, 0],
            "y_m": local_m[:, 1],
            "plot_area": plot_area,
            "building_count": building_count,
            "street_count": street_count,
            "building_area_mean": building_area,
            "label": graph.labels.astype(np.int64, copy=False),
            "known_label": known,
            "coarse_label": coarse,
        }
    )


def _assign_tiles(node_df: pd.DataFrame, tile_size_m: float) -> pd.DataFrame:
    out = node_df.copy()
    valid = np.isfinite(out["x_m"].to_numpy(dtype=float)) & np.isfinite(out["y_m"].to_numpy(dtype=float))
    out["tile_x"] = -1
    out["tile_y"] = -1
    if valid.any():
        x = out.loc[valid, "x_m"].to_numpy(dtype=np.float64)
        y = out.loc[valid, "y_m"].to_numpy(dtype=np.float64)
        out.loc[valid, "tile_x"] = np.floor((x - np.nanmin(x)) / float(tile_size_m)).astype(int)
        out.loc[valid, "tile_y"] = np.floor((y - np.nanmin(y)) / float(tile_size_m)).astype(int)
    out["tile_id"] = out["tile_x"].astype(str) + "_" + out["tile_y"].astype(str)
    out.loc[out["tile_x"] < 0, "tile_id"] = "invalid"
    out["city_tile_id"] = out["city"] + ":" + out["tile_id"]
    return out


def _edge_context(graph, node_df: pd.DataFrame) -> pd.DataFrame:
    edge_index_u, edge_attr_u, edge_weight_u = undirected_unique(
        graph.edge_index,
        graph.edge_attr,
        graph.edge_weight,
        num_nodes=graph.x.shape[0],
    )
    src = edge_index_u[0].astype(np.int64, copy=False)
    dst = edge_index_u[1].astype(np.int64, copy=False)
    deg = _degree(graph.x.shape[0], edge_index_u)
    known = known_mask(graph.labels, graph.meta)
    same_known = known[src] & known[dst]
    y_same = np.where(same_known, graph.labels[src] == graph.labels[dst], np.nan)
    out = pd.DataFrame(
        {
            "edge_id": np.arange(src.size, dtype=np.int64),
            "src": src,
            "dst": dst,
            "src_tile_id": node_df.loc[src, "tile_id"].to_numpy(),
            "dst_tile_id": node_df.loc[dst, "tile_id"].to_numpy(),
            "src_city_tile_id": node_df.loc[src, "city_tile_id"].to_numpy(),
            "dst_city_tile_id": node_df.loc[dst, "city_tile_id"].to_numpy(),
            "within_tile": node_df.loc[src, "tile_id"].to_numpy() == node_df.loc[dst, "tile_id"].to_numpy(),
            "src_degree": deg[src],
            "dst_degree": deg[dst],
            "degree_mean": 0.5 * (deg[src] + deg[dst]),
            "degree_gap": np.abs(deg[src] - deg[dst]),
            "known_edge": same_known,
            "same_label": y_same,
            "plot_area_gap": np.abs(node_df.loc[src, "plot_area"].to_numpy(dtype=np.float64) - node_df.loc[dst, "plot_area"].to_numpy(dtype=np.float64)),
            "building_count_gap": np.abs(node_df.loc[src, "building_count"].to_numpy(dtype=np.float64) - node_df.loc[dst, "building_count"].to_numpy(dtype=np.float64)),
            "street_count_gap": np.abs(node_df.loc[src, "street_count"].to_numpy(dtype=np.float64) - node_df.loc[dst, "street_count"].to_numpy(dtype=np.float64)),
        }
    )
    for rel in RELATION_FEATURES:
        out[rel] = _edge_feature(edge_attr_u, graph.feature_names, rel, default=0.0)
    for name in ["centroid_dist_m", "node_feat_l2", "log_area_gap", "orientation_diff_deg"]:
        out[name] = _edge_feature(edge_attr_u, graph.feature_names, name, default=np.nan)
    out["semantic_relation_share"] = out[[x for x in ["is_street_backed", "is_junction_only", "is_shared_building"] if x in out]].sum(axis=1).clip(upper=1.0)
    out["edge_weight"] = edge_weight_u if edge_weight_u is not None else np.nan
    return out


def _sample_negative_pairs(
    n_nodes: int,
    existing_keys: set[int],
    n_samples: int,
    seed: int,
    max_attempt_factor: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 17)
    src_list: list[int] = []
    dst_list: list[int] = []
    seen: set[int] = set()
    attempts = 0
    max_attempts = max(int(n_samples) * int(max_attempt_factor), 1000)
    while len(src_list) < int(n_samples) and attempts < max_attempts:
        batch = min(max(int(n_samples) - len(src_list), 1024), 20000)
        a = rng.integers(0, n_nodes, size=batch, endpoint=False)
        b = rng.integers(0, n_nodes, size=batch, endpoint=False)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        good = lo != hi
        for aa, bb in zip(lo[good], hi[good]):
            key = int(aa) * int(n_nodes) + int(bb)
            if key in existing_keys or key in seen:
                continue
            seen.add(key)
            src_list.append(int(aa))
            dst_list.append(int(bb))
            if len(src_list) >= int(n_samples):
                break
        attempts += batch
    return np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)


def _pair_null_features(graph, node_df: pd.DataFrame, deg: np.ndarray, src: np.ndarray, dst: np.ndarray, tile_density: dict[str, float]) -> pd.DataFrame:
    xi = node_df.loc[src, ["x_m", "y_m"]].to_numpy(dtype=np.float64)
    xj = node_df.loc[dst, ["x_m", "y_m"]].to_numpy(dtype=np.float64)
    dist = np.sqrt(np.nansum(np.square(xi - xj), axis=1))
    src_tile = node_df.loc[src, "tile_id"].to_numpy()
    dst_tile = node_df.loc[dst, "tile_id"].to_numpy()
    src_density = np.asarray([tile_density.get(str(t), 0.0) for t in src_tile], dtype=np.float64)
    dst_density = np.asarray([tile_density.get(str(t), 0.0) for t in dst_tile], dtype=np.float64)
    x_src = graph.x[src].astype(np.float32, copy=False)
    x_dst = graph.x[dst].astype(np.float32, copy=False)
    node_feat_l2 = np.sqrt(np.mean(np.square(x_src - x_dst), axis=1)).astype(np.float64)
    out = pd.DataFrame(
        {
            "log1p_distance_m": np.log1p(np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)),
            "degree_mean": 0.5 * (deg[src] + deg[dst]),
            "degree_gap": np.abs(deg[src] - deg[dst]),
            "tile_density_mean": 0.5 * (src_density + dst_density),
            "tile_density_gap": np.abs(src_density - dst_density),
            "log1p_plot_area_gap": np.log1p(np.abs(node_df.loc[src, "plot_area"].to_numpy(dtype=np.float64) - node_df.loc[dst, "plot_area"].to_numpy(dtype=np.float64))),
            "building_count_gap": np.abs(node_df.loc[src, "building_count"].to_numpy(dtype=np.float64) - node_df.loc[dst, "building_count"].to_numpy(dtype=np.float64)),
            "street_count_gap": np.abs(node_df.loc[src, "street_count"].to_numpy(dtype=np.float64) - node_df.loc[dst, "street_count"].to_numpy(dtype=np.float64)),
            "node_feat_l2": node_feat_l2,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_null_model(
    graph,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    tile_density: dict[str, float],
    seed: int,
    max_positive: int,
    max_negative: int,
    negative_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = graph.x.shape[0]
    edge_index_u, _, _ = undirected_unique(graph.edge_index, graph.edge_attr, graph.edge_weight, num_nodes=n)
    src_pos_all = edge_index_u[0].astype(np.int64, copy=False)
    dst_pos_all = edge_index_u[1].astype(np.int64, copy=False)
    deg = _degree(n, edge_index_u)
    rng = np.random.default_rng(int(seed) + 101)
    pos_idx = np.arange(src_pos_all.size)
    if pos_idx.size > int(max_positive):
        pos_idx = np.sort(rng.choice(pos_idx, size=int(max_positive), replace=False))
    src_pos = src_pos_all[pos_idx]
    dst_pos = dst_pos_all[pos_idx]
    neg_n = min(int(max_negative), max(1, int(len(src_pos) * float(negative_ratio))))
    lo = np.minimum(src_pos_all, dst_pos_all)
    hi = np.maximum(src_pos_all, dst_pos_all)
    existing_keys = set((lo.astype(np.int64) * int(n) + hi.astype(np.int64)).tolist())
    src_neg, dst_neg = _sample_negative_pairs(n, existing_keys, neg_n, seed=seed)

    x_pos = _pair_null_features(graph, node_df, deg, src_pos, dst_pos, tile_density)
    x_neg = _pair_null_features(graph, node_df, deg, src_neg, dst_neg, tile_density)
    x_train = pd.concat([x_pos, x_neg], ignore_index=True)
    y_train = np.concatenate([np.ones(len(x_pos), dtype=np.int8), np.zeros(len(x_neg), dtype=np.int8)])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, tol=1e-3, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(x_train.to_numpy(dtype=np.float64), y_train)

    x_edges = _pair_null_features(graph, node_df, deg, src_pos_all, dst_pos_all, tile_density)
    p = model.predict_proba(x_edges.to_numpy(dtype=np.float64))[:, 1]
    p = np.clip(p, 1e-5, 1 - 1e-5)
    residual = 1.0 - p
    residual_z = residual / np.sqrt(p * (1.0 - p))
    residual_df = edge_df.copy()
    residual_df["null_expected_edge_prob"] = p
    residual_df["edge_residual"] = residual
    residual_df["edge_residual_z"] = residual_z
    residual_df["positive_residual"] = residual > np.quantile(residual, 0.75)
    residual_df["negative_residual"] = residual < np.quantile(residual, 0.25)
    meta = {
        "n_positive_train": int(len(x_pos)),
        "n_negative_train": int(len(x_neg)),
        "feature_names": list(x_train.columns),
        "positive_train_rate": float(np.mean(y_train)),
    }
    return residual_df, meta


def _tile_features_for_city(
    city: str,
    graph,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    tile_size_m: float,
) -> pd.DataFrame:
    tile_area_km2 = (float(tile_size_m) * float(tile_size_m)) / 1_000_000.0
    deg_by_node = np.zeros((graph.x.shape[0],), dtype=np.float64)
    np.add.at(deg_by_node, edge_df["src"].to_numpy(dtype=np.int64), 1)
    np.add.at(deg_by_node, edge_df["dst"].to_numpy(dtype=np.int64), 1)
    node_df = node_df.copy()
    node_df["degree"] = deg_by_node
    valid_nodes = node_df[node_df["tile_id"] != "invalid"].copy()
    global_same = pd.to_numeric(edge_df.loc[edge_df["known_edge"], "same_label"], errors="coerce").mean()

    edge_incident_rows = []
    for side in ["src", "dst"]:
        tmp = residual_df.copy()
        tmp["incident_tile_id"] = node_df.loc[tmp[side].to_numpy(dtype=np.int64), "tile_id"].to_numpy()
        edge_incident_rows.append(tmp[tmp["incident_tile_id"] != "invalid"])
    incident = pd.concat(edge_incident_rows, ignore_index=True) if edge_incident_rows else pd.DataFrame()
    within = residual_df[(residual_df["within_tile"]) & (residual_df["src_tile_id"] != "invalid")].copy()

    rows = []
    for tile_id, nodes in valid_nodes.groupby("tile_id"):
        idx = nodes["node_idx"].to_numpy(dtype=np.int64)
        inc = incident[incident["incident_tile_id"] == tile_id] if not incident.empty else pd.DataFrame()
        win = within[within["src_tile_id"] == tile_id] if not within.empty else pd.DataFrame()
        known_labels = nodes.loc[nodes["known_label"], "label"].to_numpy(dtype=np.float64)
        coarse_known = nodes.loc[nodes["coarse_label"] != 5, "coarse_label"].to_numpy(dtype=np.float64)
        same_label_rate = pd.to_numeric(win.loc[win["known_edge"], "same_label"], errors="coerce").mean() if not win.empty else np.nan
        n_possible = max(1, len(idx) * (len(idx) - 1) / 2)
        n_components = _union_find_components(idx, win["src"].to_numpy(dtype=np.int64), win["dst"].to_numpy(dtype=np.int64)) if not win.empty else len(idx)
        relation_cols = [r for r in RELATION_FEATURES if r in inc.columns]
        relation_means = {r: float(pd.to_numeric(inc[r], errors="coerce").mean()) if not inc.empty else 0.0 for r in relation_cols}
        lon_vals = nodes["lon"].to_numpy(dtype=np.float64)
        lat_vals = nodes["lat"].to_numpy(dtype=np.float64)
        semantic_edge_share = float(
            relation_means.get("is_street_backed", 0.0)
            + relation_means.get("is_junction_only", 0.0)
            + relation_means.get("is_shared_building", 0.0)
        )
        semantic_edge_share = float(np.clip(semantic_edge_share, 0.0, 1.0))
        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "tile_id": tile_id,
                "city_tile_id": f"{city}:{tile_id}",
                "tile_x": int(nodes["tile_x"].iloc[0]),
                "tile_y": int(nodes["tile_y"].iloc[0]),
                "tile_size_m": float(tile_size_m),
                "n_nodes": int(len(nodes)),
                "n_known_nodes": int(nodes["known_label"].sum()),
                "known_label_ratio": float(nodes["known_label"].mean()) if len(nodes) else np.nan,
                "label_entropy": _entropy_norm(known_labels),
                "coarse_label_entropy": _entropy_norm(coarse_known),
                "n_edges_incident": int(len(inc)),
                "n_edges_within": int(len(win)),
                "density_proxy_nodes_per_km2": float(len(nodes) / max(tile_area_km2, 1e-9)),
                "induced_edge_density": float(len(win) / n_possible),
                "mean_degree": float(nodes["degree"].mean()),
                "degree_var": float(nodes["degree"].var(ddof=0)),
                "component_fragmentation": float(n_components / max(1, len(idx))),
                "boundary_crossing_edge_ratio": float((~inc["within_tile"]).mean()) if not inc.empty else np.nan,
                "same_label_rate": float(same_label_rate) if np.isfinite(same_label_rate) else np.nan,
                "same_label_lift": float(same_label_rate - global_same) if np.isfinite(same_label_rate) and np.isfinite(global_same) else np.nan,
                "street_edge_share": relation_means.get("is_street_backed", 0.0),
                "junction_edge_share": relation_means.get("is_junction_only", 0.0),
                "geom_edge_share": relation_means.get("is_geom_fallback", 0.0),
                "building_edge_share": relation_means.get("is_shared_building", 0.0),
                "semantic_edge_share": semantic_edge_share,
                "centroid_dist_m_mean": float(pd.to_numeric(inc.get("centroid_dist_m", pd.Series(dtype=float)), errors="coerce").mean()) if not inc.empty else np.nan,
                "centroid_dist_m_std": float(pd.to_numeric(inc.get("centroid_dist_m", pd.Series(dtype=float)), errors="coerce").std()) if not inc.empty else np.nan,
                "node_feat_l2_mean": float(pd.to_numeric(inc.get("node_feat_l2", pd.Series(dtype=float)), errors="coerce").mean()) if not inc.empty else np.nan,
                "edge_residual_mean": float(pd.to_numeric(win.get("edge_residual", pd.Series(dtype=float)), errors="coerce").mean()) if not win.empty else np.nan,
                "edge_residual_z_mean": float(pd.to_numeric(win.get("edge_residual_z", pd.Series(dtype=float)), errors="coerce").mean()) if not win.empty else np.nan,
                "edge_residual_z_std": float(pd.to_numeric(win.get("edge_residual_z", pd.Series(dtype=float)), errors="coerce").std()) if not win.empty else np.nan,
                "high_residual_edge_share": float(pd.to_numeric(win.get("positive_residual", pd.Series(dtype=float)), errors="coerce").mean()) if not win.empty else np.nan,
                "negative_residual_edge_share": float(pd.to_numeric(win.get("negative_residual", pd.Series(dtype=float)), errors="coerce").mean()) if not win.empty else np.nan,
                "plot_area_mean": float(nodes["plot_area"].mean()),
                "plot_area_std": float(nodes["plot_area"].std(ddof=0)),
                "building_count_mean": float(nodes["building_count"].mean()),
                "street_count_mean": float(nodes["street_count"].mean()),
                "building_area_mean": float(nodes["building_area_mean"].mean()),
                "lon_mean": float(np.nanmean(lon_vals)),
                "lat_mean": float(np.nanmean(lat_vals)),
                "lon_min": float(np.nanmin(lon_vals)),
                "lon_max": float(np.nanmax(lon_vals)),
                "lat_min": float(np.nanmin(lat_vals)),
                "lat_max": float(np.nanmax(lat_vals)),
            }
        )
    return pd.DataFrame(rows)


def _detector_scores(tile_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for city, cdf in tile_df.groupby("city", sort=False):
        cur = cdf.copy()
        relation_cols = ["street_edge_share", "junction_edge_share", "building_edge_share", "geom_edge_share"]
        relation_z = np.column_stack([_robust_z(cur[c]).fillna(0.0).to_numpy(dtype=np.float64) for c in relation_cols])
        cur["relation_mix_outlier_score"] = np.sqrt(np.sum(np.square(relation_z), axis=1))
        cur["residual_edge_outlier_score"] = (
            _robust_z(cur["edge_residual_z_mean"]).abs().fillna(0.0)
            + 0.5 * _robust_z(cur["high_residual_edge_share"]).abs().fillna(0.0)
            + 0.5 * _robust_z(cur["negative_residual_edge_share"]).abs().fillna(0.0)
        )
        cur["geometry_confound_score"] = (
            _robust_z(cur["geom_edge_share"]).fillna(0.0)
            + 0.5 * _robust_z(cur["centroid_dist_m_mean"]).fillna(0.0)
            - 0.5 * _robust_z(cur["semantic_edge_share"]).fillna(0.0)
        )
        if len(cur) >= 8:
            form_cols = [c for c in FORM_FEATURES_FOR_MISMATCH if c in cur.columns]
            graph_cols = [c for c in GRAPH_FEATURES_FOR_MISMATCH if c in cur.columns]
            form_z = _standardize_frame(cur, form_cols)
            graph_z = _standardize_frame(cur, graph_cols)
            k = min(6, len(cur) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nn.fit(form_z)
            _, idx = nn.kneighbors(form_z)
            pred = np.zeros_like(graph_z)
            for i in range(len(cur)):
                neigh = idx[i, 1:]
                pred[i] = graph_z[neigh].mean(axis=0)
            cur["graph_form_mismatch_score"] = np.sqrt(np.mean(np.square(graph_z - pred), axis=1))
        else:
            graph_cols = [c for c in GRAPH_FEATURES_FOR_MISMATCH if c in cur.columns]
            graph_z = _standardize_frame(cur, graph_cols)
            cur["graph_form_mismatch_score"] = np.sqrt(np.mean(np.square(graph_z), axis=1))
        detector_cols = [
            "relation_mix_outlier_score",
            "residual_edge_outlier_score",
            "geometry_confound_score",
            "graph_form_mismatch_score",
        ]
        for col in detector_cols:
            cur[f"{col}_z"] = _robust_z(cur[col]).fillna(0.0)
            cur[f"{col}_percentile"] = cur[col].rank(pct=True)
        cur["combined_anomaly_score"] = cur[[f"{c}_percentile" for c in detector_cols]].mean(axis=1)
        cur["city_anomaly_rank"] = cur["combined_anomaly_score"].rank(ascending=False, method="first").astype(int)
        score_map = cur[detector_cols].copy()
        score_map["geometry_confound_score"] = score_map["geometry_confound_score"].abs()
        top_detector = score_map.idxmax(axis=1).str.replace("_score", "", regex=False)
        cur["tile_typology"] = np.where(
            cur["combined_anomaly_score"] < 0.65,
            "stable/control tile",
            top_detector + " tile",
        )
        frames.append(cur)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _weak_label_validation(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detector_cols = [
        "combined_anomaly_score",
        "relation_mix_outlier_score",
        "residual_edge_outlier_score",
        "geometry_confound_score",
        "graph_form_mismatch_score",
    ]
    label_cols = ["known_label_ratio", "label_entropy", "coarse_label_entropy", "same_label_lift"]
    for city, cdf in scores.groupby("city", sort=False):
        for detector in detector_cols:
            for label_col in label_cols:
                x = pd.to_numeric(cdf[detector], errors="coerce")
                y = pd.to_numeric(cdf[label_col], errors="coerce")
                ok = x.notna() & y.notna()
                corr = float(x[ok].corr(y[ok], method="spearman")) if ok.sum() >= 5 else np.nan
                rows.append(
                    {
                        "city": city,
                        "detector": detector,
                        "weak_label_metric": label_col,
                        "spearman_rho": corr,
                        "n_tiles": int(ok.sum()),
                    }
                )
    return pd.DataFrame(rows)


def _city_typology(scores: pd.DataFrame, factor_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.DataFrame()
    if factor_root is not None and (factor_root / "city_decision_matrix.csv").exists():
        try:
            prior = pd.read_csv(factor_root / "city_decision_matrix.csv")
        except Exception:
            prior = pd.DataFrame()
    rows = []
    for city, cdf in scores.groupby("city", sort=False):
        rel_p95 = float(cdf["relation_mix_outlier_score"].quantile(0.95))
        residual_p95 = float(cdf["residual_edge_outlier_score"].quantile(0.95))
        geom_p95 = float(cdf["geometry_confound_score"].abs().quantile(0.95))
        mismatch_p95 = float(cdf["graph_form_mismatch_score"].quantile(0.95))
        combined_p95 = float(cdf["combined_anomaly_score"].quantile(0.95))
        known_mean = float(cdf["known_label_ratio"].mean())
        semantic_mean = float(cdf["semantic_edge_share"].mean())
        geom_mean = float(cdf["geom_edge_share"].mean())
        prior_row = prior[prior["city"] == city] if not prior.empty and "city" in prior.columns else pd.DataFrame()
        projection_risk = False
        spatial_confound = False
        label_unstable = False
        if not prior_row.empty:
            projection_risk = _truthy(prior_row.get("projection_rule_sensitive", pd.Series([False])).iloc[0])
            spatial_confound = _truthy(prior_row.get("spatial_geometry_confound", pd.Series([False])).iloc[0])
            label_unstable = _truthy(prior_row.get("label_unstable", pd.Series([False])).iloc[0])
        label_limited = bool(known_mean < 0.25 or label_unstable)
        strongest_other_for_residual = max(rel_p95, geom_p95, mismatch_p95)
        strongest_other_for_geometry = max(rel_p95, residual_p95, mismatch_p95)
        if residual_p95 > 2.0 and residual_p95 >= 0.9 * strongest_other_for_residual and semantic_mean >= geom_mean:
            typology = "relation-structured city"
        elif spatial_confound and geom_p95 >= max(2.0, 0.9 * strongest_other_for_geometry):
            typology = "geometry-driven city"
        elif projection_risk and rel_p95 > 2.0:
            typology = "projection-risk city"
        elif combined_p95 >= 0.85 or mismatch_p95 > 1.5:
            typology = "heterogeneous/anomaly-rich city"
        elif label_limited:
            typology = "label-limited city"
        else:
            typology = "stable/control city"
        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "n_tiles": int(len(cdf)),
                "city_typology": typology,
                "relation_mix_outlier_p95": rel_p95,
                "residual_edge_outlier_p95": residual_p95,
                "geometry_confound_p95": geom_p95,
                "graph_form_mismatch_p95": mismatch_p95,
                "combined_anomaly_p95": combined_p95,
                "known_label_ratio_mean": known_mean,
                "label_limited_flag": label_limited,
                "semantic_edge_share_mean": semantic_mean,
                "geom_edge_share_mean": geom_mean,
                "prior_projection_rule_sensitive": projection_risk,
                "prior_spatial_geometry_confound": spatial_confound,
                "prior_label_unstable": label_unstable,
            }
        )
    profile = pd.DataFrame(rows)
    profile_cols = [
        "relation_mix_outlier_p95",
        "residual_edge_outlier_p95",
        "geometry_confound_p95",
        "graph_form_mismatch_p95",
        "combined_anomaly_p95",
        "semantic_edge_share_mean",
        "geom_edge_share_mean",
        "known_label_ratio_mean",
    ]
    if profile.shape[0] >= 2:
        z = _standardize_frame(profile, profile_cols)
        dist = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).mean(axis=2))
        sim_rows = []
        for i, city_i in enumerate(profile["city"]):
            for j, city_j in enumerate(profile["city"]):
                sim_rows.append({"city_i": city_i, "city_j": city_j, "profile_distance": float(dist[i, j])})
        sim = pd.DataFrame(sim_rows)
    else:
        sim = pd.DataFrame(columns=["city_i", "city_j", "profile_distance"])
    return profile, sim


def _baseline_profile(city: str, graph, node_df: pd.DataFrame, edge_df: pd.DataFrame, tile_df: pd.DataFrame) -> dict[str, Any]:
    known = known_mask(graph.labels, graph.meta)
    edge_index_u, _, _ = undirected_unique(graph.edge_index, graph.edge_attr, graph.edge_weight, graph.x.shape[0])
    n_nodes = graph.x.shape[0]
    n_edges = edge_index_u.shape[1]
    deg = _degree(n_nodes, edge_index_u)
    relation_means = {r: float(pd.to_numeric(edge_df[r], errors="coerce").mean()) if r in edge_df else np.nan for r in RELATION_FEATURES}
    return {
        "city": city,
        "city_label": city_label(city),
        "dataset": graph.dataset,
        "n_nodes": int(n_nodes),
        "n_edges_undirected": int(n_edges),
        "graph_density": float(2 * n_edges / max(1, n_nodes * (n_nodes - 1))),
        "mean_degree": float(np.mean(deg)),
        "degree_p95": float(np.quantile(deg, 0.95)) if deg.size else np.nan,
        "known_label_ratio": float(np.mean(known)),
        "n_tiles": int(tile_df.shape[0]),
        "median_tile_nodes": float(tile_df["n_nodes"].median()) if not tile_df.empty else np.nan,
        "street_edge_share": relation_means.get("is_street_backed", np.nan),
        "junction_edge_share": relation_means.get("is_junction_only", np.nan),
        "geom_edge_share": relation_means.get("is_geom_fallback", np.nan),
        "building_edge_share": relation_means.get("is_shared_building", np.nan),
        "largest_component_ratio_proxy": np.nan,
    }


def _tile_geojson(scores: pd.DataFrame, top_k: int, out_path: Path) -> None:
    features = []
    for city, cdf in scores.groupby("city", sort=False):
        top = cdf.sort_values("combined_anomaly_score", ascending=False).head(int(top_k))
        for _, row in top.iterrows():
            lon_min = float(row["lon_min"])
            lon_max = float(row["lon_max"])
            lat_min = float(row["lat_min"])
            lat_max = float(row["lat_max"])
            if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
                continue
            if abs(lon_max - lon_min) < 1e-5:
                lon_min -= 0.0005
                lon_max += 0.0005
            if abs(lat_max - lat_min) < 1e-5:
                lat_min -= 0.0005
                lat_max += 0.0005
            props = {
                "city": row["city"],
                "city_label": row["city_label"],
                "tile_id": row["tile_id"],
                "tile_typology": row["tile_typology"],
                "combined_anomaly_score": float(row["combined_anomaly_score"]),
                "city_anomaly_rank": int(row["city_anomaly_rank"]),
                "relation_mix_outlier_score": float(row["relation_mix_outlier_score"]),
                "residual_edge_outlier_score": float(row["residual_edge_outlier_score"]),
                "geometry_confound_score": float(row["geometry_confound_score"]),
                "graph_form_mismatch_score": float(row["graph_form_mismatch_score"]),
                "n_nodes": int(row["n_nodes"]),
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [lon_min, lat_min],
                                [lon_max, lat_min],
                                [lon_max, lat_max],
                                [lon_min, lat_max],
                                [lon_min, lat_min],
                            ]
                        ],
                    },
                }
            )
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def _write_manifest(out_dir: Path, args: argparse.Namespace, null_meta_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    manifest = {
        "created_at_unix": time.time(),
        "args": vars(args),
        "null_models": null_meta_rows,
        "n_failures": len(failures),
        "failures": failures,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    base_root = Path(args.base_graph_root)
    urban_root = Path(args.urban_root)
    factor_root = Path(args.factor_results_root) if args.factor_results_root else None
    variant = str(args.variant).strip()
    cities = parse_cities(args.cities, graph_root=base_root, pattern=f"urban_*_plot_{variant}")

    baseline_rows: list[dict[str, Any]] = []
    tile_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    null_meta_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for city in cities:
        dataset = city_to_dataset(city, variant)
        try:
            print(f"[city] {city}", flush=True)
            graph = load_sparse_graph(base_root, dataset)
            node_df = _assign_tiles(_node_context(base_root, urban_root, city, graph), float(args.tile_size_m))
            edge_df = _edge_context(graph, node_df)
            tile_density = (
                node_df[node_df["tile_id"] != "invalid"]
                .groupby("tile_id")
                .size()
                .div((float(args.tile_size_m) * float(args.tile_size_m)) / 1_000_000.0)
                .to_dict()
            )
            residual_df, null_meta = _fit_null_model(
                graph,
                node_df,
                edge_df,
                tile_density,
                seed=int(args.seed),
                max_positive=int(args.max_null_positive),
                max_negative=int(args.max_null_negative),
                negative_ratio=float(args.negative_ratio),
            )
            null_meta.update({"city": city, "dataset": dataset})
            null_meta_rows.append(null_meta)
            tile_df = _tile_features_for_city(city, graph, node_df, edge_df, residual_df, float(args.tile_size_m))
            baseline_rows.append(_baseline_profile(city, graph, node_df, edge_df, tile_df))
            residual_frames.append(residual_df.assign(city=city, dataset=dataset))
            tile_frames.append(tile_df)
        except Exception as exc:
            failures.append(
                {
                    "city": city,
                    "dataset": dataset,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    baseline = pd.DataFrame(baseline_rows)
    tile_features = pd.concat(tile_frames, ignore_index=True) if tile_frames else pd.DataFrame()
    residual_table = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    scores = _detector_scores(tile_features) if not tile_features.empty else pd.DataFrame()
    weak = _weak_label_validation(scores) if not scores.empty else pd.DataFrame()
    typology, similarity = _city_typology(scores, factor_root=factor_root) if not scores.empty else (pd.DataFrame(), pd.DataFrame())

    baseline.to_csv(out_dir / "city_baseline_profile.csv", index=False)
    tile_features.to_csv(out_dir / "tile_feature_table.csv", index=False)
    residual_table.to_csv(out_dir / "edge_residual_table.csv", index=False)
    scores.to_csv(out_dir / "tile_anomaly_scores.csv", index=False)
    weak.to_csv(out_dir / "weak_label_validation.csv", index=False)
    typology.to_csv(out_dir / "city_typology_summary.csv", index=False)
    similarity.to_csv(out_dir / "city_similarity_matrix.csv", index=False)
    pd.DataFrame(failures, columns=["city", "dataset", "error", "traceback"]).to_csv(out_dir / "failures.csv", index=False)
    if not scores.empty:
        _tile_geojson(scores, top_k=int(args.top_k_tiles), out_path=out_dir / "top_anomaly_tiles.geojson")
    else:
        (out_dir / "top_anomaly_tiles.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    _write_manifest(out_dir, args, null_meta_rows, failures)
    write_readme(
        out_dir / "README.md",
        "Urban Data-Paper Tile Anomaly Detection",
        [
            "This pipeline supports a cross-city Urban graph dataset paper.",
            "",
            "Main outputs:",
            "- `city_baseline_profile.csv`: city-level graph baseline audit.",
            "- `tile_feature_table.csv`: graph/form/weak-label features for each city tile.",
            "- `edge_residual_table.csv`: spatial-degree corrected residuals for observed edges.",
            "- `tile_anomaly_scores.csv`: interpretable detector scores and tile typology.",
            "- `city_typology_summary.csv`: cross-city graph mechanism typology.",
            "- `weak_label_validation.csv`: weak-label-only validation correlations.",
            "- `top_anomaly_tiles.geojson`: top tile anomalies per city for map display.",
            "",
            f"Graph variant: `{variant}`.",
            "Labels are not used by the detectors; they are only used for weak-label validation.",
        ],
    )
    return {
        "out_dir": str(out_dir),
        "cities": cities,
        "n_failures": len(failures),
        "n_tiles": int(len(tile_features)),
        "n_edges_residual": int(len(residual_table)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Urban data-paper tile anomaly detection pipeline.")
    parser.add_argument("--cities", type=str, default="all")
    parser.add_argument("--variant", type=str, default="v3bsjg")
    parser.add_argument("--urban_root", type=str, default=DEFAULT_URBAN_ROOT)
    parser.add_argument("--base_graph_root", type=str, default=DEFAULT_BASE_GRAPH_ROOT)
    parser.add_argument("--factor_results_root", type=str, default="results/urban_factor_experiments")
    parser.add_argument("--out_dir", type=str, default="results/urban_data_paper")
    parser.add_argument("--tile_size_m", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_null_positive", type=int, default=120000)
    parser.add_argument("--max_null_negative", type=int, default=120000)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--top_k_tiles", type=int, default=25)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
