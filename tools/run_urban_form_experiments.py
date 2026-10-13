#!/usr/bin/env python3
"""Run Urban-form factor controls for existing V3b plot graphs."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import prepare_urban_plot_graph_v3 as v3
from urban_factor_experiment_utils import (
    add_probe_deltas,
    city_label,
    city_to_dataset,
    ensure_dir,
    feature_index,
    load_sparse_graph,
    parse_cities,
    parse_int_csv,
    prepare_known_edges,
    run_edge_probe_arrays,
    sample_edge_indices,
    write_readme,
)


STRATIFY_METRICS = [
    "plot_area",
    "building_count",
    "street_count",
    "building_area_mean",
    "centroid_dist_m",
]
MATCH_METRICS = ["distance", "degree_mean", "plot_area_gap", "node_feat_l2"]


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


def _read_node_table(base: Path, dataset: str, n: int) -> pd.DataFrame:
    path = base / dataset / f"{dataset}_node_table.csv"
    if not path.exists():
        return pd.DataFrame({"node_idx": np.arange(n)})
    df = pd.read_csv(path)
    if len(df) != n:
        df = df.reindex(range(n)).reset_index(drop=True)
        df["node_idx"] = np.arange(n)
    return df


def _raw_plot_frame(urban_root: Path, city: str, n: int) -> pd.DataFrame:
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


def _centroids_m(plot_df: pd.DataFrame, n: int) -> np.ndarray:
    if "geometry" not in plot_df.columns:
        return np.full((n, 2), np.nan, dtype=np.float64)
    try:
        lonlat = v3._geometry_centroids_from_wkb(plot_df["geometry"])
        if lonlat.shape[0] != n:
            lonlat = np.resize(lonlat, (n, 2))
        return v3._lonlat_to_local_m(lonlat)
    except Exception:
        return np.full((n, 2), np.nan, dtype=np.float64)


def _quantile_strata(values: np.ndarray, q: int = 4) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int16)
    finite = np.isfinite(arr)
    if finite.sum() < 10 or np.unique(arr[finite]).size < 2:
        return out
    try:
        codes = pd.qcut(arr[finite], q=min(q, np.unique(arr[finite]).size), labels=False, duplicates="drop")
        out[np.flatnonzero(finite)] = np.asarray(codes, dtype=np.int16)
    except Exception:
        return out
    return out


def _degree(num_nodes: int, src_all: np.ndarray, dst_all: np.ndarray) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.float64)
    np.add.at(deg, src_all.astype(np.int64), 1.0)
    np.add.at(deg, dst_all.astype(np.int64), 1.0)
    return deg


def _matched_indices(
    target: np.ndarray,
    match_frame: pd.DataFrame,
    seed: int,
    max_edges: int,
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(int(seed) + 211)
    if target.size == 0:
        return np.zeros((0,), dtype=np.int64), "empty"
    keys = []
    for col in MATCH_METRICS:
        codes = _quantile_strata(match_frame[col].to_numpy(dtype=np.float64), q=4)
        keys.append(codes)
    key_arr = np.stack(keys, axis=1)
    valid = np.all(key_arr >= 0, axis=1)
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx in np.flatnonzero(valid):
        groups.setdefault(tuple(int(x) for x in key_arr[idx]), []).append(int(idx))
    chosen: list[int] = []
    for vals in groups.values():
        arr = np.asarray(vals, dtype=np.int64)
        pos = arr[target[arr] == 1]
        neg = arr[target[arr] == 0]
        m = min(pos.size, neg.size)
        if m <= 0:
            continue
        chosen.extend(rng.choice(pos, size=m, replace=False).tolist())
        chosen.extend(rng.choice(neg, size=m, replace=False).tolist())
    chosen_arr = np.asarray(chosen, dtype=np.int64)
    if chosen_arr.size < 40:
        return sample_edge_indices(target, seed=seed, max_edges=max_edges, mode="target_balanced"), "fallback_target_balanced"
    if chosen_arr.size > int(max_edges):
        chosen_arr = rng.choice(chosen_arr, size=int(max_edges), replace=False)
    return np.sort(chosen_arr), "matched_quantile"


def _probe_subset(
    graph,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    target: np.ndarray,
    idx: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    return run_edge_probe_arrays(
        graph=graph,
        src=src[idx],
        dst=dst[idx],
        edge_attr=edge_attr[idx],
        target=target[idx],
        seed=seed,
    )


def _load_city_context(base_graph_root: Path, urban_root: Path, city: str):
    dataset = city_to_dataset(city, "v3bsjg")
    graph = load_sparse_graph(base_graph_root, dataset)
    n = graph.x.shape[0]
    node_table = _read_node_table(base_graph_root, dataset, n)
    plot_df = _raw_plot_frame(urban_root, city, n)

    plot_area = pd.to_numeric(plot_df.get("plot_area", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    if plot_area.shape[0] != n or not np.isfinite(plot_area).any():
        plot_area = np.expm1(_node_feature(graph, "log_plot_area", default=np.nan))
    building_count = pd.to_numeric(
        node_table.get("building_membership_count", np.expm1(_node_feature(graph, "log_building_count"))),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    street_count = pd.to_numeric(
        node_table.get("street_membership_count", np.expm1(_node_feature(graph, "log_street_count"))),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    building_area_mean = _building_area_mean(urban_root, city, n)
    centroids = _centroids_m(plot_df, n)

    src, dst, edge_attr, target = prepare_known_edges(graph)
    full_src, full_dst, full_edge_attr, _ = prepare_known_edges(
        graph,
        labels=np.where(graph.labels >= 0, 0, -1),
        known=np.ones((n,), dtype=bool),
    )
    deg = _degree(n, full_src, full_dst)

    edge_metrics = pd.DataFrame(
        {
            "plot_area": 0.5 * (plot_area[src] + plot_area[dst]),
            "building_count": 0.5 * (building_count[src] + building_count[dst]),
            "street_count": 0.5 * (street_count[src] + street_count[dst]),
            "building_area_mean": 0.5 * (building_area_mean[src] + building_area_mean[dst]),
            "centroid_dist_m": _edge_feature(edge_attr, graph.feature_names, "centroid_dist_m"),
            "distance": _edge_feature(edge_attr, graph.feature_names, "centroid_dist_m"),
            "degree_mean": 0.5 * (deg[src] + deg[dst]),
            "plot_area_gap": np.abs(plot_area[src] - plot_area[dst]),
            "node_feat_l2": _edge_feature(edge_attr, graph.feature_names, "node_feat_l2"),
        }
    )

    node_metrics = pd.DataFrame(
        {
            "plot_area": plot_area,
            "building_count": building_count,
            "street_count": street_count,
            "building_area_mean": building_area_mean,
            "tile_x": np.nan,
            "tile_y": np.nan,
        }
    )
    if np.isfinite(centroids).any():
        node_metrics["centroid_x_m"] = centroids[:, 0]
        node_metrics["centroid_y_m"] = centroids[:, 1]
    else:
        node_metrics["centroid_x_m"] = np.nan
        node_metrics["centroid_y_m"] = np.nan

    return graph, src, dst, edge_attr, target, edge_metrics, node_metrics


def _stratified_rows(
    *,
    graph,
    dataset: str,
    city: str,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    target: np.ndarray,
    edge_metrics: pd.DataFrame,
    seeds: list[int],
    max_edges: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in STRATIFY_METRICS:
        strata = _quantile_strata(edge_metrics[metric].to_numpy(dtype=np.float64), q=4)
        for sid in sorted(int(x) for x in np.unique(strata) if int(x) >= 0):
            candidates = np.flatnonzero(strata == sid)
            if candidates.size < 20:
                continue
            for seed in seeds:
                sampled_local = sample_edge_indices(target[candidates], seed=seed, max_edges=max_edges, mode="random")
                idx = candidates[sampled_local]
                probe_rows = _probe_subset(graph, src, dst, edge_attr, target, idx, seed)
                vals = edge_metrics[metric].to_numpy(dtype=np.float64)[candidates]
                for row in probe_rows:
                    row.update(
                        {
                            "city": city,
                            "city_label": city_label(city),
                            "dataset": dataset,
                            "stratify_metric": metric,
                            "stratum_id": sid,
                            "stratum_label": f"q{sid + 1}",
                            "stratum_value_min": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
                            "stratum_value_max": float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan,
                            "n_edges_in_stratum": int(candidates.size),
                            "sample_mode": "within_stratum_random",
                        }
                    )
                rows.extend(probe_rows)
    return rows


def _matched_rows(
    *,
    graph,
    dataset: str,
    city: str,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    target: np.ndarray,
    edge_metrics: pd.DataFrame,
    seeds: list[int],
    max_edges: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        full_idx = sample_edge_indices(target, seed=seed, max_edges=max_edges, mode="random")
        full_rows = _probe_subset(graph, src, dst, edge_attr, target, full_idx, seed)
        for row in full_rows:
            row.update(
                {
                    "city": city,
                    "city_label": city_label(city),
                    "dataset": dataset,
                    "match_condition": "full_reference",
                    "match_status": "random",
                    "sample_mode": "random",
                    "n_candidate_edges": int(target.shape[0]),
                }
            )
        rows.extend(full_rows)

        idx, status = _matched_indices(target, edge_metrics[MATCH_METRICS], seed=seed, max_edges=max_edges)
        matched_rows = _probe_subset(graph, src, dst, edge_attr, target, idx, seed)
        for row in matched_rows:
            row.update(
                {
                    "city": city,
                    "city_label": city_label(city),
                    "dataset": dataset,
                    "match_condition": "matched_distance_degree_area_l2",
                    "match_status": status,
                    "sample_mode": "matched_or_balanced",
                    "n_candidate_edges": int(target.shape[0]),
                }
            )
        rows.extend(matched_rows)
    return rows


def _tile_rows(
    *,
    graph,
    dataset: str,
    city: str,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    target: np.ndarray,
    edge_metrics: pd.DataFrame,
    node_metrics: pd.DataFrame,
    tile_size_m: float,
) -> list[dict[str, Any]]:
    x = node_metrics["centroid_x_m"].to_numpy(dtype=np.float64)
    y = node_metrics["centroid_y_m"].to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return []
    tx = np.full(x.shape, -1, dtype=np.int64)
    ty = np.full(y.shape, -1, dtype=np.int64)
    tx[valid] = np.floor((x[valid] - np.nanmin(x[valid])) / float(tile_size_m)).astype(np.int64)
    ty[valid] = np.floor((y[valid] - np.nanmin(y[valid])) / float(tile_size_m)).astype(np.int64)
    node_metrics = node_metrics.copy()
    node_metrics["tile_x"] = tx
    node_metrics["tile_y"] = ty

    known_node = np.zeros((graph.x.shape[0],), dtype=bool)
    known_node[src] = True
    known_node[dst] = True
    global_same = float(np.mean(target)) if target.size else np.nan

    rel_cols = [c for c in ["is_street_backed", "is_junction_only", "is_geom_fallback", "is_shared_building"] if feature_index(graph.feature_names, c) is not None]
    rel_arrays = {
        c: _edge_feature(edge_attr, graph.feature_names, c, default=0.0)
        for c in rel_cols
    }

    rows: list[dict[str, Any]] = []
    tile_groups = node_metrics[node_metrics["tile_x"] >= 0].groupby(["tile_x", "tile_y"], dropna=True)
    for (tile_x, tile_y), nodes in tile_groups:
        node_idx = nodes.index.to_numpy(dtype=np.int64)
        if node_idx.size < 5:
            continue
        in_tile_edge = (tx[src] == int(tile_x)) & (ty[src] == int(tile_y)) & (tx[dst] == int(tile_x)) & (ty[dst] == int(tile_y))
        edge_idx = np.flatnonzero(in_tile_edge)
        if edge_idx.size < 5:
            continue
        same_rate = float(np.mean(target[edge_idx])) if edge_idx.size else np.nan
        rel_means = {f"{name}_share": float(np.mean(vals[edge_idx])) for name, vals in rel_arrays.items()}
        rel_counts = np.asarray([max(0.0, rel_means[f"{name}_share"]) for name in rel_cols], dtype=np.float64)
        rel_total = float(rel_counts.sum())
        rel_entropy = 0.0
        if rel_total > 0 and rel_counts.size > 1:
            p = rel_counts[rel_counts > 0] / rel_total
            rel_entropy = float(-(p * np.log(p)).sum() / np.log(rel_counts.size))
        density_proxy = float(node_idx.size / max(1.0, float(tile_size_m) * float(tile_size_m) / 1_000_000.0))
        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "dataset": dataset,
                "tile_x": int(tile_x),
                "tile_y": int(tile_y),
                "n_nodes": int(node_idx.size),
                "n_known_nodes": int(known_node[node_idx].sum()),
                "known_label_ratio": float(known_node[node_idx].mean()) if node_idx.size else np.nan,
                "n_known_edges_within_tile": int(edge_idx.size),
                "same_label_rate_within_tile": same_rate,
                "global_same_label_rate": global_same,
                "edge_gain_proxy_same_label_lift": same_rate - global_same if np.isfinite(same_rate) and np.isfinite(global_same) else np.nan,
                "density_proxy_nodes_per_km2": density_proxy,
                "projection_tail_proxy_relation_entropy": rel_entropy,
                "mean_centroid_dist_m": float(np.nanmean(edge_metrics["centroid_dist_m"].to_numpy(dtype=np.float64)[edge_idx])),
                "mean_plot_area": float(np.nanmean(node_metrics.loc[node_idx, "plot_area"])),
                "mean_building_count": float(np.nanmean(node_metrics.loc[node_idx, "building_count"])),
                "mean_street_count": float(np.nanmean(node_metrics.loc[node_idx, "street_count"])),
                **rel_means,
            }
        )
    return rows


def _mean_gain(df: pd.DataFrame, feature_set: str = "endpoint_edge_attr", **filters: Any) -> float:
    if df.empty:
        return float("nan")
    cur = df.copy()
    for key, value in filters.items():
        cur = cur[cur[key] == value]
    cur = cur[cur["feature_set"] == feature_set]
    vals = pd.to_numeric(cur["auc_gain_vs_endpoint"], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _mean_drop(df: pd.DataFrame, **filters: Any) -> float:
    if df.empty:
        return float("nan")
    cur = df.copy()
    for key, value in filters.items():
        cur = cur[cur[key] == value]
    cur = cur[cur["feature_set"] == "endpoint_edge_attr"]
    vals = pd.to_numeric(cur["auc_drop_vs_perm"], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _build_summary(
    strat_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    tile_df: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    cities = sorted(
        set(strat_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(matched_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(tile_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | {str(x.get("city", "")) for x in failures if x.get("city")}
    )
    failed = {str(x.get("city")) for x in failures if x.get("city")}
    rows = []
    for city in cities:
        scity = strat_df[(strat_df["city"] == city) & (strat_df["feature_set"] == "endpoint_edge_attr")]
        metric_ranges = []
        for metric, group in scity.groupby("stratify_metric"):
            gains = pd.to_numeric(group["auc_gain_vs_endpoint"], errors="coerce").dropna()
            if not gains.empty:
                metric_ranges.append((metric, float(gains.max() - gains.min()), float(gains.max())))
        best_metric = max(metric_ranges, key=lambda x: x[1], default=("", float("nan"), float("nan")))
        strat_range = best_metric[1]
        strat_max = best_metric[2]

        full_gain = _mean_gain(matched_df, city=city, match_condition="full_reference")
        matched_gain = _mean_gain(matched_df, city=city, match_condition="matched_distance_degree_area_l2")
        matched_drop = _mean_drop(matched_df, city=city, match_condition="matched_distance_degree_area_l2")

        tcity = tile_df[tile_df["city"] == city]
        tile_lift_mean = float(pd.to_numeric(tcity.get("edge_gain_proxy_same_label_lift", pd.Series(dtype=float)), errors="coerce").mean()) if not tcity.empty else float("nan")
        tile_density_lift_corr = float("nan")
        if not tcity.empty:
            a = pd.to_numeric(tcity["density_proxy_nodes_per_km2"], errors="coerce")
            b = pd.to_numeric(tcity["edge_gain_proxy_same_label_lift"], errors="coerce")
            ok = a.notna() & b.notna()
            if ok.sum() >= 3:
                tile_density_lift_corr = float(np.corrcoef(a[ok], b[ok])[0, 1])

        form_mediated_edge_signal = bool(
            np.isfinite(strat_range)
            and np.isfinite(strat_max)
            and strat_range > 0.005
            and strat_max > 0.003
        )
        form_or_distance_confound = bool(
            np.isfinite(full_gain)
            and np.isfinite(matched_gain)
            and full_gain > 0.002
            and matched_gain <= 0.001
        )
        independent_edge_semantics_supported = bool(
            np.isfinite(matched_gain)
            and np.isfinite(matched_drop)
            and matched_gain > 0.003
            and matched_drop > 0.003
        )

        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "status": "failed" if city in failed else "ok",
                "largest_stratified_gain_range_metric": best_metric[0],
                "largest_stratified_edge_gain_range_auc": strat_range,
                "largest_stratified_edge_gain_auc": strat_max,
                "full_reference_edge_gain_auc": full_gain,
                "matched_edge_gain_auc": matched_gain,
                "matched_edge_attr_perm_drop_auc": matched_drop,
                "tile_edge_gain_proxy_mean": tile_lift_mean,
                "tile_density_lift_corr": tile_density_lift_corr,
                "form_mediated_edge_signal": form_mediated_edge_signal,
                "form_or_distance_confound": form_or_distance_confound,
                "independent_edge_semantics_supported": independent_edge_semantics_supported,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    base_graph_root = Path(args.base_graph_root)
    urban_root = Path(args.urban_root)
    cities = parse_cities(args.cities, graph_root=base_graph_root)
    seeds = parse_int_csv(args.seeds)

    strat_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for city in cities:
        dataset = city_to_dataset(city, "v3bsjg")
        try:
            graph, src, dst, edge_attr, target, edge_metrics, node_metrics = _load_city_context(
                base_graph_root,
                urban_root,
                city,
            )
            strat_rows.extend(
                _stratified_rows(
                    graph=graph,
                    dataset=dataset,
                    city=city,
                    src=src,
                    dst=dst,
                    edge_attr=edge_attr,
                    target=target,
                    edge_metrics=edge_metrics,
                    seeds=seeds,
                    max_edges=int(args.max_edges),
                )
            )
            matched_rows.extend(
                _matched_rows(
                    graph=graph,
                    dataset=dataset,
                    city=city,
                    src=src,
                    dst=dst,
                    edge_attr=edge_attr,
                    target=target,
                    edge_metrics=edge_metrics,
                    seeds=seeds,
                    max_edges=int(args.max_edges),
                )
            )
            tile_rows.extend(
                _tile_rows(
                    graph=graph,
                    dataset=dataset,
                    city=city,
                    src=src,
                    dst=dst,
                    edge_attr=edge_attr,
                    target=target,
                    edge_metrics=edge_metrics,
                    node_metrics=node_metrics,
                    tile_size_m=float(args.tile_size_m),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "city": city,
                    "dataset": dataset,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    strat_df = pd.DataFrame(strat_rows)
    matched_df = pd.DataFrame(matched_rows)
    tile_df = pd.DataFrame(tile_rows)

    if not strat_df.empty:
        strat_df = add_probe_deltas(
            strat_df,
            ["city", "dataset", "stratify_metric", "stratum_id", "stratum_label", "sample_mode"],
        )
    if not matched_df.empty:
        matched_df = add_probe_deltas(
            matched_df,
            ["city", "dataset", "match_condition", "match_status", "sample_mode"],
        )

    summary_df = _build_summary(strat_df, matched_df, tile_df, failures)

    strat_df.to_csv(out_dir / "stratified_edge_probe.csv", index=False)
    matched_df.to_csv(out_dir / "matched_edge_probe.csv", index=False)
    tile_df.to_csv(out_dir / "tile_level_form_summary.csv", index=False)
    summary_df.to_csv(out_dir / "urban_form_factor_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    write_readme(
        out_dir / "README.md",
        "Urban Form Factor Experiments",
        [
            "This package tests whether edge signal is mediated by urban-form strata, distance, degree, plot-area gaps, and local tile context.",
            "",
            "Outputs:",
            "- `stratified_edge_probe.csv`: within-city edge probe by plot area, counts, building form, and distance strata.",
            "- `matched_edge_probe.csv`: full reference versus distance/degree/area/feature matched edge probe.",
            "- `tile_level_form_summary.csv`: fixed-grid local summaries for map-oriented diagnostics.",
            "- `urban_form_factor_summary.csv`: city-level diagnostic flags.",
            "- `failures.csv`: per-city errors, if any.",
        ],
    )
    return {
        "out_dir": str(out_dir),
        "cities": cities,
        "n_failures": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Urban-form factor experiments.")
    parser.add_argument("--cities", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_edges", type=int, default=30000)
    parser.add_argument("--urban_root", type=str, default="/mnt/e/urban network datasets")
    parser.add_argument(
        "--base_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data",
    )
    parser.add_argument("--tile_size_m", type=float, default=1000.0)
    parser.add_argument("--out_dir", type=str, default="results/urban_factor_experiments/03_urban_form")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
