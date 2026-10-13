#!/usr/bin/env python3
"""Run Urban graph projection factor controls and V4 construction sweeps."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urban_factor_experiment_utils import (
    add_probe_deltas,
    city_label,
    city_to_dataset,
    ensure_dir,
    load_sparse_graph,
    parse_cities,
    parse_int_csv,
    prepare_known_edges,
    run_command,
    run_edge_probe_arrays,
    sample_edge_indices,
    write_readme,
)


PYTHON_BIN = "/home/aitx/miniconda3/envs/gnn/bin/python"
DROP_RELATIONS = ["street", "junction", "geom", "building"]
ZERO_RELATIONS = "street,junction,geom,building"


def _probe_dataset(
    root: Path,
    dataset: str,
    *,
    seeds: list[int],
    max_edges: int,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    graph = load_sparse_graph(root, dataset)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        src, dst, edge_attr, target = prepare_known_edges(graph)
        idx = sample_edge_indices(target, seed=seed, max_edges=max_edges, mode="random")
        probe_rows = run_edge_probe_arrays(
            graph=graph,
            src=src[idx],
            dst=dst[idx],
            edge_attr=edge_attr[idx],
            target=target[idx],
            seed=seed,
        )
        for row in probe_rows:
            row.update(
                {
                    "city": graph.city,
                    "city_label": city_label(graph.city),
                    "dataset": dataset,
                    **extra,
                }
            )
        rows.extend(probe_rows)
    return rows


def _prepare_intervention(
    *,
    repo_root: Path,
    source_root: Path,
    out_root: Path,
    dataset: str,
    out_name: str,
    mode: str,
    features: str,
    seed: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir = out_root / out_name
    if out_dir.exists() and not force:
        return {
            "cmd": "skip existing intervention",
            "rc": 0,
            "elapsed_sec": 0.0,
            "stdout": "",
            "stderr": "",
            "dry_run": dry_run,
            "skipped_existing": True,
        }
    cmd = [
        PYTHON_BIN,
        "tools/prepare_edge_intervention_dataset.py",
        "--dataset",
        dataset,
        "--root_path",
        str(source_root),
        "--out_root",
        str(out_root),
        "--out_name",
        out_name,
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]
    if features:
        cmd.extend(["--features", features])
    if force:
        cmd.append("--force")
    return run_command(cmd, cwd=repo_root, dry_run=dry_run)


def _prepare_v4(
    *,
    repo_root: Path,
    urban_root: Path,
    out_root: Path,
    city: str,
    dataset_name: str,
    variant: str,
    street_topk: int,
    junction_topk: int,
    building_knn: int,
    building_max_dist_m: float,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir = out_root / dataset_name
    if out_dir.exists() and not force:
        return {
            "cmd": "skip existing v4 construction",
            "rc": 0,
            "elapsed_sec": 0.0,
            "stdout": "",
            "stderr": "",
            "dry_run": dry_run,
            "skipped_existing": True,
        }
    cmd = [
        PYTHON_BIN,
        "tools/prepare_urban_plot_graph_v4_semantic.py",
        "--city",
        city,
        "--urban_root",
        str(urban_root),
        "--out_root",
        str(out_root),
        "--dataset_name",
        dataset_name,
        "--variant",
        variant,
        "--street_topk",
        str(street_topk),
        "--junction_topk",
        str(junction_topk),
        "--building_knn",
        str(building_knn),
        "--building_max_dist_m",
        str(building_max_dist_m),
    ]
    if force:
        cmd.append("--force")
    return run_command(cmd, cwd=repo_root, dry_run=dry_run)


def _mean_gain(df: pd.DataFrame, **filters: Any) -> float:
    if df.empty:
        return float("nan")
    cur = df.copy()
    for key, value in filters.items():
        cur = cur[cur[key] == value]
    cur = cur[cur["feature_set"] == "endpoint_edge_attr"]
    vals = pd.to_numeric(cur["auc_gain_vs_endpoint"], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _mean_perm_drop(df: pd.DataFrame, **filters: Any) -> float:
    if df.empty:
        return float("nan")
    cur = df.copy()
    for key, value in filters.items():
        cur = cur[cur[key] == value]
    cur = cur[cur["feature_set"] == "endpoint_edge_attr"]
    vals = pd.to_numeric(cur["auc_drop_vs_perm"], errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _build_summary(
    relation_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    v4_df: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    cities = sorted(
        set(relation_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(permutation_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(v4_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | {str(x.get("city", "")) for x in failures if x.get("city")}
    )
    failed_cities = {str(x.get("city")) for x in failures if x.get("city")}
    rows = []
    for city in cities:
        orig_gain = _mean_gain(relation_df, city=city, projection_condition="original")
        orig_perm_drop = _mean_perm_drop(relation_df, city=city, projection_condition="original")

        p_gain = _mean_gain(permutation_df, city=city, projection_condition="permute_edge_attr")
        p_gap = orig_gain - p_gain if np.isfinite(orig_gain) and np.isfinite(p_gain) else float("nan")
        edge_attr_not_used = bool(
            (np.isfinite(orig_perm_drop) and abs(orig_perm_drop) < 0.002)
            or (np.isfinite(p_gap) and abs(p_gap) < 0.002)
        )

        deltas = []
        for condition in sorted(set(relation_df.get("projection_condition", []))):
            if condition == "original":
                continue
            gain = _mean_gain(relation_df, city=city, projection_condition=condition)
            if np.isfinite(gain) and np.isfinite(orig_gain):
                deltas.append(abs(gain - orig_gain))
        max_relation_delta = float(max(deltas)) if deltas else float("nan")
        projection_rule_sensitive = bool(np.isfinite(max_relation_delta) and max_relation_delta > 0.005)

        v4g_gain = _mean_gain(v4_df, city=city, construction_condition="v4g")
        semantic_gains = []
        for condition in sorted(set(v4_df.get("construction_condition", []))):
            if condition == "v4g":
                continue
            gain = _mean_gain(v4_df, city=city, construction_condition=condition)
            if np.isfinite(gain):
                semantic_gains.append(gain)
        best_semantic = float(max(semantic_gains)) if semantic_gains else float("nan")
        spatial_geometry_confound = bool(
            np.isfinite(v4g_gain)
            and np.isfinite(best_semantic)
            and v4g_gain >= best_semantic - 0.001
        )

        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "status": "failed" if city in failed_cities else "ok",
                "v3b_original_edge_gain_auc": orig_gain,
                "v3b_original_edge_attr_perm_drop_auc": orig_perm_drop,
                "permuted_dataset_edge_gain_auc": p_gain,
                "original_minus_permuted_dataset_gain": p_gap,
                "max_relation_condition_gain_delta": max_relation_delta,
                "v4g_edge_gain_auc": v4g_gain,
                "best_semantic_v4_edge_gain_auc": best_semantic,
                "edge_attr_not_used": edge_attr_not_used,
                "projection_rule_sensitive": projection_rule_sensitive,
                "spatial_geometry_confound": spatial_geometry_confound,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    repo_root = Path(args.repo_root).resolve()
    base_graph_root = Path(args.base_graph_root)
    derived_root = Path(args.derived_graph_root)
    urban_root = Path(args.urban_root)
    ensure_dir(derived_root)

    cities = parse_cities(args.cities, graph_root=base_graph_root)
    seeds = parse_int_csv(args.seeds)
    dry_run = str(args.dry_run).strip().lower() in {"1", "true", "yes", "y"}
    force = bool(args.force)

    relation_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    v4_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for city in cities:
        source_dataset = city_to_dataset(city, "v3bsjg")
        try:
            if not dry_run:
                relation_rows.extend(
                    _probe_dataset(
                        base_graph_root,
                        source_dataset,
                        seeds=seeds,
                        max_edges=int(args.max_edges),
                        extra={
                            "source_dataset": source_dataset,
                            "derived_dataset": source_dataset,
                            "projection_condition": "original",
                            "projection_family": "v3b_intervention",
                        },
                    )
                )
                permutation_rows.extend(
                    _probe_dataset(
                        base_graph_root,
                        source_dataset,
                        seeds=seeds,
                        max_edges=int(args.max_edges),
                        extra={
                            "source_dataset": source_dataset,
                            "derived_dataset": source_dataset,
                            "projection_condition": "original",
                            "projection_family": "v3b_permutation",
                        },
                    )
                )

            for relation in DROP_RELATIONS:
                out_name = f"{source_dataset}_drop_{relation}"
                res = _prepare_intervention(
                    repo_root=repo_root,
                    source_root=base_graph_root,
                    out_root=derived_root,
                    dataset=source_dataset,
                    out_name=out_name,
                    mode="drop_relation",
                    features=relation,
                    seed=0,
                    force=force,
                    dry_run=dry_run,
                )
                res.update({"city": city, "stage": "prepare_intervention", "dataset": out_name})
                command_rows.append(res)
                if res["rc"] != 0:
                    raise RuntimeError(f"drop_relation {relation} failed: {res['stderr']}")
                if not dry_run:
                    relation_rows.extend(
                        _probe_dataset(
                            derived_root,
                            out_name,
                            seeds=seeds,
                            max_edges=int(args.max_edges),
                            extra={
                                "source_dataset": source_dataset,
                                "derived_dataset": out_name,
                                "projection_condition": f"drop_{relation}",
                                "projection_family": "v3b_intervention",
                            },
                        )
                    )

            zero_name = f"{source_dataset}_zero_relation_bits"
            res = _prepare_intervention(
                repo_root=repo_root,
                source_root=base_graph_root,
                out_root=derived_root,
                dataset=source_dataset,
                out_name=zero_name,
                mode="zero_edge_attr",
                features=ZERO_RELATIONS,
                seed=0,
                force=force,
                dry_run=dry_run,
            )
            res.update({"city": city, "stage": "prepare_intervention", "dataset": zero_name})
            command_rows.append(res)
            if res["rc"] != 0:
                raise RuntimeError(f"zero relation bits failed: {res['stderr']}")
            if not dry_run:
                relation_rows.extend(
                    _probe_dataset(
                        derived_root,
                        zero_name,
                        seeds=seeds,
                        max_edges=int(args.max_edges),
                        extra={
                            "source_dataset": source_dataset,
                            "derived_dataset": zero_name,
                            "projection_condition": "zero_relation_bits",
                            "projection_family": "v3b_intervention",
                        },
                    )
                )

            for seed in seeds:
                perm_name = f"{source_dataset}_permute_edge_attr_s{seed}"
                res = _prepare_intervention(
                    repo_root=repo_root,
                    source_root=base_graph_root,
                    out_root=derived_root,
                    dataset=source_dataset,
                    out_name=perm_name,
                    mode="permute_edge_attr",
                    features="",
                    seed=seed,
                    force=force,
                    dry_run=dry_run,
                )
                res.update({"city": city, "stage": "prepare_intervention", "dataset": perm_name})
                command_rows.append(res)
                if res["rc"] != 0:
                    raise RuntimeError(f"permute edge attr seed {seed} failed: {res['stderr']}")
                if not dry_run:
                    permutation_rows.extend(
                        _probe_dataset(
                            derived_root,
                            perm_name,
                            seeds=[seed],
                            max_edges=int(args.max_edges),
                            extra={
                                "source_dataset": source_dataset,
                                "derived_dataset": perm_name,
                                "projection_condition": "permute_edge_attr",
                                "projection_family": "v3b_permutation",
                            },
                        )
                    )

            v4_specs = [
                {"condition": "v4sl", "variant": "v4sl", "street_topk": 4, "junction_topk": 6, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4slj", "variant": "v4slj", "street_topk": 4, "junction_topk": 6, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4sljb", "variant": "v4sljb", "street_topk": 4, "junction_topk": 6, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4g", "variant": "v4g", "street_topk": 4, "junction_topk": 6, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4sljb_s2_j4", "variant": "v4sljb", "street_topk": 2, "junction_topk": 4, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4sljb_s2_j6", "variant": "v4sljb", "street_topk": 2, "junction_topk": 6, "building_knn": 1, "building_max_dist_m": 50.0},
                {"condition": "v4sljb_s4_j4", "variant": "v4sljb", "street_topk": 4, "junction_topk": 4, "building_knn": 1, "building_max_dist_m": 50.0},
            ]
            for spec in v4_specs:
                v4_name = f"urban_{city}_plot_v4factor_{spec['condition']}"
                res = _prepare_v4(
                    repo_root=repo_root,
                    urban_root=urban_root,
                    out_root=derived_root,
                    city=city,
                    dataset_name=v4_name,
                    variant=spec["variant"],
                    street_topk=int(spec["street_topk"]),
                    junction_topk=int(spec["junction_topk"]),
                    building_knn=int(spec["building_knn"]),
                    building_max_dist_m=float(spec["building_max_dist_m"]),
                    force=force,
                    dry_run=dry_run,
                )
                res.update({"city": city, "stage": "prepare_v4", "dataset": v4_name})
                command_rows.append(res)
                status = "ok" if res["rc"] == 0 else "failed"
                manifest_rows.append(
                    {
                        "city": city,
                        "city_label": city_label(city),
                        "dataset": v4_name,
                        "construction_condition": spec["condition"],
                        "variant": spec["variant"],
                        "street_topk": int(spec["street_topk"]),
                        "junction_topk": int(spec["junction_topk"]),
                        "building_knn": int(spec["building_knn"]),
                        "building_max_dist_m": float(spec["building_max_dist_m"]),
                        "root": str(derived_root),
                        "status": status,
                        "skipped_existing": bool(res.get("skipped_existing", False)),
                    }
                )
                if res["rc"] != 0:
                    raise RuntimeError(f"V4 construction {spec['condition']} failed: {res['stderr']}")
                if not dry_run:
                    v4_rows.extend(
                        _probe_dataset(
                            derived_root,
                            v4_name,
                            seeds=seeds,
                            max_edges=int(args.max_edges),
                            extra={
                                "source_dataset": source_dataset,
                                "derived_dataset": v4_name,
                                "construction_condition": spec["condition"],
                                "projection_family": "v4_construction_sweep",
                                "variant": spec["variant"],
                                "street_topk": int(spec["street_topk"]),
                                "junction_topk": int(spec["junction_topk"]),
                                "building_knn": int(spec["building_knn"]),
                                "building_max_dist_m": float(spec["building_max_dist_m"]),
                            },
                        )
                    )
        except Exception as exc:
            failures.append(
                {
                    "city": city,
                    "dataset": source_dataset,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    relation_df = pd.DataFrame(relation_rows)
    permutation_df = pd.DataFrame(permutation_rows)
    v4_df = pd.DataFrame(v4_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    commands_df = pd.DataFrame(command_rows)

    if not relation_df.empty:
        relation_df = add_probe_deltas(
            relation_df,
            ["city", "source_dataset", "derived_dataset", "projection_condition", "projection_family"],
        )
    if not permutation_df.empty:
        permutation_df = add_probe_deltas(
            permutation_df,
            ["city", "source_dataset", "derived_dataset", "projection_condition", "projection_family"],
        )
    if not v4_df.empty:
        v4_df = add_probe_deltas(
            v4_df,
            [
                "city",
                "source_dataset",
                "derived_dataset",
                "construction_condition",
                "projection_family",
                "variant",
                "street_topk",
                "junction_topk",
                "building_knn",
                "building_max_dist_m",
            ],
        )

    summary_df = _build_summary(relation_df, permutation_df, v4_df, failures)

    relation_df.to_csv(out_dir / "relation_ablation_probe.csv", index=False)
    permutation_df.to_csv(out_dir / "edge_attr_permutation_probe.csv", index=False)
    manifest_df.to_csv(out_dir / "v4_construction_sweep_manifest.csv", index=False)
    v4_df.to_csv(out_dir / "v4_construction_sweep_probe.csv", index=False)
    summary_df.to_csv(out_dir / "projection_factor_summary.csv", index=False)
    commands_df.to_csv(out_dir / "command_log.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    write_readme(
        out_dir / "README.md",
        "Urban Graph Projection Factor Experiments",
        [
            "This package tests whether Urban edge signal depends on the projection rule and whether edge attributes survive permutation controls.",
            "",
            "Outputs:",
            "- `relation_ablation_probe.csv`: original V3b plus drop/zero relation interventions.",
            "- `edge_attr_permutation_probe.csv`: original V3b versus derived edge-attribute permutation datasets.",
            "- `v4_construction_sweep_manifest.csv`: V4 semantic construction datasets written outside repo `data/`.",
            "- V4 building-neighbour controls use `building_knn=1` and `building_max_dist_m=50` to keep all-city CPU diagnostics feasible.",
            "- `v4_construction_sweep_probe.csv`: edge probe over the V4 construction sweep.",
            "- `projection_factor_summary.csv`: city-level diagnostic flags.",
            "- `command_log.csv`: CPU command execution details.",
            "- `failures.csv`: per-city errors, if any.",
        ],
    )
    return {
        "out_dir": str(out_dir),
        "derived_graph_root": str(derived_root),
        "cities": cities,
        "n_failures": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Urban graph projection factor experiments.")
    parser.add_argument("--cities", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_edges", type=int, default=30000)
    parser.add_argument("--urban_root", type=str, default="/mnt/e/urban network datasets")
    parser.add_argument(
        "--base_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data",
    )
    parser.add_argument(
        "--derived_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data/urban_factor_experiments",
    )
    parser.add_argument("--out_dir", type=str, default="results/urban_factor_experiments/02_graph_projection")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", type=str, default="false")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
