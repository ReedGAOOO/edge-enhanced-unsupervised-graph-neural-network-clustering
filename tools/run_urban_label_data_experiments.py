#!/usr/bin/env python3
"""Run Urban label/data factor controls on existing V3b plot graphs."""

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
    coarse_labels,
    ensure_dir,
    known_mask,
    load_sparse_graph,
    parse_cities,
    parse_float_csv,
    parse_int_csv,
    prepare_known_edges,
    run_edge_probe_arrays,
    sample_edge_indices,
    write_readme,
)


def _known_downsample_mask(base_known: np.ndarray, keep: float, seed: int) -> np.ndarray:
    if keep >= 0.999:
        return base_known.copy()
    rng = np.random.default_rng(int(seed) + int(round(float(keep) * 1000)) * 997)
    known_idx = np.flatnonzero(base_known)
    if known_idx.size == 0:
        return base_known.copy()
    n_keep = max(1, int(round(float(keep) * known_idx.size)))
    chosen = rng.choice(known_idx, size=min(n_keep, known_idx.size), replace=False)
    out = np.zeros_like(base_known, dtype=bool)
    out[chosen] = True
    return out


def _run_probe(
    graph,
    *,
    labels: np.ndarray,
    known: np.ndarray,
    seed: int,
    max_edges: int,
    sample_mode: str,
) -> list[dict[str, Any]]:
    src, dst, edge_attr, target = prepare_known_edges(graph, labels=labels, known=known)
    labels_i = labels[src] if src.shape[0] else None
    labels_j = labels[dst] if dst.shape[0] else None
    idx = sample_edge_indices(
        target,
        seed=seed,
        max_edges=max_edges,
        mode=sample_mode,
        labels_i=labels_i,
        labels_j=labels_j,
    )
    return run_edge_probe_arrays(
        graph=graph,
        src=src[idx],
        dst=dst[idx],
        edge_attr=edge_attr[idx],
        target=target[idx],
        seed=seed,
    )


def _feature_rows(df: pd.DataFrame, feature_set: str = "endpoint_edge_attr") -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["feature_set"] == feature_set].copy()


def _safe_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _build_summary(
    down_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    granularity_df: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    cities = sorted(
        set(down_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(balanced_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | set(granularity_df.get("city", pd.Series(dtype=str)).dropna().astype(str))
        | {str(x.get("city", "")) for x in failures if x.get("city")}
    )
    rows = []
    for city in cities:
        dcity = down_df[(down_df["city"] == city) & (down_df["feature_set"] == "endpoint_edge_attr")]
        full = dcity[np.isclose(pd.to_numeric(dcity["known_label_keep"], errors="coerce"), 1.0)]
        full_gain = _safe_mean(full["auc_gain_vs_endpoint"]) if not full.empty else float("nan")
        full_drop = _safe_mean(full["auc_drop_vs_perm"]) if not full.empty else float("nan")
        known_ratio = _safe_mean(full["known_label_keep_real"]) if "known_label_keep_real" in full else float("nan")

        partial = dcity[~np.isclose(pd.to_numeric(dcity["known_label_keep"], errors="coerce"), 1.0)]
        gains = pd.to_numeric(partial["auc_gain_vs_endpoint"], errors="coerce").dropna()
        gain_std = float(gains.std()) if len(gains) > 1 else float("nan")
        gain_range = float(gains.max() - gains.min()) if len(gains) else float("nan")
        label_unstable = bool((np.isfinite(gain_std) and gain_std > 0.003) or (np.isfinite(gain_range) and gain_range > 0.010))

        bcity = balanced_df[(balanced_df["city"] == city) & (balanced_df["feature_set"] == "endpoint_edge_attr")]
        target_bal = bcity[bcity["balance_mode"] == "target_balanced"]
        pair_bal = bcity[bcity["balance_mode"] == "label_pair_balanced"]
        target_gain = _safe_mean(target_bal["auc_gain_vs_endpoint"]) if not target_bal.empty else float("nan")
        pair_gain = _safe_mean(pair_bal["auc_gain_vs_endpoint"]) if not pair_bal.empty else float("nan")
        class_delta = target_gain - full_gain if np.isfinite(target_gain) and np.isfinite(full_gain) else float("nan")
        class_reverse = (
            np.isfinite(target_gain)
            and np.isfinite(full_gain)
            and full_gain != 0
            and np.sign(target_gain) != np.sign(full_gain)
        )
        class_composition_sensitive = bool(class_reverse or (np.isfinite(class_delta) and abs(class_delta) > 0.005))

        gcity = granularity_df[(granularity_df["city"] == city) & (granularity_df["feature_set"] == "endpoint_edge_attr")]
        coarse = gcity[gcity["label_mode"] == "coarse_4plus_other"]
        coarse_gain = _safe_mean(coarse["auc_gain_vs_endpoint"]) if not coarse.empty else float("nan")
        coarse_drop = _safe_mean(coarse["auc_drop_vs_perm"]) if not coarse.empty else float("nan")
        coarse_delta = coarse_gain - full_gain if np.isfinite(coarse_gain) and np.isfinite(full_gain) else float("nan")
        label_task_mismatch_possible = bool(
            np.isfinite(coarse_delta)
            and coarse_delta > 0.003
            and (not np.isfinite(coarse_drop) or coarse_drop > -0.001)
        )

        rows.append(
            {
                "city": city,
                "city_label": city_label(city),
                "status": "failed" if city in {str(x.get("city")) for x in failures} else "ok",
                "full_known_edge_gain_auc": full_gain,
                "full_edge_attr_perm_drop_auc": full_drop,
                "known_label_keep_real_at_full": known_ratio,
                "downsample_edge_gain_std": gain_std,
                "downsample_edge_gain_range": gain_range,
                "target_balanced_edge_gain_auc": target_gain,
                "label_pair_balanced_edge_gain_auc": pair_gain,
                "class_balanced_minus_full_gain": class_delta,
                "coarse_label_edge_gain_auc": coarse_gain,
                "coarse_minus_original_gain": coarse_delta,
                "coarse_edge_attr_perm_drop_auc": coarse_drop,
                "label_unstable": label_unstable,
                "class_composition_sensitive": class_composition_sensitive,
                "label_task_mismatch_possible": label_task_mismatch_possible,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    base_graph_root = Path(args.base_graph_root)
    cities = parse_cities(args.cities, graph_root=base_graph_root)
    seeds = parse_int_csv(args.seeds)
    keeps = parse_float_csv(args.keeps)

    down_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    granularity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for city in cities:
        dataset = city_to_dataset(city, "v3bsjg")
        try:
            graph = load_sparse_graph(base_graph_root, dataset)
            original_labels = graph.labels.astype(np.int64, copy=False)
            original_known = known_mask(original_labels, graph.meta)
            known_total = int(original_known.sum())

            for seed in seeds:
                for keep in keeps:
                    keep_mask = _known_downsample_mask(original_known, keep=keep, seed=seed)
                    rows = _run_probe(
                        graph,
                        labels=original_labels,
                        known=keep_mask,
                        seed=seed,
                        max_edges=int(args.max_edges),
                        sample_mode="random",
                    )
                    for row in rows:
                        row.update(
                            {
                                "city": city,
                                "city_label": city_label(city),
                                "dataset": dataset,
                                "experiment": "label_downsample",
                                "label_mode": "original_top1",
                                "known_label_keep": float(keep),
                                "known_label_keep_real": float(keep_mask.sum() / known_total) if known_total else np.nan,
                                "sample_mode": "random",
                            }
                        )
                    down_rows.extend(rows)

                for mode in ("target_balanced", "label_pair_balanced"):
                    rows = _run_probe(
                        graph,
                        labels=original_labels,
                        known=original_known,
                        seed=seed,
                        max_edges=int(args.max_edges),
                        sample_mode=mode,
                    )
                    for row in rows:
                        row.update(
                            {
                                "city": city,
                                "city_label": city_label(city),
                                "dataset": dataset,
                                "experiment": "class_balanced",
                                "label_mode": "original_top1",
                                "balance_mode": mode,
                                "sample_mode": mode,
                            }
                        )
                    balanced_rows.extend(rows)

                coarse_y, coarse_mapping = coarse_labels(original_labels, graph.meta)
                coarse_known = coarse_y != int(coarse_mapping["unknown"])
                rows = _run_probe(
                    graph,
                    labels=coarse_y,
                    known=coarse_known,
                    seed=seed,
                    max_edges=int(args.max_edges),
                    sample_mode="random",
                )
                for row in rows:
                    row.update(
                        {
                            "city": city,
                            "city_label": city_label(city),
                            "dataset": dataset,
                            "experiment": "label_granularity",
                            "label_mode": "coarse_4plus_other",
                            "sample_mode": "random",
                            "coarse_label_mapping_json": json.dumps(coarse_mapping, sort_keys=True),
                        }
                    )
                granularity_rows.extend(rows)

                full_rows = _run_probe(
                    graph,
                    labels=original_labels,
                    known=original_known,
                    seed=seed,
                    max_edges=int(args.max_edges),
                    sample_mode="random",
                )
                for row in full_rows:
                    row.update(
                        {
                            "city": city,
                            "city_label": city_label(city),
                            "dataset": dataset,
                            "experiment": "label_granularity",
                            "label_mode": "original_top1",
                            "sample_mode": "random",
                            "coarse_label_mapping_json": "",
                        }
                    )
                granularity_rows.extend(full_rows)
        except Exception as exc:
            failures.append(
                {
                    "city": city,
                    "dataset": dataset,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    down_df = pd.DataFrame(down_rows)
    balanced_df = pd.DataFrame(balanced_rows)
    granularity_df = pd.DataFrame(granularity_rows)

    if not down_df.empty:
        down_df = add_probe_deltas(
            down_df,
            ["city", "dataset", "experiment", "known_label_keep", "sample_mode", "label_mode"],
        )
    if not balanced_df.empty:
        balanced_df = add_probe_deltas(
            balanced_df,
            ["city", "dataset", "experiment", "balance_mode", "sample_mode", "label_mode"],
        )
    if not granularity_df.empty:
        granularity_df = add_probe_deltas(
            granularity_df,
            ["city", "dataset", "experiment", "label_mode", "sample_mode"],
        )

    summary_df = _build_summary(down_df, balanced_df, granularity_df, failures)

    down_df.to_csv(out_dir / "label_downsample_runs.csv", index=False)
    balanced_df.to_csv(out_dir / "class_balanced_probe.csv", index=False)
    granularity_df.to_csv(out_dir / "label_granularity_probe.csv", index=False)
    summary_df.to_csv(out_dir / "label_factor_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    write_readme(
        out_dir / "README.md",
        "Urban Label/Data Factor Experiments",
        [
            "This package tests whether apparent edge signal is stable under label coverage, class balance, and label granularity controls.",
            "",
            "Outputs:",
            "- `label_downsample_runs.csv`: edge probe under known-label downsampling.",
            "- `class_balanced_probe.csv`: edge probe after same/different and label-pair balancing.",
            "- `label_granularity_probe.csv`: original top-1 label versus coarse land-use mapping.",
            "- `label_factor_summary.csv`: city-level diagnostic flags.",
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
    parser = argparse.ArgumentParser(description="Run Urban label/data factor experiments.")
    parser.add_argument("--cities", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_edges", type=int, default=30000)
    parser.add_argument(
        "--base_graph_root",
        type=str,
        default="/mnt/e/wsl-project-data/edge-enhanced-unsupervised-graph-neural-network-clustering/data",
    )
    parser.add_argument("--keeps", type=str, default="0.10,0.20,0.40,0.80,1.00")
    parser.add_argument("--out_dir", type=str, default="results/urban_factor_experiments/01_label_data")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
