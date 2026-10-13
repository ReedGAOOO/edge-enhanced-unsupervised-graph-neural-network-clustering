#!/usr/bin/env python3
"""Export sampled Urban V4 graph edges for R map visualization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = [
    "urban_washingtondc_plot_v4slj",
    "urban_washingtondc_plot_v4sljb",
    "urban_washingtondc_plot_v4g",
    "urban_tokyo_plot_v4sljb",
    "urban_paris_plot_v4slj",
]


RELATION_COLUMNS = {
    "Street": ("is_street_backed", "shared_street_cnt"),
    "Junction": ("is_junction_only", "shared_junction_cnt"),
    "Building": ("is_shared_building", "is_building_neighbor", "building_neighbor_cnt"),
    "Geometry": ("is_geom_fallback",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--out-dir",
        default=Path("results/data_construction_series/urban_v4_semantic_search/map_layers"),
        type=Path,
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--max-per-relation", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def relation_indices(edge_feature_names: list[str]) -> dict[str, list[int]]:
    name_to_idx = {name: i for i, name in enumerate(edge_feature_names)}
    out: dict[str, list[int]] = {}
    for rel, names in RELATION_COLUMNS.items():
        out[rel] = [name_to_idx[name] for name in names if name in name_to_idx]
    return out


def load_dataset(data_root: Path, dataset: str):
    ds_dir = data_root / dataset
    with (ds_dir / f"{dataset}_meta.json").open() as f:
        meta = json.load(f)
    edge_index = np.load(ds_dir / f"{dataset}_edge_index.npy")
    edge_attr = np.load(ds_dir / f"{dataset}_edge_attr.npy")
    labels = np.load(ds_dir / f"{dataset}_label.npy")
    if edge_index.shape[0] == 2:
        edge_index = edge_index.T
    return meta, edge_index, edge_attr, labels


def undirected_mask(edge_index: np.ndarray) -> np.ndarray:
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    return src < dst


def edge_relations(edge_attr: np.ndarray, idx_by_relation: dict[str, list[int]]) -> dict[str, np.ndarray]:
    out = {}
    for rel, idxs in idx_by_relation.items():
        if not idxs:
            out[rel] = np.zeros(edge_attr.shape[0], dtype=bool)
            continue
        rel_values = edge_attr[:, idxs]
        out[rel] = np.nan_to_num(rel_values, nan=0.0).max(axis=1) > 0
    return out


def sample_relation_rows(
    dataset: str,
    meta: dict,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    labels: np.ndarray,
    max_per_relation: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    feature_names = meta.get("edge_feature_names", [])
    idx_by_relation = relation_indices(feature_names)

    mask = undirected_mask(edge_index)
    edge_index = edge_index[mask]
    edge_attr = edge_attr[mask]
    relations = edge_relations(edge_attr, idx_by_relation)

    rows: list[dict] = []
    manifest: list[dict] = []
    for relation, rel_mask in relations.items():
        candidates = np.flatnonzero(rel_mask)
        n_total = int(candidates.size)
        if n_total == 0:
            manifest.append(
                {
                    "dataset": dataset,
                    "city": meta.get("city", ""),
                    "variant": meta.get("variant", ""),
                    "relation": relation,
                    "n_edges_total": 0,
                    "n_edges_exported": 0,
                }
            )
            continue

        if n_total > max_per_relation:
            candidates = rng.choice(candidates, size=max_per_relation, replace=False)
        candidates = np.sort(candidates)

        for idx in candidates:
            src = int(edge_index[idx, 0])
            dst = int(edge_index[idx, 1])
            y0 = int(labels[src])
            y1 = int(labels[dst])
            known_pair = y0 >= 0 and y1 >= 0 and y0 != 8 and y1 != 8
            rows.append(
                {
                    "dataset": dataset,
                    "city": meta.get("city", ""),
                    "variant": meta.get("variant", ""),
                    "relation": relation,
                    "src": src,
                    "dst": dst,
                    "src_label": y0,
                    "dst_label": y1,
                    "known_pair": int(known_pair),
                    "same_label": int(known_pair and y0 == y1),
                }
            )

        manifest.append(
            {
                "dataset": dataset,
                "city": meta.get("city", ""),
                "variant": meta.get("variant", ""),
                "relation": relation,
                "n_edges_total": n_total,
                "n_edges_exported": int(candidates.size),
            }
        )

    return rows, manifest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    manifest: list[dict] = []

    for dataset in args.datasets:
        meta, edge_index, edge_attr, labels = load_dataset(args.data_root, dataset)
        rows, ds_manifest = sample_relation_rows(
            dataset=dataset,
            meta=meta,
            edge_index=edge_index,
            edge_attr=edge_attr,
            labels=labels,
            max_per_relation=args.max_per_relation,
            rng=rng,
        )
        all_rows.extend(rows)
        manifest.extend(ds_manifest)

    write_csv(args.out_dir / "urban_v4_map_edge_samples.csv", all_rows)
    write_csv(args.out_dir / "urban_v4_map_edge_sample_manifest.csv", manifest)
    print(f"Wrote {len(all_rows)} sampled relation-edge rows to {args.out_dir}")


if __name__ == "__main__":
    main()
