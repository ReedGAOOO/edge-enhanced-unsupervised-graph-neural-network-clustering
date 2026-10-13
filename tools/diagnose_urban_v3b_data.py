#!/usr/bin/env python3
"""
Build a diagnostics table for Urban V3b homogeneous plot graphs.

The table is meant to answer whether the Urban data is suitable for edge
semantics experiments before running expensive GNN training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


RELATION_FEATURES = {
    "street": "is_street_backed",
    "junction": "is_junction_only",
    "geom": "is_geom_fallback",
    "shared_building": "is_shared_building",
}


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_name(path: Path) -> str:
    return path.name


def _discover(root: Path, pattern: str) -> list[str]:
    return sorted(p.name for p in root.glob(pattern) if p.is_dir())


def _label_names(meta: dict[str, Any]) -> dict[int, str]:
    mapping = meta.get("label_mapping", {})
    out: dict[int, str] = {}
    if isinstance(mapping, dict):
        for name, value in mapping.items():
            try:
                out[int(value)] = str(name)
            except Exception:
                continue
    return out


def _unknown_label(meta: dict[str, Any]) -> int | None:
    mapping = meta.get("label_mapping", {})
    if isinstance(mapping, dict) and "unknown" in mapping:
        try:
            return int(mapping["unknown"])
        except Exception:
            return None
    return None


def _class_distribution(labels: np.ndarray, known_mask: np.ndarray, label_names: dict[int, str]) -> tuple[str, str]:
    known = labels[known_mask]
    if known.size == 0:
        return "{}", "{}"
    vals, counts = np.unique(known, return_counts=True)
    total = float(known.size)
    counts_obj = {label_names.get(int(v), str(int(v))): int(c) for v, c in zip(vals, counts)}
    frac_obj = {label_names.get(int(v), str(int(v))): float(c / total) for v, c in zip(vals, counts)}
    return json.dumps(counts_obj, sort_keys=True), json.dumps(frac_obj, sort_keys=True)


def _entropy_from_counts(counts: np.ndarray) -> float:
    if counts.size == 0 or counts.sum() <= 0:
        return float("nan")
    p = counts.astype(np.float64) / float(counts.sum())
    return float(-(p * np.log(p + 1e-12)).sum() / max(np.log(len(p)), 1e-12))


def _connectivity(num_nodes: int, edge_index: np.ndarray) -> dict[str, float]:
    if edge_index.size == 0:
        return {
            "n_components": float(num_nodes),
            "largest_component_ratio": 0.0,
            "isolated_node_ratio": 1.0,
            "component_size_p50": 1.0,
            "component_size_p90": 1.0,
        }
    row = edge_index[0].astype(np.int64, copy=False)
    col = edge_index[1].astype(np.int64, copy=False)
    data = np.ones(row.shape[0], dtype=np.float32)
    adj = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    n_comp, comp = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(comp, minlength=n_comp)
    return {
        "n_components": float(n_comp),
        "largest_component_ratio": float(sizes.max() / max(num_nodes, 1)),
        "isolated_node_ratio": float(np.mean(sizes[comp] == 1)),
        "component_size_p50": float(np.percentile(sizes, 50)),
        "component_size_p90": float(np.percentile(sizes, 90)),
    }


def _edge_homophily(labels: np.ndarray, known_mask: np.ndarray, edge_index: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if edge_index.size == 0:
        return {"known_edge_ratio": float("nan"), "same_label_ratio": float("nan"), "known_edges": 0.0}
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    edge_mask = np.ones(src.shape[0], dtype=bool) if mask is None else mask.astype(bool, copy=False)
    both_known = edge_mask & known_mask[src] & known_mask[dst]
    known_edges = int(both_known.sum())
    total_edges = int(edge_mask.sum())
    if known_edges == 0:
        same = float("nan")
    else:
        same = float(np.mean(labels[src[both_known]] == labels[dst[both_known]]))
    return {
        "known_edge_ratio": float(known_edges / max(total_edges, 1)),
        "same_label_ratio": same,
        "known_edges": float(known_edges),
    }


def _feature_index(feature_names: list[str], feature: str) -> int | None:
    if feature in feature_names:
        return int(feature_names.index(feature))
    return None


def _diagnose_one(root: Path, dataset: str) -> dict[str, Any]:
    base = root / dataset
    meta = _load_json(base / f"{dataset}_meta.json")
    feat = np.load(base / f"{dataset}_feat.npy", mmap_mode="r")
    labels = np.load(base / f"{dataset}_label.npy", mmap_mode="r")
    edge_index = np.load(base / f"{dataset}_edge_index.npy", mmap_mode="r")
    edge_attr = np.load(base / f"{dataset}_edge_attr.npy", mmap_mode="r")
    edge_weight_path = base / f"{dataset}_edge_weight.npy"
    edge_weight = np.load(edge_weight_path, mmap_mode="r") if edge_weight_path.exists() else None

    num_nodes = int(feat.shape[0])
    num_edges_directed = int(edge_index.shape[1])
    num_edges_undirected = int(num_edges_directed // 2)
    edge_dim = int(edge_attr.shape[1])
    feature_names = list(meta.get("edge_feature_names", []))
    if len(feature_names) != edge_dim:
        feature_names = [str(i) for i in range(edge_dim)]

    unknown = _unknown_label(meta)
    if unknown is None:
        known_mask = labels >= 0
    else:
        known_mask = (labels >= 0) & (labels != int(unknown))
    known_count = int(known_mask.sum())
    label_names = _label_names(meta)
    class_counts_json, class_frac_json = _class_distribution(labels, known_mask, label_names)
    known_vals, known_counts = np.unique(labels[known_mask], return_counts=True) if known_count > 0 else (np.array([]), np.array([]))

    row: dict[str, Any] = {
        "city": str(meta.get("city", dataset.replace("urban_", "").replace("_plot_v3bsjg", ""))),
        "dataset": dataset,
        "variant": str(meta.get("variant", "")),
        "edge_schema": str(meta.get("edge_schema", "")),
        "n_nodes": num_nodes,
        "n_features": int(feat.shape[1]),
        "n_edges_directed": num_edges_directed,
        "n_edges_undirected": num_edges_undirected,
        "avg_undirected_degree": float(2.0 * num_edges_undirected / max(num_nodes, 1)),
        "edge_attr_dim": edge_dim,
        "known_label_ratio": float(known_count / max(num_nodes, 1)),
        "unknown_label": int(unknown) if unknown is not None else "",
        "n_known_classes": int(len(known_vals)),
        "known_label_entropy_norm": _entropy_from_counts(known_counts.astype(np.int64)),
        "largest_known_class_ratio": float(known_counts.max() / known_count) if known_count > 0 else float("nan"),
        "known_class_counts_json": class_counts_json,
        "known_class_frac_json": class_frac_json,
    }

    meta_rel = meta.get("relation_counts_undirected", {}) if isinstance(meta.get("relation_counts_undirected", {}), dict) else {}
    for rel in ("street", "junction", "geom"):
        count = int(meta_rel.get(rel, 0))
        row[f"{rel}_edges_undirected"] = count
        row[f"{rel}_edge_frac"] = float(count / max(num_edges_undirected, 1))

    for rel, feature in RELATION_FEATURES.items():
        idx = _feature_index(feature_names, feature)
        if idx is None:
            row[f"{rel}_bit_frac"] = float("nan")
            row[f"{rel}_known_edge_ratio"] = float("nan")
            row[f"{rel}_same_label_ratio"] = float("nan")
            continue
        rel_mask = np.asarray(edge_attr[:, idx] > 0.5)
        row[f"{rel}_bit_frac"] = float(rel_mask.mean())
        stats = _edge_homophily(labels=np.asarray(labels), known_mask=np.asarray(known_mask), edge_index=np.asarray(edge_index), mask=rel_mask)
        row[f"{rel}_known_edge_ratio"] = stats["known_edge_ratio"]
        row[f"{rel}_same_label_ratio"] = stats["same_label_ratio"]
        row[f"{rel}_known_edges"] = stats["known_edges"]

    overall_h = _edge_homophily(labels=np.asarray(labels), known_mask=np.asarray(known_mask), edge_index=np.asarray(edge_index))
    row["overall_known_edge_ratio"] = overall_h["known_edge_ratio"]
    row["overall_same_label_ratio"] = overall_h["same_label_ratio"]
    row["overall_known_edges"] = overall_h["known_edges"]

    row.update(_connectivity(num_nodes=num_nodes, edge_index=np.asarray(edge_index)))

    if edge_weight is not None:
        ew = np.asarray(edge_weight, dtype=np.float32)
        row["edge_weight_mean"] = float(np.mean(ew))
        row["edge_weight_std"] = float(np.std(ew))
        row["edge_weight_min"] = float(np.min(ew))
        row["edge_weight_max"] = float(np.max(ew))
    else:
        row["edge_weight_mean"] = float("nan")
        row["edge_weight_std"] = float("nan")
        row["edge_weight_min"] = float("nan")
        row["edge_weight_max"] = float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Urban V3b homogeneous plot datasets.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="all", help='Comma-separated names, or "all".')
    parser.add_argument("--pattern", type=str, default="urban_*_plot_v3bsjg")
    parser.add_argument("--out_dir", type=str, default="results/data_diagnostics/urban_v3bsjg")
    args = parser.parse_args()

    root = Path(args.root_path)
    if str(args.datasets).strip().lower() == "all":
        datasets = _discover(root, args.pattern)
    else:
        datasets = _parse_csv(args.datasets)
    if not datasets:
        raise ValueError("No datasets selected.")

    rows = [_diagnose_one(root, dataset) for dataset in datasets]
    df = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "urban_v3bsjg_diagnostics.csv"
    json_path = out_dir / "urban_v3bsjg_diagnostics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    readme = [
        "# Urban V3b Data Diagnostics",
        "",
        "This table is a data-readiness check before using Urban as edge-semantics evidence.",
        "",
        "Important columns:",
        "- `known_label_ratio`: fraction of non-unknown labels.",
        "- `overall_same_label_ratio`: label homophily over edges whose endpoints both have known labels.",
        "- `<relation>_same_label_ratio`: same statistic restricted to a relation bit.",
        "- `largest_component_ratio`: graph connectivity health.",
        "- relation edge fractions: whether one relation dominates the topology.",
        "",
        "A dataset is stronger evidence when it has enough known labels, high connectivity, non-degenerate relation mix, and relation-wise homophily patterns that are not random/noisy.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[ok] datasets={len(df)}")
    print(f" csv={csv_path}")
    print(f" json={json_path}")


if __name__ == "__main__":
    main()
