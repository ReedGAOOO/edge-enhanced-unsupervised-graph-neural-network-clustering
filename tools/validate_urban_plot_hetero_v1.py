#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np


def _load_meta(dataset_dir: Path) -> dict:
    meta_fp = dataset_dir / "meta.json"
    if not meta_fp.exists():
        raise FileNotFoundError(f"meta.json not found in {dataset_dir}")
    return json.loads(meta_fp.read_text(encoding="utf-8"))


def _check_indices(edge_index: np.ndarray, num_src: int, num_dst: int, name: str) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{name}: edge_index must be [2, E], got {edge_index.shape}")
    if edge_index.shape[1] == 0:
        return
    if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= int(num_src):
        raise ValueError(f"{name}: src index out of range for num_src={num_src}")
    if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= int(num_dst):
        raise ValueError(f"{name}: dst index out of range for num_dst={num_dst}")


def _validate_reverse_pair(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> bool:
    if a.shape[1] != b.shape[1]:
        return False
    sa = {tuple(x) for x in a.T.tolist()}
    sb = {tuple(x[::-1]) for x in b.T.tolist()}
    ok = sa == sb
    if not ok:
        print(f"[warn] reverse pair mismatch: {name_a} vs {name_b}")
    return ok


def _validate_self_symmetric(edge_index: np.ndarray, name: str) -> bool:
    pairs = {tuple(x) for x in edge_index.T.tolist()}
    ok = all((j, i) in pairs for i, j in pairs)
    if not ok:
        print(f"[warn] self-symmetric relation missing reverse edges: {name}")
    return ok


def _plot_component_stats(edge_index: np.ndarray, num_nodes: int) -> Dict[str, float]:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    if edge_index.shape[1] > 0:
        g.add_edges_from(edge_index.T.tolist())
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    largest = len(comps[0]) if comps else 0
    return {
        "num_components": int(len(comps)),
        "largest_component_size": int(largest),
        "largest_component_ratio": float(largest / max(num_nodes, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plot-centric heterogeneous urban dataset v1.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    meta = _load_meta(dataset_dir)
    node_types = meta["node_types"]
    edge_types = meta["edge_types"]

    summary = {
        "dataset_name": meta["dataset_name"],
        "city": meta["city"],
        "node_counts": {},
        "edge_counts": {},
        "reverse_checks": {},
        "self_symmetric_checks": {},
        "plot_component_stats": {},
        "label_counts": meta.get("label_counts", {}),
    }

    node_sizes = {}
    for node_type, spec in node_types.items():
        feat = np.load(dataset_dir / spec["feature_file"])
        if feat.ndim != 2:
            raise ValueError(f"{node_type}: feature matrix must be [N, F], got {feat.shape}")
        node_sizes[node_type] = int(feat.shape[0])
        if int(feat.shape[0]) != int(spec["num_nodes"]):
            raise ValueError(f"{node_type}: meta num_nodes mismatch: feat={feat.shape[0]} meta={spec['num_nodes']}")
        if not np.all(np.isfinite(feat)):
            raise ValueError(f"{node_type}: feature matrix contains non-finite values")
        summary["node_counts"][node_type] = int(feat.shape[0])
        if node_type == "plot" and "label_file" in spec:
            y = np.load(dataset_dir / spec["label_file"])
            if y.ndim != 1 or y.shape[0] != feat.shape[0]:
                raise ValueError(f"{node_type}: label vector mismatch {y.shape} vs {feat.shape[0]}")

    edge_cache: Dict[str, np.ndarray] = {}
    for name, spec in edge_types.items():
        edge_index = np.load(dataset_dir / spec["edge_index_file"])
        num_src = node_sizes[spec["src_type"]]
        num_dst = node_sizes[spec["dst_type"]]
        _check_indices(edge_index, num_src, num_dst, name)
        edge_cache[name] = edge_index
        summary["edge_counts"][name] = int(edge_index.shape[1])
        edge_attr_file = spec.get("edge_attr_file")
        if edge_attr_file:
            edge_attr = np.load(dataset_dir / edge_attr_file)
            if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(f"{name}: edge_attr mismatch {edge_attr.shape} vs E={edge_index.shape[1]}")
            if not np.all(np.isfinite(edge_attr)):
                raise ValueError(f"{name}: edge_attr contains non-finite values")

    handled_pairs = set()
    for name, spec in edge_types.items():
        reverse_key = spec.get("reverse_key")
        if not reverse_key:
            continue
        pair_key = tuple(sorted([name, reverse_key]))
        if pair_key in handled_pairs:
            continue
        handled_pairs.add(pair_key)
        if name == reverse_key:
            summary["self_symmetric_checks"][name] = bool(_validate_self_symmetric(edge_cache[name], name))
        elif reverse_key in edge_cache:
            summary["reverse_checks"][name] = bool(_validate_reverse_pair(edge_cache[name], edge_cache[reverse_key], name, reverse_key))

    plot_edge_name = "plot__shares_street__plot"
    if plot_edge_name in edge_cache:
        undirected = edge_cache[plot_edge_name]
        if undirected.shape[1] > 0:
            pairs = undirected.T.tolist()
            seen = set()
            uniq = []
            for i, j in pairs:
                key = (i, j) if i <= j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(key)
            edge_index_uniq = np.asarray(uniq, dtype=np.int64).T if uniq else np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index_uniq = np.zeros((2, 0), dtype=np.int64)
        summary["plot_component_stats"] = _plot_component_stats(edge_index_uniq, node_sizes["plot"])

    summary["status"] = "ok"
    if args.out_json:
        out_fp = Path(args.out_json).resolve()
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
