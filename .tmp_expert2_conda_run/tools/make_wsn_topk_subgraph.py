#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build top-k-degree induced subgraph dataset from prepared WSN dataset.")
    parser.add_argument("--src_dataset", type=str, required=True, help="Prepared dataset name under data/, e.g. bitcoin_wsn_epinion")
    parser.add_argument("--topk_nodes", type=int, required=True, help="Number of highest-degree nodes to keep")
    parser.add_argument("--out_dataset", type=str, default="", help="Output dataset name; default: {src}_top{K}")
    args = parser.parse_args()

    data_root = Path("data")
    src_name = args.src_dataset
    out_name = args.out_dataset.strip() if args.out_dataset.strip() else f"{src_name}_top{args.topk_nodes}"

    src_dir = data_root / src_name
    out_dir = data_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_index = np.load(src_dir / f"{src_name}_edge_index.npy")
    edge_weight = np.load(src_dir / f"{src_name}_edge_weight.npy")
    edge_attr = np.load(src_dir / f"{src_name}_edge_attr.npy")
    feat = np.load(src_dir / f"{src_name}_feat.npy")
    label = np.load(src_dir / f"{src_name}_label.npy")
    node_ids_path = src_dir / f"{src_name}_node_ids.npy"
    if node_ids_path.exists():
        node_ids = np.load(node_ids_path)
    else:
        # For datasets whose node IDs are already contiguous [0, N), use identity mapping.
        node_ids = np.arange(feat.shape[0], dtype=np.int64)

    n = int(feat.shape[0])
    k = int(args.topk_nodes)
    if k <= 0 or k > n:
        raise ValueError(f"topk_nodes must be in (0, {n}], got {k}")

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)

    keep_old = np.argpartition(deg, -k)[-k:]
    keep_old = np.sort(keep_old)
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_old] = True

    edge_keep = keep_mask[src] & keep_mask[dst]
    src_f = src[edge_keep]
    dst_f = dst[edge_keep]
    ew_f = edge_weight[edge_keep]
    ea_f = edge_attr[edge_keep]

    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[keep_old] = np.arange(k, dtype=np.int64)
    src_new = old_to_new[src_f]
    dst_new = old_to_new[dst_f]
    edge_index_new = np.stack([src_new, dst_new], axis=0).astype(np.int64, copy=False)

    feat_new = feat[keep_old]
    label_new = label[keep_old]
    node_ids_new = node_ids[keep_old]

    np.save(out_dir / f"{out_name}_edge_index.npy", edge_index_new)
    np.save(out_dir / f"{out_name}_edge_weight.npy", ew_f.astype(np.float32, copy=False))
    np.save(out_dir / f"{out_name}_edge_attr.npy", ea_f.astype(np.float32, copy=False))
    np.save(out_dir / f"{out_name}_feat.npy", feat_new.astype(np.float32, copy=False))
    np.save(out_dir / f"{out_name}_label.npy", label_new.astype(np.int64, copy=False))
    np.save(out_dir / f"{out_name}_node_ids.npy", node_ids_new.astype(np.int64, copy=False))

    summary = {
        "src_dataset": src_name,
        "out_dataset": out_name,
        "src_num_nodes": int(n),
        "src_num_edges": int(edge_index.shape[1]),
        "topk_nodes": int(k),
        "out_num_nodes": int(feat_new.shape[0]),
        "out_num_edges": int(edge_index_new.shape[1]),
        "edge_retention_ratio": float(edge_index_new.shape[1] / max(1, edge_index.shape[1])),
        "unknown_label_ratio": float((label_new < 0).mean()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
