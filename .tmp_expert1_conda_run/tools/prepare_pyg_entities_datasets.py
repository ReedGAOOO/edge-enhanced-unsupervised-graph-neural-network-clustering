#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Entities


ENTITY_NAMES = ["AIFB", "MUTAG", "BGS", "AM"]


def to_numpy(x: torch.Tensor, dtype=None):
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def build_node_struct_feat(edge_index: np.ndarray, edge_type: np.ndarray, num_nodes: int) -> np.ndarray:
    src = edge_index[0]
    dst = edge_index[1]
    rel = edge_type.astype(np.float32, copy=False)
    rel_norm = rel / max(1.0, float(rel.max()))

    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)

    out_rel_sum = np.bincount(src, weights=rel_norm, minlength=num_nodes).astype(np.float32)
    in_rel_sum = np.bincount(dst, weights=rel_norm, minlength=num_nodes).astype(np.float32)

    eps = 1e-6
    out_rel_mean = out_rel_sum / (out_deg + eps)
    in_rel_mean = in_rel_sum / (in_deg + eps)

    feat = np.stack(
        [
            np.log1p(out_deg),
            np.log1p(in_deg),
            np.log1p(out_deg + in_deg),
            out_rel_mean,
            in_rel_mean,
            out_rel_mean - in_rel_mean,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return feat


def convert_entity(root: Path, name: str, out_root: Path) -> dict:
    ds = Entities(root=str(root), name=name)
    data = ds[0]

    edge_index = to_numpy(data.edge_index, np.int64)
    edge_type = to_numpy(data.edge_type, np.int64)

    n = int(data.num_nodes)
    e = int(edge_index.shape[1])
    rel_count = int(np.unique(edge_type).shape[0])

    # Relation-frequency as a second edge channel so discrete edge_type is not only an ID scalar.
    rel_hist = np.bincount(edge_type)
    rel_freq = rel_hist[edge_type].astype(np.float32)
    rel_freq = rel_freq / max(1.0, float(rel_freq.max()))
    rel_norm = edge_type.astype(np.float32) / max(1.0, float(edge_type.max()))

    edge_attr = np.stack([rel_norm, rel_freq], axis=1).astype(np.float32, copy=False)
    edge_weight = np.ones((e,), dtype=np.float32)

    feat = build_node_struct_feat(edge_index=edge_index, edge_type=edge_type, num_nodes=n)

    # Build full-node label vector with unknown=-1; known labels come from train/test split.
    y_full = np.full((n,), -1, dtype=np.int64)
    train_idx = to_numpy(data.train_idx, np.int64)
    test_idx = to_numpy(data.test_idx, np.int64)
    train_y = to_numpy(data.train_y, np.int64)
    test_y = to_numpy(data.test_y, np.int64)
    y_full[train_idx] = train_y
    y_full[test_idx] = test_y

    out_name = f"entities_{name.lower()}"
    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{out_name}_edge_index.npy", edge_index)
    np.save(out_dir / f"{out_name}_edge_weight.npy", edge_weight)
    np.save(out_dir / f"{out_name}_edge_attr.npy", edge_attr)
    np.save(out_dir / f"{out_name}_feat.npy", feat)
    np.save(out_dir / f"{out_name}_label.npy", y_full)

    known_mask = y_full >= 0
    num_known = int(known_mask.sum())
    num_classes = int(y_full[known_mask].max() + 1) if num_known > 0 else 0
    summary = {
        "dataset": out_name,
        "source": f"PyG Entities/{name}",
        "num_nodes": n,
        "num_edges": e,
        "num_relations": rel_count,
        "edge_attr_dim": int(edge_attr.shape[1]),
        "node_feat_dim": int(feat.shape[1]),
        "num_train_labels": int(train_y.shape[0]),
        "num_test_labels": int(test_y.shape[0]),
        "num_known_labels_total": num_known,
        "known_label_ratio": float(num_known / n),
        "num_classes": num_classes,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "label_mapping": {"unknown": -1},
        "source_dataset": name,
        "source_type": "torch_geometric.datasets.Entities",
        "num_classes": num_classes,
        "num_relations": rel_count,
    }
    with open(out_dir / f"{out_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare PyG Entities datasets for this repository.")
    parser.add_argument("--root", type=str, default="data", help="PyG download root")
    parser.add_argument("--out_root", type=str, default="data", help="Output root")
    parser.add_argument(
        "--datasets",
        type=str,
        default="AIFB,MUTAG,BGS,AM",
        help="Comma-separated subset of {AIFB,MUTAG,BGS,AM}",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    selected = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    for n in selected:
        if n not in ENTITY_NAMES:
            raise ValueError(f"Unknown dataset name: {n}. Valid: {ENTITY_NAMES}")

    summaries = []
    for name in selected:
        s = convert_entity(root=root, name=name, out_root=out_root)
        summaries.append(s)
        print(
            f"[ok] {s['dataset']}: nodes={s['num_nodes']}, edges={s['num_edges']}, "
            f"relations={s['num_relations']}, known_ratio={s['known_label_ratio']:.6f}"
        )

    out_sum = out_root / "Entities" / "prepared_summary.json"
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[done] summary: {out_sum}")


if __name__ == "__main__":
    main()
