#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.hetero_model import EdgeAwareHeteroGraphSAGE
from utils.hetero_data_utils import hetero_edge_dims, hetero_in_dims, load_saved_hetero_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for urban hetero v2 dataset + edge-aware hetero GraphSAGE.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    data, meta = load_saved_hetero_dataset(args.dataset_dir)
    target_node_type = str(meta.get("target_node_type", "plot"))
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    edge_attr_dict = {edge_type: getattr(data[edge_type], "edge_attr", None) for edge_type in data.edge_types}

    y = getattr(data[target_node_type], "y", None)
    if y is None:
        out_dim = 8
        known_count = 0
    else:
        known = y >= 0
        known_count = int(known.sum().item())
        out_dim = int(y[known].max().item() + 1) if known_count > 0 else 1

    model = EdgeAwareHeteroGraphSAGE(
        in_dims=hetero_in_dims(data),
        edge_dims=hetero_edge_dims(data),
        metadata=data.metadata(),
        hidden_dim=int(args.hidden_dim),
        out_dim=int(out_dim),
        num_layers=int(args.num_layers),
        target_node_type=target_node_type,
        dropout=0.0,
    )
    model.train()

    embeds, logits = model(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
    )
    loss = None
    if y is not None and known_count > 0 and logits.shape[1] > 1:
        known = y >= 0
        loss = F.cross_entropy(logits[known], y[known])
        loss.backward()

    summary = {
        "dataset_name": meta.get("dataset_name", Path(args.dataset_dir).name),
        "city": meta.get("city", ""),
        "target_node_type": target_node_type,
        "node_types": {node_type: list(data[node_type].x.shape) for node_type in data.node_types},
        "edge_types": {str(edge_type): int(data[edge_type].edge_index.shape[1]) for edge_type in data.edge_types},
        "target_logits_shape": list(logits.shape),
        "known_target_labels": int(known_count),
        "loss": float(loss.item()) if loss is not None else None,
        "status": "ok",
    }

    if args.out_json:
        out_fp = Path(args.out_json).resolve()
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
