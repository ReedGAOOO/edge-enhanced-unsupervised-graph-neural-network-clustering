from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


def load_saved_hetero_dataset(dataset_dir: str | Path) -> Tuple[HeteroData, dict]:
    dataset_dir = Path(dataset_dir).resolve()
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {dataset_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    data = HeteroData()
    for node_type, spec in meta["node_types"].items():
        x = np.load(dataset_dir / spec["feature_file"])
        if x.ndim != 2:
            raise ValueError(f"{node_type}: feature matrix must be [N, F], got {x.shape}")
        data[node_type].x = torch.tensor(x, dtype=torch.float32)
        id_file = spec.get("id_file")
        if id_file:
            data[node_type].node_id = torch.tensor(np.load(dataset_dir / id_file), dtype=torch.long)
        label_file = spec.get("label_file")
        if label_file:
            data[node_type].y = torch.tensor(np.load(dataset_dir / label_file), dtype=torch.long)

    for name, spec in meta["edge_types"].items():
        edge_index = np.load(dataset_dir / spec["edge_index_file"])
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{name}: edge_index must be [2, E], got {edge_index.shape}")
        edge_type = (spec["src_type"], spec["relation"], spec["dst_type"])
        data[edge_type].edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_file = spec.get("edge_attr_file")
        if edge_attr_file:
            edge_attr = np.load(dataset_dir / edge_attr_file)
            if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(f"{name}: edge_attr mismatch {edge_attr.shape} vs E={edge_index.shape[1]}")
            data[edge_type].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data.dataset_name = meta.get("dataset_name", dataset_dir.name)
    data.city = meta.get("city", "")
    data.target_node_type = meta.get("target_node_type", "")
    data.label_mapping = meta.get("label_mapping", {})
    data.node_type_order = meta.get("node_type_order", list(data.node_types))
    return data, meta


def hetero_in_dims(data: HeteroData) -> Dict[str, int]:
    return {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}


def hetero_edge_dims(data: HeteroData) -> Dict[Tuple[str, str, str], int]:
    dims: Dict[Tuple[str, str, str], int] = {}
    for edge_type in data.edge_types:
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        dims[edge_type] = int(edge_attr.shape[1]) if edge_attr is not None else 0
    return dims
