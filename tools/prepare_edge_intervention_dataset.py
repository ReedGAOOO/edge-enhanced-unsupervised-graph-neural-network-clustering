#!/usr/bin/env python3
"""
Create sparse-format intervention datasets for edge-semantics validation.

The script keeps node features and labels unchanged, then perturbs edge
attributes or topology. It is intentionally dataset-format generic, but the
default relation aliases match the Urban V3b schema.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


URBAN_V3B_RELATION_BITS = {
    "street": "is_street_backed",
    "junction": "is_junction_only",
    "geom": "is_geom_fallback",
    "building": "is_shared_building",
    "shared_building": "is_shared_building",
}


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_meta(dataset_dir: Path, dataset: str) -> dict:
    path = dataset_dir / f"{dataset}_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_file(dataset_dir: Path, dataset: str, suffix: str) -> Path:
    return dataset_dir / f"{dataset}_{suffix}"


def _target_file(out_dir: Path, out_name: str, suffix: str) -> Path:
    return out_dir / f"{out_name}_{suffix}"


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _feature_name_to_index(name: str, feature_names: list[str]) -> int:
    key = URBAN_V3B_RELATION_BITS.get(name, name)
    if key.isdigit():
        idx = int(key)
    elif key in feature_names:
        idx = int(feature_names.index(key))
    else:
        raise ValueError(f"Unknown edge feature/relation '{name}'. Known names: {feature_names}")
    if idx < 0 or idx >= len(feature_names):
        raise ValueError(f"Feature index out of range: {idx}, dim={len(feature_names)}")
    return idx


def _feature_indices(names: Iterable[str], feature_names: list[str]) -> list[int]:
    return sorted(set(_feature_name_to_index(name, feature_names) for name in names))


def _condition_label(mode: str, values: list[str], seed: int) -> str:
    if mode == "permute_edge_attr":
        return f"permEA_s{seed}"
    if mode == "zero_edge_attr":
        return "zeroEA_" + "-".join(values)
    if mode == "drop_relation":
        return "dropREL_" + "-".join(values)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare edge intervention dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--out_name", type=str, default="")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["permute_edge_attr", "zero_edge_attr", "drop_relation"],
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature/relation names or indices. Required for zero/drop modes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--copy_static",
        action="store_true",
        help="Copy unchanged feature/label files instead of symlinking them.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_path)
    out_root = Path(args.out_root)
    dataset = args.dataset
    src_dir = root / dataset
    if not src_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {src_dir}")

    meta = _load_meta(src_dir, dataset)
    feature_names = list(meta.get("edge_feature_names", []))

    edge_index_path = _source_file(src_dir, dataset, "edge_index.npy")
    edge_attr_path = _source_file(src_dir, dataset, "edge_attr.npy")
    feat_path = _source_file(src_dir, dataset, "feat.npy")
    label_path = _source_file(src_dir, dataset, "label.npy")
    edge_weight_path = _source_file(src_dir, dataset, "edge_weight.npy")

    for path in (edge_index_path, edge_attr_path, feat_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"Required sparse dataset file missing: {path}")

    edge_index = np.load(edge_index_path, allow_pickle=False)
    edge_attr = np.load(edge_attr_path, allow_pickle=False).astype(np.float32, copy=True)
    edge_weight = np.load(edge_weight_path, allow_pickle=False).astype(np.float32, copy=True) if edge_weight_path.exists() else None

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(f"edge_attr must be [E, D], got {edge_attr.shape}, E={edge_index.shape[1]}")
    if not feature_names:
        feature_names = [str(i) for i in range(edge_attr.shape[1])]
    if len(feature_names) != edge_attr.shape[1]:
        feature_names = [str(i) for i in range(edge_attr.shape[1])]

    feature_values = _parse_csv(args.features)
    if args.mode in {"zero_edge_attr", "drop_relation"} and not feature_values:
        raise ValueError("--features is required for zero_edge_attr/drop_relation.")

    rng = np.random.default_rng(int(args.seed))
    intervention: dict[str, object] = {
        "mode": args.mode,
        "seed": int(args.seed),
        "features": feature_values,
        "threshold": float(args.threshold),
    }

    if args.mode == "permute_edge_attr":
        order = rng.permutation(edge_attr.shape[0])
        edge_attr = edge_attr[order]
        keep = np.ones(edge_attr.shape[0], dtype=bool)
    elif args.mode == "zero_edge_attr":
        indices = _feature_indices(feature_values, feature_names)
        edge_attr[:, indices] = 0.0
        keep = np.ones(edge_attr.shape[0], dtype=bool)
        intervention["feature_indices"] = indices
        intervention["feature_names"] = [feature_names[i] for i in indices]
    elif args.mode == "drop_relation":
        indices = _feature_indices(feature_values, feature_names)
        active = np.any(edge_attr[:, indices] > float(args.threshold), axis=1)
        keep = ~active
        if int(keep.sum()) <= 0:
            raise ValueError("Intervention would remove all edges.")
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]
        if edge_weight is not None:
            edge_weight = edge_weight[keep]
        intervention["feature_indices"] = indices
        intervention["feature_names"] = [feature_names[i] for i in indices]
        intervention["removed_edges_directed"] = int((~keep).sum())
        intervention["kept_edges_directed"] = int(keep.sum())
    else:
        raise AssertionError(args.mode)

    label = _condition_label(args.mode, feature_values, int(args.seed))
    out_name = args.out_name.strip() or f"{dataset}_{label}"
    out_dir = out_root / out_name
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {out_dir}. Use --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    _link_or_copy(feat_path, _target_file(out_dir, out_name, "feat.npy"), copy=bool(args.copy_static))
    _link_or_copy(label_path, _target_file(out_dir, out_name, "label.npy"), copy=bool(args.copy_static))
    np.save(_target_file(out_dir, out_name, "edge_index.npy"), edge_index.astype(np.int64, copy=False))
    np.save(_target_file(out_dir, out_name, "edge_attr.npy"), edge_attr.astype(np.float32, copy=False))
    if edge_weight is not None:
        np.save(_target_file(out_dir, out_name, "edge_weight.npy"), edge_weight.astype(np.float32, copy=False))

    out_meta = dict(meta)
    out_meta.update(
        {
            "dataset_name": out_name,
            "source_dataset": dataset,
            "edge_intervention": intervention,
            "n_edges_directed": int(edge_index.shape[1]),
            "n_edges_undirected": int(edge_index.shape[1] // 2),
            "edge_feature_names": feature_names,
            "edge_feature_dim": int(edge_attr.shape[1]),
        }
    )
    _target_file(out_dir, out_name, "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print("[ok] edge intervention dataset created")
    print(f" source={dataset}")
    print(f" output={out_name}")
    print(f" mode={args.mode}")
    print(f" edges_directed={edge_index.shape[1]}")


if __name__ == "__main__":
    main()
