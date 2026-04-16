#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


DATASET_CFG = {
    "amazon": {
        "mat_path": "data/FraudAmazon/Amazon.mat",
        "out_name": "fraud_amazon_union",
        "rel_keys": ["net_upu", "net_usu", "net_uvu"],
        "prefer_base": "union",
    },
    "yelp": {
        "mat_path": "data/FraudYelp/YelpChi.mat",
        "out_name": "fraud_yelp_homo",
        "rel_keys": ["net_rur", "net_rtr", "net_rsr"],
        "prefer_base": "homo",
    },
}


def to_csr01(m: sp.spmatrix) -> sp.csr_matrix:
    m = m.tocsr().astype(np.float32)
    if m.nnz > 0:
        m.data = np.ones_like(m.data, dtype=np.float32)
        m.eliminate_zeros()
    return m


def build_base_graph(homo: sp.csr_matrix, rels: list[sp.csr_matrix], base_mode: str) -> sp.csr_matrix:
    base_mode = base_mode.lower()
    if base_mode == "homo":
        base = homo.copy()
    elif base_mode == "union":
        base = homo.copy()
        for r in rels:
            base = base + r
        if base.nnz > 0:
            base.data = np.ones_like(base.data, dtype=np.float32)
            base.eliminate_zeros()
    else:
        raise ValueError(f"Unknown base mode: {base_mode}, expected one of [homo, union].")
    return base


def stabilize_node_features(x: np.ndarray) -> np.ndarray:
    # FraudAmazon raw features can be very heavy-tailed and include negatives.
    # Signed log-compression + column-wise z-score keeps geometry numerically stable.
    x = np.asarray(x, dtype=np.float32)
    x = np.sign(x) * np.log1p(np.abs(x))
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    x = (x - mean) / std
    x = np.clip(x, -10.0, 10.0).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32, copy=False)
    return x


def build_edge_attr_from_rel(base: sp.csr_matrix, mats: list[sp.csr_matrix]) -> np.ndarray:
    base_coo = base.tocoo()
    rows = base_coo.row.astype(np.int64, copy=False)
    cols = base_coo.col.astype(np.int64, copy=False)
    chans = []
    for mat in mats:
        v = np.asarray(mat[rows, cols]).reshape(-1).astype(np.float32, copy=False)
        chans.append((v > 0).astype(np.float32, copy=False))
    edge_attr = np.stack(chans, axis=1).astype(np.float32, copy=False)
    return edge_attr


def convert_single(
    mat_path: Path,
    out_root: Path,
    out_name: str,
    rel_keys: list[str],
    base_mode: str,
) -> dict:
    d = sio.loadmat(mat_path)
    homo = to_csr01(d["homo"])
    rels = [to_csr01(d[k]) for k in rel_keys]

    features = d["features"]
    if sp.issparse(features):
        x = features.tocsr().astype(np.float32).toarray()
    else:
        x = np.asarray(features, dtype=np.float32)
    x = stabilize_node_features(x)
    y = np.asarray(d["label"]).reshape(-1).astype(np.int64, copy=False)

    base = build_base_graph(homo=homo, rels=rels, base_mode=base_mode)
    coo = base.tocoo()
    edge_index = np.stack([coo.row, coo.col], axis=0).astype(np.int64, copy=False)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)

    # Channels: [homo_presence, relation_1, relation_2, relation_3]
    edge_attr = build_edge_attr_from_rel(base, [homo] + rels)

    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{out_name}_edge_index.npy", edge_index)
    np.save(out_dir / f"{out_name}_edge_weight.npy", edge_weight)
    np.save(out_dir / f"{out_name}_edge_attr.npy", edge_attr)
    np.save(out_dir / f"{out_name}_feat.npy", x)
    np.save(out_dir / f"{out_name}_label.npy", y)

    meta = {
        "source_type": "CARE-GNN mat",
        "source_mat": str(mat_path),
        "base_mode": base_mode,
        "edge_attr_channels": ["homo"] + rel_keys,
        "label_mapping": {"benign_or_legit": 0, "fraud_or_spam": 1},
    }
    with open(out_dir / f"{out_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    class_vals, class_cnt = np.unique(y, return_counts=True)
    summary = {
        "dataset": out_name,
        "mat_path": str(mat_path),
        "base_mode": base_mode,
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "node_feat_dim": int(x.shape[1]),
        "edge_attr_dim": int(edge_attr.shape[1]),
        "known_label_ratio": float((y >= 0).mean()),
        "class_hist": {str(int(k)): int(v) for k, v in zip(class_vals, class_cnt)},
        "edge_attr_channel_nonzero_ratio": {
            ch: float((edge_attr[:, i] > 0).mean())
            for i, ch in enumerate(["homo"] + rel_keys)
        },
        "feat_min": float(x.min()),
        "feat_max": float(x.max()),
        "feat_mean": float(x.mean()),
        "feat_std": float(x.std()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FraudAmazon/FraudYelp .mat for ATsDataset format.")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument(
        "--datasets",
        type=str,
        default="amazon,yelp",
        help="Comma-separated subset of [amazon,yelp]",
    )
    parser.add_argument(
        "--base_mode",
        type=str,
        default="auto",
        choices=["auto", "homo", "union"],
        help="Graph base mode. auto: use dataset-specific default (amazon=union, yelp=homo).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    selected = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    summaries = []
    for name in selected:
        if name not in DATASET_CFG:
            raise ValueError(f"Unknown dataset '{name}', expected one of {list(DATASET_CFG.keys())}")
        cfg = DATASET_CFG[name]
        base_mode = cfg["prefer_base"] if args.base_mode == "auto" else args.base_mode
        summary = convert_single(
            mat_path=Path(cfg["mat_path"]),
            out_root=out_root,
            out_name=cfg["out_name"],
            rel_keys=cfg["rel_keys"],
            base_mode=base_mode,
        )
        summaries.append(summary)
        print(
            f"[ok] {summary['dataset']}: nodes={summary['num_nodes']}, "
            f"edges={summary['num_edges']}, feat_dim={summary['node_feat_dim']}, "
            f"edge_attr_dim={summary['edge_attr_dim']}"
        )

    report = out_root / "Fraud" / "prepared_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[done] summary: {report}")


if __name__ == "__main__":
    main()
