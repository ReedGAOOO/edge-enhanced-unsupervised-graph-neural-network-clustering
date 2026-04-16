#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_MAP = {
    "otc": "OTCNet.csv",
    "alpha": "BTCAlphaNet.csv",
    "rfa": "RFAnet.csv",
    "wikisigned": "WikiSignedNet.csv",
    "epinion": "EpinionNet.csv",
}


def iter_chunks(csv_path: Path, chunksize: int):
    return pd.read_csv(
        csv_path,
        header=None,
        names=["src", "dst", "w"],
        chunksize=chunksize,
    )


def collect_node_ids(csv_path: Path, chunksize: int) -> np.ndarray:
    node_set = set()
    for ch in iter_chunks(csv_path, chunksize):
        node_set.update(ch["src"].astype(np.int64).tolist())
        node_set.update(ch["dst"].astype(np.int64).tolist())
    node_ids = np.array(sorted(node_set), dtype=np.int64)
    return node_ids


def convert_single(
    csv_path: Path,
    out_dir: Path,
    dataset_name: str,
    chunksize: int = 1_000_000,
    drop_zero_weight: bool = True,
    make_undirected: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids = collect_node_ids(csv_path, chunksize)
    num_nodes = int(node_ids.shape[0])

    src_parts = []
    dst_parts = []
    w_parts = []

    raw_edges = 0
    kept_edges = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    raw_zero = 0

    for ch in iter_chunks(csv_path, chunksize):
        src_raw = ch["src"].to_numpy(dtype=np.int64, copy=False)
        dst_raw = ch["dst"].to_numpy(dtype=np.int64, copy=False)
        w_raw = ch["w"].to_numpy(dtype=np.float32, copy=False)

        raw_edges += int(w_raw.shape[0])
        if w_raw.shape[0] > 0:
            raw_min = min(raw_min, float(w_raw.min()))
            raw_max = max(raw_max, float(w_raw.max()))
            raw_zero += int((w_raw == 0).sum())

        if drop_zero_weight:
            mask = w_raw != 0
            src_raw = src_raw[mask]
            dst_raw = dst_raw[mask]
            w_raw = w_raw[mask]

        if w_raw.size == 0:
            continue

        src = np.searchsorted(node_ids, src_raw).astype(np.int64, copy=False)
        dst = np.searchsorted(node_ids, dst_raw).astype(np.int64, copy=False)

        if make_undirected:
            src_ud = np.concatenate([src, dst], axis=0)
            dst_ud = np.concatenate([dst, src], axis=0)
            w_ud = np.concatenate([w_raw, w_raw], axis=0)
            src, dst, w_raw = src_ud, dst_ud, w_ud

        src_parts.append(src)
        dst_parts.append(dst)
        w_parts.append(w_raw.astype(np.float32, copy=False))
        kept_edges += int(w_raw.shape[0])

    if not src_parts:
        raise RuntimeError(f"No edges kept after filtering in {csv_path}")

    src = np.concatenate(src_parts, axis=0)
    dst = np.concatenate(dst_parts, axis=0)
    w = np.concatenate(w_parts, axis=0).astype(np.float32, copy=False)

    # Normalize signed edge value to a bounded range while retaining sign semantics.
    abs_max = float(np.max(np.abs(w)))
    scale = abs_max if abs_max > 1e-12 else 1.0
    w_norm = (w / scale).astype(np.float32, copy=False)

    sign = np.sign(w_norm).astype(np.float32, copy=False)
    mag = np.abs(w_norm).astype(np.float32, copy=False)

    edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)
    edge_weight = mag.astype(np.float32, copy=False)
    edge_attr = np.stack([w_norm, mag, sign], axis=1).astype(np.float32, copy=False)

    # Node structural features derived from signed, directed weighted edges.
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float32)

    out_abs = np.bincount(src, weights=mag, minlength=num_nodes).astype(np.float32)
    in_abs = np.bincount(dst, weights=mag, minlength=num_nodes).astype(np.float32)

    pos_mask = (w_norm > 0).astype(np.float32)
    neg_mask = (w_norm < 0).astype(np.float32)

    out_pos = np.bincount(src, weights=(mag * pos_mask), minlength=num_nodes).astype(np.float32)
    in_pos = np.bincount(dst, weights=(mag * pos_mask), minlength=num_nodes).astype(np.float32)
    out_neg = np.bincount(src, weights=(mag * neg_mask), minlength=num_nodes).astype(np.float32)
    in_neg = np.bincount(dst, weights=(mag * neg_mask), minlength=num_nodes).astype(np.float32)

    eps = 1e-6
    out_mean_abs = out_abs / (out_deg + eps)
    in_mean_abs = in_abs / (in_deg + eps)
    out_balance = (out_pos - out_neg) / (out_pos + out_neg + eps)
    in_balance = (in_pos - in_neg) / (in_pos + in_neg + eps)

    feat = np.stack(
        [
            np.log1p(out_deg),
            np.log1p(in_deg),
            np.log1p(out_abs),
            np.log1p(in_abs),
            out_mean_abs,
            in_mean_abs,
            out_balance,
            in_balance,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    # Unsupervised setting: labels unknown.
    label = np.full((num_nodes,), -1, dtype=np.int64)

    np.save(out_dir / f"{dataset_name}_edge_index.npy", edge_index)
    np.save(out_dir / f"{dataset_name}_edge_weight.npy", edge_weight)
    np.save(out_dir / f"{dataset_name}_edge_attr.npy", edge_attr)
    np.save(out_dir / f"{dataset_name}_feat.npy", feat)
    np.save(out_dir / f"{dataset_name}_label.npy", label)
    np.save(out_dir / f"{dataset_name}_node_ids.npy", node_ids)

    summary = {
        "dataset": dataset_name,
        "source_csv": str(csv_path),
        "num_nodes": num_nodes,
        "raw_edges": raw_edges,
        "kept_edges": kept_edges,
        "drop_zero_weight": bool(drop_zero_weight),
        "make_undirected": bool(make_undirected),
        "raw_weight_min": raw_min,
        "raw_weight_max": raw_max,
        "raw_zero_edges": int(raw_zero),
        "kept_weight_min": float(w.min()),
        "kept_weight_max": float(w.max()),
        "normalized_abs_max_before_scale": abs_max,
        "edge_attr_dim": int(edge_attr.shape[1]),
        "node_feat_dim": int(feat.shape[1]),
        "positive_ratio_nonzero": float((w > 0).sum() / max(1, int((w != 0).sum()))),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare Bitcoin_WSN CSV files for this project.")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="data/Bitcoin_WSN/data-wsn",
        help="Directory containing source CSV files.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data",
        help="Output root dir where converted datasets are created.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="otc,alpha,rfa,wikisigned,epinion",
        help="Comma-separated dataset keys: otc,alpha,rfa,wikisigned,epinion",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="CSV chunksize for streaming conversion.",
    )
    parser.add_argument(
        "--keep_zero_weight",
        action="store_true",
        help="Keep zero-weight edges (default drops them).",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Mirror each edge to make graph undirected.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_root = Path(args.out_root)
    selected = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    all_summary = []
    for key in selected:
        if key not in DATASET_MAP:
            raise ValueError(f"Unknown dataset key: {key}. Valid keys: {sorted(DATASET_MAP.keys())}")
        src_csv = src_dir / DATASET_MAP[key]
        if not src_csv.exists():
            raise FileNotFoundError(f"Source file not found: {src_csv}")
        dataset_name = f"bitcoin_wsn_{key}"
        out_dir = out_root / dataset_name
        summary = convert_single(
            csv_path=src_csv,
            out_dir=out_dir,
            dataset_name=dataset_name,
            chunksize=args.chunksize,
            drop_zero_weight=(not args.keep_zero_weight),
            make_undirected=args.undirected,
        )
        all_summary.append(summary)
        print(
            f"[ok] {dataset_name}: nodes={summary['num_nodes']}, "
            f"edges(raw/kept)={summary['raw_edges']}/{summary['kept_edges']}, "
            f"raw_w_range=[{summary['raw_weight_min']}, {summary['raw_weight_max']}]"
        )

    report_path = out_root / "Bitcoin_WSN" / "prepared_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    print(f"[done] saved summary: {report_path}")


if __name__ == "__main__":
    main()
