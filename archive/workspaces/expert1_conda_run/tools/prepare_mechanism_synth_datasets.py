#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


@dataclass
class DatasetSpec:
    dataset: str
    num_nodes: int
    num_classes: int
    feat_dim: int
    avg_degree_target: float
    homophily_target: float
    edge_signal_target: float
    data_seed: int
    p_in: float
    p_out: float
    num_edges_directed: int
    density_directed: float
    avg_degree_observed: float
    homophily_observed: float
    edge_attr_signal_corr: float
    edge_attr_signal_snr: float


def make_balanced_labels(num_nodes: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    y = np.arange(num_nodes, dtype=np.int64) % int(num_classes)
    rng.shuffle(y)
    return y


def solve_sbm_probs(num_nodes: int, num_classes: int, avg_degree: float, homophily: float) -> tuple[float, float]:
    block_size = max(2, num_nodes // max(1, num_classes))
    inter_size = max(1, num_nodes - block_size)
    h = float(np.clip(homophily, 1e-4, 1.0 - 1e-4))
    d = float(max(1e-4, avg_degree))
    p_in = (h * d) / float(max(1, block_size - 1))
    p_out = ((1.0 - h) * d) / float(max(1, inter_size))
    p_in = float(np.clip(p_in, 1e-6, 0.95))
    p_out = float(np.clip(p_out, 1e-6, 0.95))
    return p_in, p_out


def sample_sbm_edges(y: np.ndarray, p_in: float, p_out: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    u, v = np.triu_indices(n, k=1)
    same = y[u] == y[v]
    probs = np.where(same, p_in, p_out)
    keep = rng.random(size=u.shape[0]) < probs
    src_u = u[keep].astype(np.int64)
    dst_u = v[keep].astype(np.int64)
    src = np.concatenate([src_u, dst_u], axis=0)
    dst = np.concatenate([dst_u, src_u], axis=0)
    return src.astype(np.int64), dst.astype(np.int64)


def connect_isolates(src: np.ndarray, dst: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    deg = np.bincount(src, minlength=n)
    isolates = np.where(deg == 0)[0]
    if isolates.size == 0:
        return src, dst
    add_s, add_d = [], []
    for i in isolates.tolist():
        same_pool = np.where((y == y[i]) & (np.arange(n) != i))[0]
        if same_pool.size > 0:
            j = int(same_pool[rng.integers(0, same_pool.size)])
        else:
            j = int((i + 1) % n)
        add_s.extend([i, j])
        add_d.extend([j, i])
    src_new = np.concatenate([src, np.asarray(add_s, dtype=np.int64)], axis=0)
    dst_new = np.concatenate([dst, np.asarray(add_d, dtype=np.int64)], axis=0)
    return src_new, dst_new


def build_node_features(
    y: np.ndarray,
    feat_dim: int,
    feature_sep: float,
    feature_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    k = int(y.max()) + 1
    centroids = rng.normal(0.0, 1.0, size=(k, feat_dim)).astype(np.float32)
    centroids = (feature_sep * centroids).astype(np.float32)
    x = centroids[y] + rng.normal(0.0, feature_noise, size=(y.shape[0], feat_dim)).astype(np.float32)
    return x.astype(np.float32)


def build_edge_attr(
    src: np.ndarray,
    dst: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    edge_signal: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    same = (y[src] == y[dst]).astype(np.float32)
    same_sign = 2.0 * same - 1.0
    signal = float(np.clip(edge_signal, 0.0, 1.0))
    noise_scale = float(np.sqrt(max(0.0, 1.0 - signal * signal)))
    attr0 = signal * same_sign + noise_scale * rng.normal(0.0, 1.0, size=same.shape[0]).astype(np.float32)

    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    attr1 = np.sum(xn[src] * xn[dst], axis=1).astype(np.float32)

    deg = np.bincount(src, minlength=x.shape[0]).astype(np.float32)
    attr2 = np.log1p(np.abs(deg[src] - deg[dst])).astype(np.float32)
    attr3 = rng.normal(0.0, 1.0, size=same.shape[0]).astype(np.float32)
    edge_attr = np.stack([attr0, attr1, attr2, attr3], axis=1).astype(np.float32)

    if np.std(attr0) < 1e-12 or np.std(same_sign) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(attr0, same_sign)[0, 1])
    mean_same = float(attr0[same > 0.5].mean()) if np.any(same > 0.5) else 0.0
    mean_diff = float(attr0[same <= 0.5].mean()) if np.any(same <= 0.5) else 0.0
    snr = float((mean_same - mean_diff) / (float(np.std(attr0)) + 1e-12))
    return edge_attr, corr, snr


def write_dataset(
    root: Path,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    edge_attr: np.ndarray,
    meta: dict,
    overwrite: bool,
) -> None:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    req = [
        out / f"{name}_feat.npy",
        out / f"{name}_label.npy",
        out / f"{name}_edge_index.npy",
        out / f"{name}_edge_weight.npy",
        out / f"{name}_edge_attr.npy",
        out / f"{name}_meta.json",
    ]
    if (not overwrite) and all(p.exists() for p in req):
        return

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)

    np.save(out / f"{name}_feat.npy", x.astype(np.float32))
    np.save(out / f"{name}_label.npy", y.astype(np.int64))
    np.save(out / f"{name}_edge_index.npy", edge_index)
    np.save(out / f"{name}_edge_weight.npy", edge_weight)
    np.save(out / f"{name}_edge_attr.npy", edge_attr.astype(np.float32))
    with open(out / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare controllable synthetic graph datasets for mechanism validation.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="synth_mech")
    parser.add_argument("--num_nodes", type=int, default=1200)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--avg_degree", type=float, default=16.0)
    parser.add_argument("--feature_sep", type=float, default=1.5)
    parser.add_argument("--feature_noise", type=float, default=1.0)
    parser.add_argument("--homophily", type=str, default="0.85,0.65,0.45")
    parser.add_argument("--edge_signal", type=str, default="0.9,0.6,0.3,0.0")
    parser.add_argument("--data_seeds", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest_name", type=str, default="synth_mech_manifest.csv")
    args = parser.parse_args()

    root = Path(args.root_path)
    root.mkdir(parents=True, exist_ok=True)

    homophily_list = parse_float_list(args.homophily)
    edge_signal_list = parse_float_list(args.edge_signal)
    data_seeds = parse_int_list(args.data_seeds)

    rows: List[DatasetSpec] = []
    for h in homophily_list:
        for s in edge_signal_list:
            for data_seed in data_seeds:
                rng = np.random.default_rng(int(data_seed))
                y = make_balanced_labels(args.num_nodes, args.num_classes, rng)
                p_in, p_out = solve_sbm_probs(
                    num_nodes=args.num_nodes,
                    num_classes=args.num_classes,
                    avg_degree=args.avg_degree,
                    homophily=h,
                )
                src, dst = sample_sbm_edges(y, p_in=p_in, p_out=p_out, rng=rng)
                src, dst = connect_isolates(src, dst, y, rng)
                x = build_node_features(
                    y=y,
                    feat_dim=args.feat_dim,
                    feature_sep=args.feature_sep,
                    feature_noise=args.feature_noise,
                    rng=rng,
                )
                edge_attr, attr_corr, attr_snr = build_edge_attr(src=src, dst=dst, y=y, x=x, edge_signal=s, rng=rng)
                same = (y[src] == y[dst]).astype(np.float32)
                num_edges = int(src.shape[0])
                density = float(num_edges / max(1.0, float(args.num_nodes * (args.num_nodes - 1))))
                avg_deg_obs = float(np.bincount(src, minlength=args.num_nodes).mean())
                hom_obs = float(same.mean()) if same.size > 0 else 0.0
                name = (
                    f"{args.prefix}"
                    f"_h{int(round(100 * h)):02d}"
                    f"_s{int(round(100 * s)):02d}"
                    f"_ds{int(data_seed):02d}"
                )

                spec = DatasetSpec(
                    dataset=name,
                    num_nodes=int(args.num_nodes),
                    num_classes=int(args.num_classes),
                    feat_dim=int(args.feat_dim),
                    avg_degree_target=float(args.avg_degree),
                    homophily_target=float(h),
                    edge_signal_target=float(s),
                    data_seed=int(data_seed),
                    p_in=float(p_in),
                    p_out=float(p_out),
                    num_edges_directed=num_edges,
                    density_directed=density,
                    avg_degree_observed=avg_deg_obs,
                    homophily_observed=hom_obs,
                    edge_attr_signal_corr=float(attr_corr),
                    edge_attr_signal_snr=float(attr_snr),
                )
                meta = asdict(spec)
                meta["description"] = "Synthetic mechanism-validation graph with controllable homophily and edge-attribute signal."
                meta["label_mapping"] = {}
                write_dataset(
                    root=root,
                    name=name,
                    x=x,
                    y=y,
                    src=src,
                    dst=dst,
                    edge_attr=edge_attr,
                    meta=meta,
                    overwrite=bool(args.overwrite),
                )
                rows.append(spec)
                print(
                    f"[ok] {name}: "
                    f"E={num_edges}, hom_obs={hom_obs:.4f}, attr_corr={attr_corr:.4f}, p_in={p_in:.4g}, p_out={p_out:.4g}"
                )

    manifest = root / args.manifest_name
    if rows:
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows([asdict(r) for r in rows])
    print(f"[done] wrote {manifest}")


if __name__ == "__main__":
    main()
