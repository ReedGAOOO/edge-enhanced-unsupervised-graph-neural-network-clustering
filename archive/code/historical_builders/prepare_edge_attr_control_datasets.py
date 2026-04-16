#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prepare_mechanism_synth_datasets import (
    DatasetSpec,
    build_node_features,
    connect_isolates,
    make_balanced_labels,
    parse_float_list,
    parse_int_list,
    sample_sbm_edges,
    solve_sbm_probs,
    write_dataset,
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_control_edge_attr(
    src: np.ndarray,
    dst: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    edge_signal: float,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    mode = str(mode).lower()
    same = (y[src] == y[dst]).astype(np.float32)
    same_sign = 2.0 * same - 1.0
    coarse_y = (y.astype(np.int64) // 2).astype(np.int64)
    coarse_same = (coarse_y[src] == coarse_y[dst]).astype(np.float32)
    coarse_same_sign = 2.0 * coarse_same - 1.0

    signal = float(np.clip(edge_signal, 0.0, 1.0))
    noise_scale = float(np.sqrt(max(0.0, 1.0 - signal * signal)))

    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    cos = np.sum(xn[src] * xn[dst], axis=1).astype(np.float32)
    l2 = np.linalg.norm(x[src] - x[dst], axis=1).astype(np.float32)
    deg = np.bincount(src, minlength=x.shape[0]).astype(np.float32)
    deg_gap = np.log1p(np.abs(deg[src] - deg[dst])).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=same.shape[0]).astype(np.float32)

    if mode == "redundant":
        # Edge attributes repeat node-level similarity rather than adding new relation semantics.
        attr0 = signal * cos + noise_scale * noise
        semantics = "redundant_node_similarity"
    elif mode == "misleading":
        # Edge attributes are anti-correlated with the fine-grained cluster relation.
        attr0 = -signal * same_sign + noise_scale * noise
        semantics = "anti_fine_label_relation"
    elif mode in {"hier", "hierarchical", "hierarchical_only"}:
        # Edge attributes align with a coarse super-cluster (2-way) rather than fine 4-class labels.
        attr0 = signal * coarse_same_sign + noise_scale * noise
        semantics = "coarse_supercluster_relation"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    attr1 = cos
    attr2 = deg_gap
    attr3 = noise
    edge_attr = np.stack([attr0, attr1, attr2, attr3], axis=1).astype(np.float32)

    mean_same = float(attr0[same > 0.5].mean()) if np.any(same > 0.5) else 0.0
    mean_diff = float(attr0[same <= 0.5].mean()) if np.any(same <= 0.5) else 0.0
    snr = float((mean_same - mean_diff) / (float(np.std(attr0)) + 1e-12))
    meta = {
        "edge_attr_mode": mode,
        "edge_attr_semantics": semantics,
        "edge_attr_signal_corr": _safe_corr(attr0, same_sign),
        "edge_attr_signal_snr": snr,
        "edge_attr_coarse_signal_corr": _safe_corr(attr0, coarse_same_sign),
    }
    return edge_attr, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired control datasets for edge-attribute mechanism analysis.")
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="synth_edgectrl_v1")
    parser.add_argument("--num_nodes", type=int, default=1200)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--avg_degree", type=float, default=16.0)
    parser.add_argument("--feature_sep", type=float, default=1.5)
    parser.add_argument("--feature_noise", type=float, default=1.0)
    parser.add_argument("--homophily", type=str, default="0.85,0.65,0.45")
    parser.add_argument("--edge_signal", type=str, default="0.9")
    parser.add_argument("--modes", type=str, default="redundant,misleading,hierarchical")
    parser.add_argument("--data_seeds", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_path)
    root.mkdir(parents=True, exist_ok=True)

    homophily_list = parse_float_list(args.homophily)
    edge_signal_list = parse_float_list(args.edge_signal)
    data_seeds = parse_int_list(args.data_seeds)
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
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
                    edge_attr, ctrl_meta = build_control_edge_attr(
                        src=src,
                        dst=dst,
                        y=y,
                        x=x,
                        edge_signal=s,
                        mode=mode,
                        rng=rng,
                    )
                    same = (y[src] == y[dst]).astype(np.float32)
                    num_edges = int(src.shape[0])
                    density = float(num_edges / max(1.0, float(args.num_nodes * (args.num_nodes - 1))))
                    avg_deg_obs = float(np.bincount(src, minlength=args.num_nodes).mean())
                    hom_obs = float(same.mean()) if same.size > 0 else 0.0

                    name = (
                        f"{args.prefix}"
                        f"_m{mode[:4]}"
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
                        edge_attr_signal_corr=float(ctrl_meta["edge_attr_signal_corr"]),
                        edge_attr_signal_snr=float(ctrl_meta["edge_attr_signal_snr"]),
                    )
                    meta = asdict(spec)
                    meta.update(ctrl_meta)
                    meta["description"] = (
                        "Synthetic paired control graph for edge-attribute analysis. "
                        "Graph structure and node features are fixed; only edge-attribute semantics vary."
                    )
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
                        overwrite=args.overwrite,
                    )


if __name__ == "__main__":
    main()
