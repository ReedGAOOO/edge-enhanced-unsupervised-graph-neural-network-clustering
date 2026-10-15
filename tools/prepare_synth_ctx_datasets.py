#!/usr/bin/env python3
"""Synthetic benchmark where edge attributes are actually needed (replaces synth_mech / synth_edgectrl).

The old suites had two flaws: node features alone gave KMeans NMI = 1.0 (feature_sep=1.5, 64 dims),
and edge_attr[:, 0] was a noisy copy of the pairwise same-label bit. This generator fixes both:

- weak node features (calibrated so KMeans(X) is far from 1.0),
- SBM topology swept across the Kesten-Stigum threshold (h=0.45 is below it for the defaults),
- three edge modes for channel 0:
    none        pure noise (edge attributes carry nothing)
    direct      s * same_sign + noise (context-free signal, usable by scalar edge weighting)
    contextual  s * same_sign * role_sign + noise, where role_sign = +1 if both endpoints share a
                latent binary role and -1 otherwise. Marginally uninformative; it can only be decoded
                together with endpoint context. With --role_mode onehot (default) roles are exposed as two
                one-hot node feature columns. With --role_mode xor2 each node has two latent bits exposed as two
                +-1 feature columns and role = bit1 XOR bit2, so role_sign is a product of two endpoint-context
                terms: a single edge-channel x endpoint-product reading can no longer decode it.
  Channels 1-3 are descriptive nuisances: endpoint feature cosine, log degree gap, pure noise.
- optional `_permEA` twins with edge attributes shuffled across edges.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_mechanism_synth_datasets import (
    connect_isolates,
    make_balanced_labels,
    parse_float_list,
    parse_int_list,
    sample_sbm_edges,
    solve_sbm_probs,
    write_dataset,
)

MODES = ("none", "direct", "contextual")


def build_features(y, role_cols, feat_dim, feature_sep, role_scale, rng):
    k = int(y.max()) + 1
    centroids = feature_sep * rng.normal(0.0, 1.0, size=(k, feat_dim))
    x = centroids[y] + rng.normal(0.0, 1.0, size=(y.shape[0], feat_dim))
    return np.concatenate([x, role_cols * role_scale], axis=1).astype(np.float32)


def draw_roles(num_nodes, role_mode, rng):
    """Return (roles, role feature columns)."""
    if role_mode == "onehot":
        roles = rng.integers(0, 2, size=num_nodes)
        return roles, np.eye(2)[roles]
    bits = rng.integers(0, 2, size=(num_nodes, 2))
    return bits[:, 0] ^ bits[:, 1], 2.0 * bits - 1.0


def build_edge_attr(src, dst, y, roles, x, mode, signal, rng):
    same_sign = np.where(y[src] == y[dst], 1.0, -1.0)
    role_sign = np.where(roles[src] == roles[dst], 1.0, -1.0)
    noise_scale = float(np.sqrt(max(0.0, 1.0 - signal * signal)))
    noise = rng.normal(0.0, 1.0, size=src.shape[0])
    if mode == "none":
        attr0 = noise
    elif mode == "direct":
        attr0 = signal * same_sign + noise_scale * noise
    elif mode == "contextual":
        attr0 = signal * same_sign * role_sign + noise_scale * noise
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    cos = np.sum(xn[src] * xn[dst], axis=1)
    deg = np.bincount(src, minlength=x.shape[0]).astype(np.float64)
    deg_gap = np.log1p(np.abs(deg[src] - deg[dst]))
    nuisance = rng.normal(0.0, 1.0, size=src.shape[0])
    return np.stack([attr0, cos, deg_gap, nuisance], axis=1).astype(np.float32)


def edge_stats(src, dst, y, roles, edge_attr):
    same_sign = np.where(y[src] == y[dst], 1.0, -1.0)
    role_sign = np.where(roles[src] == roles[dst], 1.0, -1.0)
    attr0 = edge_attr[:, 0].astype(np.float64)
    return {
        "attr0_corr_same": float(np.corrcoef(attr0, same_sign)[0, 1]),
        "attr0_x_role_corr_same": float(np.corrcoef(attr0 * role_sign, same_sign)[0, 1]),
    }


def symmetric_permutation(src, dst, edge_attr, rng):
    """Shuffle attributes across undirected edges, keeping both directions of an edge identical."""
    key = np.minimum(src, dst) * (int(max(src.max(), dst.max())) + 1) + np.maximum(src, dst)
    uniq, inv = np.unique(key, return_inverse=True)
    first = np.zeros(uniq.shape[0], dtype=np.int64)
    first[inv[::-1]] = np.arange(inv.shape[0])[::-1]
    perm = rng.permutation(uniq.shape[0])
    return edge_attr[first[perm]][inv]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="synth_ctx_v1")
    parser.add_argument("--num_nodes", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--avg_degree", type=float, default=12.0)
    parser.add_argument("--feature_sep", type=float, default=0.25)
    parser.add_argument("--role_scale", type=float, default=1.0)
    parser.add_argument("--role_mode", choices=["onehot", "xor2"], default="onehot",
                        help="onehot: role one-hot columns; xor2: role = XOR of two exposed latent bits.")
    parser.add_argument("--homophily", type=str, default="0.45,0.55,0.70")
    parser.add_argument("--modes", type=str, default="none,direct,contextual")
    parser.add_argument("--edge_signal", type=float, default=0.6)
    parser.add_argument("--perm_modes", type=str, default="direct,contextual",
                        help="Modes that also get an edge-attribute-permuted twin.")
    parser.add_argument("--data_seeds", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_path)
    root.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    perm_modes = {m.strip() for m in args.perm_modes.split(",") if m.strip()}
    rows = []
    for h in parse_float_list(args.homophily):
        for data_seed in parse_int_list(args.data_seeds):
            # Graph, labels and features are shared across modes so that only edge_attr differs.
            rng = np.random.default_rng(int(data_seed))
            y = make_balanced_labels(args.num_nodes, args.num_classes, rng)
            roles, role_cols = draw_roles(args.num_nodes, args.role_mode, rng)
            p_in, p_out = solve_sbm_probs(args.num_nodes, args.num_classes, args.avg_degree, h)
            src, dst = sample_sbm_edges(y, p_in=p_in, p_out=p_out, rng=rng)
            src, dst = connect_isolates(src, dst, y, rng)
            x = build_features(y, role_cols, args.feat_dim, args.feature_sep, args.role_scale, rng)
            for mode in modes:
                mode_rng = np.random.default_rng(1000 * int(data_seed) + MODES.index(mode))
                edge_attr = build_edge_attr(src, dst, y, roles, x, mode, args.edge_signal, mode_rng)
                variants = [("", edge_attr)]
                if mode in perm_modes:
                    variants.append(("_permEA", symmetric_permutation(src, dst, edge_attr, mode_rng)))
                for suffix, ea in variants:
                    stats = edge_stats(src, dst, y, roles, ea)
                    name = f"{args.prefix}_h{int(round(100 * h)):02d}_{mode}_ds{int(data_seed):02d}{suffix}"
                    meta = {
                        "dataset": name, "num_nodes": args.num_nodes, "num_classes": args.num_classes,
                        "feat_dim": int(x.shape[1]), "avg_degree_target": args.avg_degree,
                        "homophily_target": h, "homophily_observed": float(np.mean(y[src] == y[dst])),
                        "feature_sep": args.feature_sep, "edge_mode": mode, "edge_signal": args.edge_signal,
                        "edge_attr_permuted": bool(suffix), "data_seed": int(data_seed),
                        "p_in": p_in, "p_out": p_out, "num_edges_directed": int(src.shape[0]),
                        "role_mode": args.role_mode, "role_feature_columns": [args.feat_dim, args.feat_dim + 1], **stats,
                        "description": "Weak-feature SBM benchmark with none/direct/contextual edge semantics.",
                    }
                    write_dataset(root, name, x, y, src, dst, ea, meta, args.overwrite)
                    rows.append(meta)
                    print(f"[ok] {name}: E={src.shape[0]} hom={meta['homophily_observed']:.3f} "
                          f"corr0={stats['attr0_corr_same']:.3f} corr0*role={stats['attr0_x_role_corr_same']:.3f}")
    with open(root / f"{args.prefix}_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "role_feature_columns"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
