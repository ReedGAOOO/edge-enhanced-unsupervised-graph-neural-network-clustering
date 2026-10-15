#!/usr/bin/env python3
"""Hold out a fraction of a graph's undirected edges for label-free edge cross-validation.

Writes <name><suffix>/ with the training graph (features, labels, the remaining directed edges and their edge_attr /
edge_weight rows) plus <name><suffix>_heldout.npz inside it with the held-out undirected pairs and their weights. A pair is held
out only while both endpoints keep at least one training edge, so the split creates no new isolated nodes. Partitions
computed on the training graph are then scored on the held-out edges (tools/select_partition_rules.py), which removes
the advantage a method gets from being scored on the very edges it optimised.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classic_baselines import load_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default="_cv80")
    args = parser.parse_args()
    root = Path(args.root_path)
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        x, y, ei, ea, meta = load_graph(root, name)
        w_path = root / name / f"{name}_edge_weight.npy"
        w = np.load(w_path).astype(np.float64) if w_path.exists() else None
        n = x.shape[0]
        src, dst = ei
        loop = src == dst
        key = np.minimum(src, dst) * n + np.maximum(src, dst)
        uniq, inv = np.unique(key[~loop], return_inverse=True)
        pu, pv = uniq // n, uniq % n
        deg = np.bincount(np.concatenate([pu, pv]), minlength=n)
        rng = np.random.default_rng(args.seed)
        target = int(round(args.frac * uniq.size))
        held = np.zeros(uniq.size, dtype=bool)
        for e in rng.permutation(uniq.size):
            if held.sum() >= target:
                break
            if deg[pu[e]] > 1 and deg[pv[e]] > 1:
                held[e] = True
                deg[pu[e]] -= 1
                deg[pv[e]] -= 1
        held_dir = np.zeros(ei.shape[1], dtype=bool)
        held_dir[np.where(~loop)[0]] = held[inv]
        keep = ~held_dir
        out_name = f"{name}{args.suffix}"
        out = root / out_name
        out.mkdir(parents=True, exist_ok=True)
        label_src = root / name / f"{name}_label.npy"
        np.save(out / f"{out_name}_feat.npy", x.astype(np.float32))
        np.save(out / f"{out_name}_label.npy", np.load(label_src) if label_src.exists() else y)
        np.save(out / f"{out_name}_edge_index.npy", ei[:, keep].astype(np.int64))
        if ea is not None:
            np.save(out / f"{out_name}_edge_attr.npy", ea[keep].astype(np.float32))
        if w is not None:
            np.save(out / f"{out_name}_edge_weight.npy", w[keep].astype(np.float32))
        # Held-out pair weight = mean over its directed copies (1 when the graph is unweighted).
        pair_w = np.ones(uniq.size)
        if w is not None:
            sums = np.bincount(inv, weights=w[~loop], minlength=uniq.size)
            cnts = np.bincount(inv, minlength=uniq.size)
            pair_w = sums / np.maximum(cnts, 1)
        np.savez(out / f"{out_name}_heldout.npz", pairs=np.stack([pu[held], pv[held]]), weight=pair_w[held])
        meta = dict(meta)
        meta.update({"dataset": out_name, "edge_cv_source": name, "edge_cv_frac": args.frac, "edge_cv_seed": args.seed})
        (out / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{out_name}: train pairs {int((~held).sum())}, held-out pairs {int(held.sum())} "
              f"({held.mean():.3f}), weighted={w is not None}", flush=True)


if __name__ == "__main__":
    main()
