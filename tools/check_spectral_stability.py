#!/usr/bin/env python3
"""How much does a spectral partition depend on the eigensolver's start vector?

For each dataset (uniform weights) plain and regularized spectral clustering are run with several eigsh start vectors
and a fixed KMeans seed. A spread across start vectors means the partition is not a property of the graph, so a
label-free search that scores spectral partitions can pick different weightings on different machines. The number
of connected components is reported because more components than clusters makes plain spectral degenerate.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classic_baselines import load_graph, spectral, spectral_reg, sym_adj  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--starts", type=int, default=5, help="Number of eigsh start vectors.")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        x, y, ei, _, _ = load_graph(Path(args.root_path), name)
        a = sym_adj(x.shape[0], ei)
        known = y >= 0
        k = int(np.unique(y[known]).size)
        n_comp = connected_components(a, directed=False)[0]
        row = {"dataset": name, "n": x.shape[0], "k": k, "components": int(n_comp)}
        for method, fn in (("plain", spectral), ("reg", spectral_reg)):
            nmis = []
            for s in range(args.starts):
                v0 = np.random.default_rng(1000 + s).normal(size=x.shape[0])
                lab = fn(a, k, 0, v0=v0)
                nmis.append(normalized_mutual_info_score(y[known], lab[known]))
            row.update({f"{method}_nmi_mean": round(float(np.mean(nmis)), 4), f"{method}_nmi_std": round(float(np.std(nmis)), 4),
                        f"{method}_nmi_min": round(float(np.min(nmis)), 4), f"{method}_nmi_max": round(float(np.max(nmis)), 4)})
        rows.append(row)
        print(row, flush=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
