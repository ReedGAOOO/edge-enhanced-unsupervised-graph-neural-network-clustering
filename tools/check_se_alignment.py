#!/usr/bin/env python3
"""Does minimising structural entropy (SE) over edge weights move a clustering towards or away from the labels?

For each candidate edge weighting w the script compares, on the graph weighted by w, the 2-level SE of the label
partition with the best of a few unsupervised partitions at the same k (spectral, SGC2+KMeans):
    label_gap = SE(label) - min SE(unsupervised)      (> 0: SE prefers a non-label partition)
Candidate weightings stay inside the learned-factor range of B31 (logclip 0.7, factors in [0.5, 2]):
  * uniform;
  * per edge-attribute channel c: factor exp(+-0.7 * tanh(z_c)), z_c the standardised channel;
  * multiplex graphs (meta channels named <layer>_member): one layer at 2.0, the others at 0.5, averaged over the
    layers an edge belongs to.
An SE-driven weight learner drifts towards the weighting with the lowest achievable SE. If labels are not preferred
there, learning edge weights under the SE objective moves away from the labels even when some weighting (with a
negative label_gap) would favour them. Datasets without native edge_attr use the generic attributes (cosine, L2, L1,
log-degrees, gap). Labels are used for the diagnosis only. Graphs with unlabeled nodes are skipped.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from utils.warm_start import structural_entropy_2level  # noqa: E402
from run_classic_baselines import kmeans, load_graph, sgc, spectral, sym_adj  # noqa: E402
from screen_edge_headroom import generic_edge_attr  # noqa: E402

LOGCLIP = 0.7
FIELDS = ["dataset", "native_edge_attr", "weighting", "se_label", "best_unsup", "se_best_unsup",
          "nmi_best_unsup", "label_gap", "se_preferred"]


def candidate_weightings(ea, meta, max_channels):
    yield "uniform", None
    chans = meta.get("edge_attr_channels") or [f"attr{i}" for i in range(ea.shape[1])]
    members = {c[:-7]: ea[:, i] > 0 for i, c in enumerate(chans) if c.endswith("_member") and i < ea.shape[1]}
    if members:
        n_layers = np.maximum(1.0, sum(m.astype(float) for m in members.values()))
        for up in members:
            w = sum((2.0 if layer == up else 0.5) * m for layer, m in members.items()) / n_layers
            yield f"layer+{up}", w
    for i in range(min(ea.shape[1], max_channels)):
        name = chans[i] if i < len(chans) else f"attr{i}"
        if name.endswith("_member"):
            continue
        z = (ea[:, i] - ea[:, i].mean()) / max(ea[:, i].std(), 1e-12)
        for sign in (1, -1):
            yield f"{'+' if sign > 0 else '-'}{name}", np.exp(sign * LOGCLIP * np.tanh(z))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--max_channels", type=int, default=8, help="Emphasise at most this many attribute channels.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        x, y, ei, ea, meta = load_graph(Path(args.root_path), name)
        if (y < 0).any():
            print(f"{name}: {int((y < 0).sum())} unlabeled nodes; the label partition's SE is undefined, skipped")
            continue
        native = ea is not None
        if not native:
            ea = generic_edge_attr(x, ei)
            meta = {"edge_attr_channels": ["cos", "l2", "l1", "logdeg_src", "logdeg_dst", "logdeg_gap"]}
        n, k = x.shape[0], int(np.unique(y).size)
        ds_rows = []
        for tag, w in candidate_weightings(ea, meta, args.max_channels):
            a = sym_adj(n, ei, w)
            parts = {"spectral": spectral(a, k, args.seed), "sgc": kmeans(sgc(a, x), k, args.seed)}
            se_unsup = {p: structural_entropy_2level(a, lab) for p, lab in parts.items()}
            best = min(se_unsup, key=se_unsup.get)
            se_label = structural_entropy_2level(a, y)
            ds_rows.append({"dataset": name, "native_edge_attr": native, "weighting": tag, "se_label": round(se_label, 4),
                            "best_unsup": best, "se_best_unsup": round(se_unsup[best], 4),
                            "nmi_best_unsup": round(normalized_mutual_info_score(y, parts[best]), 4),
                            "label_gap": round(se_label - se_unsup[best], 4)})
        lowest = min(ds_rows, key=lambda r: r["se_best_unsup"])
        for r in ds_rows:
            r["se_preferred"] = r is lowest
            print(f"{name:34s} {r['weighting']:16s} SE(label)={r['se_label']:.4f} best unsup {r['best_unsup']:8s} "
                  f"SE={r['se_best_unsup']:.4f} NMI={r['nmi_best_unsup']:.3f} gap={r['label_gap']:+.4f}"
                  + ("  <- lowest achievable SE" if r["se_preferred"] else ""), flush=True)
        favour = [r["weighting"] for r in ds_rows if r["label_gap"] < 0]
        print(f"{name:34s} SE-preferred weighting: {lowest['weighting']} (gap {lowest['label_gap']:+.4f}); "
              f"weightings where labels win: {', '.join(favour) or 'none'}", flush=True)
        rows += ds_rows
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
