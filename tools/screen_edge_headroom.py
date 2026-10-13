#!/usr/bin/env python3
"""Screen a dataset for edge-attribute headroom before spending GPU time on edge-aware models.

1. On edges whose endpoints are both labeled, cross-fitted logistic regression predicts "same label" from
   (a) edge_attr, (b) endpoint feature cosine only, (c) both. AUC(c) - AUC(b) is what edge_attr adds beyond
   node features.
2. The edge_attr model's probability reweights every edge; SGC2+KMeans and spectral clustering on the weighted
   graph are compared with the unweighted graph. The mapping is fitted with labels, so the NMI gain is an upper
   bound on what context-free edge reweighting can buy. A dataset with ~0 headroom cannot reward an edge-aware model.

Datasets without native edge_attr get the generic attributes data.py builds (cosine, L2, L1, log-degrees, gap).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classic_baselines import kmeans, load_graph, score, sgc, spectral, sym_adj  # noqa: E402


def rowwise(fn, x, src, dst, chunk=100000):
    """Apply fn(x[src_chunk], x[dst_chunk]) in chunks; materialising x[src] at once can need tens of GB."""
    return np.concatenate([fn(x[src[i:i + chunk]], x[dst[i:i + chunk]]) for i in range(0, src.shape[0], chunk)])


def generic_edge_attr(x, ei):
    src, dst = ei
    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    cos = rowwise(lambda a, b: np.sum(a * b, axis=1), xn, src, dst)
    l2 = rowwise(lambda a, b: np.linalg.norm(a - b, axis=1), x, src, dst)
    l1 = rowwise(lambda a, b: np.abs(a - b).mean(1), x, src, dst)
    deg = np.bincount(src, minlength=x.shape[0]).astype(float)
    ls, ld = np.log1p(deg[src]), np.log1p(deg[dst])
    return np.stack([cos, l2, l1, ls, ld, np.abs(ls - ld)], 1)


def cross_fit_proba(feats, target, mask, seed, max_fit=200000):
    """Out-of-fold P(same) on labeled edges; unlabeled edges get the mean of the fold models."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    if idx.size > max_fit:
        idx = rng.choice(idx, max_fit, replace=False)
    scaler = StandardScaler().fit(feats[idx])
    z = scaler.transform(feats)
    proba = np.zeros(feats.shape[0])
    other = np.ones(feats.shape[0], dtype=bool)
    other[idx] = False
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in folds.split(idx, target[idx]):
        clf = LogisticRegression(max_iter=2000).fit(z[idx[tr]], target[idx[tr]])
        proba[idx[te]] = clf.predict_proba(z[idx[te]])[:, 1]
        if other.any():
            proba[other] += clf.predict_proba(z[other])[:, 1] / 5.0
    return proba, idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        x, y, ei, ea, meta = load_graph(Path(args.root_path), name)
        native = ea is not None
        if not native:
            ea = generic_edge_attr(x, ei)
        n = x.shape[0]
        k = int(np.unique(y[y >= 0]).size)
        src, dst = ei
        labeled = (y[src] >= 0) & (y[dst] >= 0)
        same = (y[src] == y[dst]).astype(int)
        if labeled.sum() < 200:
            print(f"{name:40s} only {int(labeled.sum())} edges with both endpoints labeled; headroom not estimable", flush=True)
            rows.append({"dataset": name, "native_edge_attr": native, "edge_attr_dim": ea.shape[1],
                         "labeled_edge_frac": float(labeled.mean())})
            continue
        xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
        cos = rowwise(lambda a, b: np.sum(a * b, axis=1), xn, src, dst)[:, None]
        auc = {}
        for tag, feats in (("edge_attr", ea), ("endpoint_cos", cos), ("edge_attr+cos", np.hstack([ea, cos]))):
            p, idx = cross_fit_proba(feats, same, labeled, args.seed)
            auc[tag] = roc_auc_score(same[idx], p[idx])
            if tag == "edge_attr":
                w_edge = p
        a = sym_adj(n, ei)
        a_w = sym_adj(n, ei, np.clip(w_edge, 1e-3, None))
        res = {
            "sgc_plain": score(y, kmeans(sgc(a, x), k, args.seed), a)["nmi"],
            "sgc_weighted": score(y, kmeans(sgc(a_w, x), k, args.seed), a)["nmi"],
            "spectral_plain": score(y, spectral(a, k, args.seed), a)["nmi"],
            "spectral_weighted": score(y, spectral(a_w, k, args.seed), a)["nmi"],
        }
        row = {"dataset": name, "native_edge_attr": native, "edge_attr_dim": ea.shape[1],
               "labeled_edge_frac": float(labeled.mean()), "homophily_labeled": float(same[labeled].mean()),
               **{f"auc_{k_}": round(v, 4) for k_, v in auc.items()},
               "auc_gain_over_cos": round(auc["edge_attr+cos"] - auc["endpoint_cos"], 4),
               **{k_: round(v, 4) for k_, v in res.items()},
               "headroom_sgc": round(res["sgc_weighted"] - res["sgc_plain"], 4),
               "headroom_spectral": round(res["spectral_weighted"] - res["spectral_plain"], 4)}
        rows.append(row)
        print(f"{name:40s} native={native!s:5s} hom={row['homophily_labeled']:.3f} AUC ea={auc['edge_attr']:.3f} "
              f"cos={auc['endpoint_cos']:.3f} ea+cos={auc['edge_attr+cos']:.3f} | SGC {res['sgc_plain']:.3f}->{res['sgc_weighted']:.3f} "
              f"spectral {res['spectral_plain']:.3f}->{res['spectral_weighted']:.3f}", flush=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        fields = max((list(r.keys()) for r in rows), key=len)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
