#!/usr/bin/env python3
"""Non-deep reference floors for every clustering table: KMeans(X), spectral(A), Louvain(A), SGC(A,X)+KMeans.

Any deep model that does not beat these on a dataset has not learned anything the dataset did not give
away for free. Rows prefixed `oracle_` use label knowledge (which edge channel carries the signal and
its sign, plus the latent role for contextual edges); they bound how much edge information could help
and are never a fair baseline.
"""
import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def load_graph(root: Path, name: str):
    d = root / name
    if (d / f"{name}_feat.npy").exists():
        x = np.load(d / f"{name}_feat.npy").astype(np.float64)
        y = np.load(d / f"{name}_label.npy").astype(np.int64)
        ei = np.load(d / f"{name}_edge_index.npy").astype(np.int64)
        ea_path = d / f"{name}_edge_attr.npy"
        ea = np.load(ea_path).astype(np.float64) if ea_path.exists() else None
        meta_path = d / f"{name}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        mapping = meta.get("label_mapping") or {}
        if "unknown" in mapping:
            # Urban graphs encode unlabeled plots as an explicit class; score on known labels only.
            y = np.where(y == int(mapping["unknown"]), -1, y)
        return x, y, ei, ea, meta
    from torch_geometric.datasets import Amazon, Planetoid
    key = name.lower()
    if key in ("cora", "citeseer", "pubmed"):
        ds = Planetoid(str(root), name={"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed"}[key])
    elif key in ("photo", "computers"):
        ds = Amazon(str(root), name="Photo" if key == "photo" else "Computers")
    else:
        raise FileNotFoundError(name)
    g = ds[0]
    return g.x.numpy().astype(np.float64), g.y.numpy(), g.edge_index.numpy(), None, {}


def sym_adj(n, ei, w=None):
    w = np.ones(ei.shape[1]) if w is None else w
    a = sp.coo_matrix((w, (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = (a + a.T) / 2.0
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def norm_adj(a, self_loops):
    if self_loops:
        a = a + sp.eye(a.shape[0])
    deg = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    return (dinv @ a @ dinv).tocsr()


def kmeans(z, k, seed):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(z)


def spectral(a, k, seed):
    _, vec = eigsh(norm_adj(a, self_loops=True), k=k, which="LA")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True).clip(1e-12)
    return kmeans(vec, k, seed)


def sgc(a, x, hops=2):
    s = norm_adj(a, self_loops=True)
    h = StandardScaler().fit_transform(x)
    for _ in range(hops):
        h = s @ h
    return h


def louvain(a, seed):
    comms = nx.community.louvain_communities(nx.from_scipy_sparse_array(a), seed=seed)
    lab = np.zeros(a.shape[0], dtype=np.int64)
    for ci, c in enumerate(comms):
        lab[list(c)] = ci
    return lab


def modularity(a, lab):
    m2 = a.sum()
    deg = np.asarray(a.sum(1)).ravel()
    q = 0.0
    for c in np.unique(lab):
        idx = lab == c
        q += a[idx][:, idx].sum() / m2 - (deg[idx].sum() / m2) ** 2
    return float(q)


def score(y, lab, a):
    known = y >= 0
    return {
        "nmi": float(normalized_mutual_info_score(y[known], lab[known])),
        "ari": float(adjusted_rand_score(y[known], lab[known])),
        "modularity": modularity(a, lab),
        "n_clusters": int(np.unique(lab).size),
    }


def run_dataset(root: Path, name: str, seed: int):
    x, y, ei, ea, meta = load_graph(root, name)
    n = x.shape[0]
    k = int(np.unique(y[y >= 0]).size)
    a = sym_adj(n, ei)
    out = []
    methods = {
        "kmeans_x": lambda: kmeans(StandardScaler().fit_transform(x), k, seed),
        "spectral_a": lambda: spectral(a, k, seed),
        "louvain_a": lambda: louvain(a, seed),
        "sgc2_ax": lambda: kmeans(sgc(a, x), k, seed),
    }
    if ea is not None and "edge_mode" in meta:
        # attr0 is the designed signal channel; contextual edges need the latent role to decode it.
        attr0 = ea[:, 0].copy()
        if meta["edge_mode"] == "contextual":
            rc = meta["role_feature_columns"]
            roles = x[:, rc].argmax(1)
            attr0 = attr0 * np.where(roles[ei[0]] == roles[ei[1]], 1.0, -1.0)
        w = 1.0 / (1.0 + np.exp(-3.0 * attr0))
        a_w = sym_adj(n, ei, w)
        methods["oracle_sgc2_ax_attr0"] = lambda: kmeans(sgc(a_w, x), k, seed)
    for method, fn in methods.items():
        lab = fn()
        out.append({"dataset": name, "method": method, "seed": seed, **score(y, lab, a)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            for r in run_dataset(Path(args.root_path), name, seed):
                rows.append(r)
                print(f"{r['dataset']:44s} {r['method']:22s} s{seed} nmi={r['nmi']:.3f} ari={r['ari']:.3f} "
                      f"mod={r['modularity']:.3f} k={r['n_clusters']}", flush=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
