#!/usr/bin/env python3
"""Few-label use of edge semantics without training a DSE (E21 methods 0-2).

The m labeled nodes are drawn with data.choose_labeled_nodes (the same draw as main.py --label_budget m
--label_seed r) and NMI is scored on the remaining labeled nodes only.

  m0 node_clf      logistic regression on SGC(A, X) features: semi-supervised classification, the baseline any
                   few-label clustering has to beat;
  m1 se_top_label  the K lowest-SE weightings of an SE-guided search (tools/run_se_guided_baseline.py candidate CSV,
                   uniform added), each clustered by regularized spectral clustering and SGC+KMeans; the labeled nodes
                   pick one by ARI;
  m2 edge_clf      a same-label classifier on edges whose endpoints are both labeled, fed with edge attributes and
                   endpoint context (logistic regression `lin`, one-hidden-layer MLP `mlp`); P(same) reweights every
                   edge (clipped to [0.01, 1]) and the labels pick spectral or SGC on that graph. Too few labeled edges
                   (< 10 or one class) falls back to the uniform graph;
  m2 edge_clf_pseudo  the MLP trained on labeled edges plus edges whose endpoints m0 predicts confidently
                   (product of top probabilities >= 0.6, at most 50k edges, same/different by predicted class).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from data import choose_labeled_nodes  # noqa: E402
from run_classic_baselines import kmeans, load_graph, sgc, spectral_reg, sym_adj  # noqa: E402
from run_se_guided_baseline import endpoint_context  # noqa: E402
from screen_edge_headroom import generic_edge_attr  # noqa: E402
from write_se_reweighted_dataset import pick_weight  # noqa: E402


def cluster_pair(a, x, k):
    return {"spectral": spectral_reg(a, k, 0), "sgc": kmeans(sgc(a, x), k, 0)}


def pick_by_labels(parts, y, lab_idx):
    scores = {n: adjusted_rand_score(y[lab_idx], p[lab_idx]) for n, p in parts.items()}
    return max(scores, key=scores.get)


def top_k_weightings(cand_csv, k):
    best = {}
    for r in csv.DictReader(open(cand_csv)):
        best[r["weighting"]] = min(best.get(r["weighting"], np.inf), float(r["se"]))
    tags = sorted(best, key=best.get)[:k]
    return tags if "uniform" in tags else tags + ["uniform"]


def run_dataset(job):
    name, cand_csv, ctx_dim, budgets, draws, top_k = job
    x, y, ei, ea_native, meta = load_graph(REPO / "data", name)
    n = x.shape[0]
    known = y >= 0
    k = int(np.unique(y[known]).size)
    ea = ea_native if ea_native is not None else generic_edge_attr(x, ei)
    ctx = endpoint_context(x, ei, ctx_dim)
    feats = StandardScaler().fit_transform(np.hstack([ea, ctx]))
    a_uni = sym_adj(n, ei)
    h = sgc(a_uni, x)
    uni_parts = cluster_pair(a_uni, x, k)
    pool = {}
    for tag in top_k_weightings(cand_csv, top_k):
        w = pick_weight(tag, x, ei, ea_native, meta, ctx_dim)
        for meth, p in cluster_pair(sym_adj(n, ei, w), x, k).items():
            pool[f"{tag}/{meth}"] = p
    src, dst = ei
    rows = []
    for m in budgets:
        for r in draws:
            lab = choose_labeled_nodes(torch.from_numpy(y), m, r).numpy()
            if lab.size < m:
                continue
            ev = known.copy()
            ev[lab] = False
            score = lambda p: normalized_mutual_info_score(y[ev], p[ev])  # noqa: E731
            rec = {"dataset": name, "budget": m, "draw": r, "n_eval": int(ev.sum())}
            if np.unique(y[lab]).size >= 2:
                clf = LogisticRegression(max_iter=3000).fit(h[lab], y[lab])
                probs = clf.predict_proba(h)
                pred = clf.classes_[probs.argmax(1)]
            else:
                probs, pred = np.ones((n, 1)), np.full(n, y[lab][0])
            rec["node_clf"] = score(pred)
            pk = pick_by_labels(pool, y, lab)
            rec["se_top_label"], rec["se_top_label_pick"] = score(pool[pk]), pk
            is_lab = np.zeros(n, dtype=bool)
            is_lab[lab] = True
            e_lab = is_lab[src] & is_lab[dst]
            target = (y[src] == y[dst]).astype(int)
            rec["labeled_edges"] = int(e_lab.sum())
            for tag, make in (("lin", lambda: LogisticRegression(max_iter=3000)),
                              ("mlp", lambda: MLPClassifier((64,), max_iter=400, random_state=r))):
                if e_lab.sum() >= 10 and np.unique(target[e_lab]).size == 2:
                    model = make().fit(feats[e_lab], target[e_lab])
                    w = np.clip(model.predict_proba(feats)[:, 1], 0.01, 1.0)
                    parts = cluster_pair(sym_adj(n, ei, w), x, k)
                    rec[f"edge_clf_{tag}_fallback"] = 0
                else:
                    parts = uni_parts
                    rec[f"edge_clf_{tag}_fallback"] = 1
                rec[f"edge_clf_{tag}"] = score(parts[pick_by_labels(parts, y, lab)])
            # pseudo-labeled edges from the node classifier
            conf = probs.max(1)
            pc = conf[src] * conf[dst] >= 0.6
            pseudo_t = (pred[src] == pred[dst]).astype(int)
            train = np.where(pc & ~e_lab)[0]
            rng = np.random.default_rng(r)
            if train.size > 50000:
                train = rng.choice(train, 50000, replace=False)
            idx = np.concatenate([np.where(e_lab)[0], train])
            t = np.concatenate([target[e_lab], pseudo_t[train]])
            if idx.size >= 10 and np.unique(t).size == 2:
                model = MLPClassifier((64,), max_iter=200, early_stopping=idx.size >= 200, random_state=r).fit(feats[idx], t)
                w = np.clip(model.predict_proba(feats)[:, 1], 0.01, 1.0)
                parts = cluster_pair(sym_adj(n, ei, w), x, k)
                rec["edge_clf_pseudo_fallback"] = 0
            else:
                parts = uni_parts
                rec["edge_clf_pseudo_fallback"] = 1
            rec["edge_clf_pseudo"] = score(parts[pick_by_labels(parts, y, lab)])
            rec["pseudo_edges"] = int(train.size)
            rows.append({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rec.items()})
            print({k_: rec[k_] for k_ in ("dataset", "budget", "draw", "labeled_edges", "node_clf", "se_top_label",
                                           "edge_clf_lin", "edge_clf_mlp", "edge_clf_pseudo")}, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True, help="CSV with columns dataset,candidates_csv,ctx_dim")
    parser.add_argument("--budgets", default="10,20,50,100,200")
    parser.add_argument("--draws", default="0,1,2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    draws = [int(d) for d in args.draws.split(",")]
    jobs = [(r["dataset"], r["candidates_csv"], int(r["ctx_dim"]), budgets, draws, args.top_k)
            for r in csv.DictReader(open(args.jobs))]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(run_dataset, jobs):
            rows += part
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
