#!/usr/bin/env python3
"""Choose between the structure route and the feature route with the labeled nodes themselves (E22).

With m labeled nodes (data.choose_labeled_nodes, the same draws as E21), two few-label routes are compared:
  structure  `se_top_label` of tools/few_label_edge_semantics.py: the K lowest-SE weightings of an SE-guided search
             (plus uniform), each clustered by regularized spectral clustering and SGC+KMeans; the labels pick one by ARI;
  feature    `node_clf`: logistic regression on SGC(A, X) features (semi-supervised node classification).
The route is chosen by 5-fold cross-validation inside the labeled set: the structure route picks its partition on the
training fold and maps each cluster to the training fold's majority label (clusters without training labels get the
fold's majority label); the feature route trains on the training fold. Higher held-out accuracy wins (ties go to the
structure route), and the winner is refit on all m labels. NMI is scored on the remaining labeled nodes. Reported:
each route alone, the chosen route, and the better of the two per draw (an oracle, for regret).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from data import choose_labeled_nodes  # noqa: E402
from few_label_edge_semantics import cluster_pair, pick_by_labels, top_k_weightings  # noqa: E402
from run_classic_baselines import load_graph, sgc, sym_adj  # noqa: E402
from write_se_reweighted_dataset import pick_weight  # noqa: E402


def feature_route(h, y, train):
    if np.unique(y[train]).size < 2:
        return np.full(h.shape[0], y[train][0])
    return LogisticRegression(max_iter=3000).fit(h[train], y[train]).predict(h)


def structure_route(pool, y, train):
    return pool[pick_by_labels(pool, y, train)]


def map_clusters(part, y, train):
    """Predict labels from a partition: each cluster takes the majority training label inside it."""
    default = Counter(y[train]).most_common(1)[0][0]
    lut = {}
    for c in np.unique(part):
        inside = y[train][part[train] == c]
        lut[c] = Counter(inside).most_common(1)[0][0] if inside.size else default
    return np.array([lut[c] for c in part])


def cv_accuracy(route, h, pool, y, lab, seed):
    folds = KFold(min(5, lab.size), shuffle=True, random_state=seed)
    accs = []
    for tr, te in folds.split(lab):
        train, test = lab[tr], lab[te]
        if route == "feature":
            pred = feature_route(h, y, train)
        else:
            pred = map_clusters(structure_route(pool, y, train), y, train)
        accs.append(float((pred[test] == y[test]).mean()))
    return float(np.mean(accs))


def run_dataset(job):
    name, cand_csv, ctx_dim, budgets, draws, top_k = job
    x, y, ei, ea_native, meta = load_graph(REPO / "data", name)
    n = x.shape[0]
    known = y >= 0
    k = int(np.unique(y[known]).size)
    h = sgc(sym_adj(n, ei), x)
    pool = {}
    for tag in top_k_weightings(cand_csv, top_k):
        w = pick_weight(tag, x, ei, ea_native, meta, ctx_dim)
        for meth, p in cluster_pair(sym_adj(n, ei, w), x, k).items():
            pool[f"{tag}/{meth}"] = p
    rows = []
    for m in budgets:
        for r in draws:
            lab = choose_labeled_nodes(torch.from_numpy(y), m, r).numpy()
            if lab.size < m:
                continue
            ev = known.copy()
            ev[lab] = False
            nmi = lambda p: normalized_mutual_info_score(y[ev], p[ev])  # noqa: E731
            s_out, f_out = structure_route(pool, y, lab), feature_route(h, y, lab)
            s_nmi, f_nmi = nmi(s_out), nmi(f_out)
            s_cv, f_cv = cv_accuracy("structure", h, pool, y, lab, r), cv_accuracy("feature", h, pool, y, lab, r)
            choice = "structure" if s_cv >= f_cv else "feature"
            auto = s_nmi if choice == "structure" else f_nmi
            rows.append({"dataset": name, "budget": m, "draw": r, "structure_nmi": round(s_nmi, 4),
                         "feature_nmi": round(f_nmi, 4), "structure_cv_acc": round(s_cv, 4), "feature_cv_acc": round(f_cv, 4),
                         "choice": choice, "auto_nmi": round(auto, 4), "oracle_nmi": round(max(s_nmi, f_nmi), 4)})
            print(rows[-1], flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True, help="CSV with columns dataset,candidates_csv,ctx_dim")
    parser.add_argument("--budgets", default="10,20,50,100,200")
    parser.add_argument("--draws", default="0,1,2,3,4")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    draws = [int(d) for d in args.draws.split(",")]
    jobs = [(r["dataset"], r["candidates_csv"], int(r["ctx_dim"]), budgets, draws, args.top_k)
            for r in csv.DictReader(open(args.jobs))]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for part in ex.map(run_dataset, jobs):
            rows += part
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "draws.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for m in budgets:
        for scope in ("all", "synthetic", "real"):
            rs = [r for r in rows if r["budget"] == m and (scope == "all" or (scope == "synthetic") == r["dataset"].startswith("synth"))]
            if not rs:
                continue
            rec = {"budget": m, "scope": scope, "draws": len(rs)}
            for key in ("structure", "feature", "auto", "oracle"):
                rec[f"{key}_nmi"] = round(float(np.mean([r[f"{key}_nmi"] for r in rs])), 4)
            rec["auto_regret"] = round(float(np.mean([r["oracle_nmi"] - r["auto_nmi"] for r in rs])), 4)
            rec["auto_regret_over_0.05"] = round(float(np.mean([r["oracle_nmi"] - r["auto_nmi"] > 0.05 for r in rs])), 4)
            rec["structure_chosen"] = round(float(np.mean([r["choice"] == "structure" for r in rs])), 4)
            summary.append(rec)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s, flush=True)


if __name__ == "__main__":
    main()
