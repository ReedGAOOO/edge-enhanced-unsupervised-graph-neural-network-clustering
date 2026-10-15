#!/usr/bin/env python3
"""Compare label-free rules for choosing one partition among candidates from different methods.

Each instance (a JSON spec) lists candidate partitions of one graph: trained model runs (decoded from their
checkpoints) and classic clusterers (regularized spectral, SGC2+KMeans) at the class count or at the first model's
cluster count. Candidates carry a `family` (the method). Rules, fixed before evaluation and without tunable parameters:

  R0 se       lowest two-level structural entropy on the input graph (the rule E14/E16 used);
  R1 medoid   one candidate per family (the first listed); pick the one with the highest mean AMI to the others;
  R2 edge_cv  every candidate is recomputed on a training graph with 20% of edges held out (model runs retrained on it,
              given as `cv_run`); pick the lowest structural entropy on the held-out edges, then report that
              candidate's full-graph partition;
  R3 two_view rank by structural entropy on the input graph and on a 10-NN cosine graph of the (standardized,
              PCA-64) node features; pick the lowest rank sum, ties broken by the input-graph rank.
  R4 gated_cv defined after the E18 results, so its E18 score is not evidence: the feature view is trusted only if a
              feature-only partition (KMeans on the same features, class count) compresses the held-out edges better
              than a random partition of the same sizes (gate r > 0, where r = (SE_random - SE_features) /
              (SE_random - min candidate SE), all on held-out edges). If so, pick the lowest rank sum of held-out SE
              and feature-graph SE; otherwise use R2. Validate on instances not used to design it.

Labels only score the picks: NMI on labeled nodes, the best candidate's NMI (oracle) and the regret. Output: one row
per instance and a summary per rule and instance set.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from inspect_edge_factors import configs_from_cmd  # noqa: E402
from run_classic_baselines import kmeans, load_graph, sgc, spectral_reg, sym_adj  # noqa: E402
from utils.warm_start import structural_entropy_2level as se2  # noqa: E402

RULES = ("se", "medoid", "edge_cv", "two_view", "gated_cv")
_DECODED = {}
_GRAPHS = {}
_FEATURE_GRAPHS = {}


def decode(run_dir, root):
    run_dir = str(run_dir)
    if run_dir in _DECODED:
        return _DECODED[run_dir]
    from data import load_data
    from exp import Exp
    cfg = configs_from_cmd((Path(run_dir) / "stdout.txt").read_text().splitlines()[0])
    cfg.root_path = str(root)  # older runs recorded a machine-specific data path
    exp = Exp(cfg)
    data = load_data(cfg).to(exp.device)
    model = exp.build_model(data)
    model.load_state_dict(torch.load(REPO / "checkpoints" / cfg.save_path, map_location=exp.device))
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        emb, clu = model.get_cluster_results(data)
        pred = model.fix_cluster_results(clu[1], emb, cfg.epsInt).cpu().numpy()
    _DECODED[run_dir] = pred
    return pred


def graph(root, name):
    if name not in _GRAPHS:
        x, y, ei, _, _ = load_graph(root, name)
        w_path = root / name / f"{name}_edge_weight.npy"
        w = np.load(w_path).astype(np.float64) if w_path.exists() else None
        _GRAPHS[name] = (x, y, sym_adj(x.shape[0], ei, w))
    return _GRAPHS[name]


def feature_graph(name, x, k=10, dim=64):
    if name not in _FEATURE_GRAPHS:
        z = StandardScaler().fit_transform(x)
        if z.shape[1] > dim:
            z = PCA(dim, random_state=0).fit_transform(z)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(z)
        idx = nn.kneighbors(z, return_distance=False)[:, 1:]
        src = np.repeat(np.arange(x.shape[0]), k)
        _FEATURE_GRAPHS[name] = sym_adj(x.shape[0], np.stack([src, idx.ravel()]))
    return _FEATURE_GRAPHS[name]


def heldout_graph(root, cv_name, n):
    z = np.load(root / cv_name / f"{cv_name}_heldout.npz")
    return sym_adj(n, z["pairs"], z["weight"])


def partitions(cands, a, x, k_cls, use_cv, root):
    """Partitions for every candidate on graph `a`; model candidates use their full or cross-validation run."""
    out = {}
    first_model = next(c for c in cands if c["kind"] == "model")
    k_anchor = None
    for c in cands:
        if c["kind"] == "model":
            out[c["name"]] = decode(c["cv_run"] if use_cv else c["run"], root)
            if c is first_model:
                k_anchor = int(np.unique(out[c["name"]]).size)
    for c in cands:
        if c["kind"] == "model":
            continue
        k = k_cls if c["k"] == "kcls" else k_anchor
        out[c["name"]] = spectral_reg(a, k, 0) if c["kind"] == "spectral" else kmeans(sgc(a, x), k, 0)
    return out


def evaluate(inst, root):
    x, y, a = graph(root, inst["dataset"])
    known = y >= 0
    k_cls = int(np.unique(y[known]).size)
    cands = inst["candidates"]
    full = partitions(cands, a, x, k_cls, use_cv=False, root=root)
    names = [c["name"] for c in cands]
    nmi = {n: normalized_mutual_info_score(y[known], full[n][known]) for n in names}
    se_a = {n: se2(a, full[n]) for n in names}
    gx = feature_graph(inst.get("feature_key", inst["dataset"]), x)
    se_x = {n: se2(gx, full[n]) for n in names}
    picks = {"se": min(names, key=se_a.get)}
    reps, seen = [], set()
    for c in cands:
        if c["family"] not in seen:
            seen.add(c["family"])
            reps.append(c["name"])
    agree = {r: np.mean([adjusted_mutual_info_score(full[r], full[o]) for o in reps if o != r]) for r in reps}
    picks["medoid"] = max(reps, key=agree.get)
    rank_a = {n: i for i, n in enumerate(sorted(names, key=se_a.get))}
    rank_x = {n: i for i, n in enumerate(sorted(names, key=se_x.get))}
    picks["two_view"] = min(names, key=lambda n: (rank_a[n] + rank_x[n], rank_a[n]))
    se_cv = {}
    if inst.get("cv_dataset") and all(c.get("cv_run") for c in cands if c["kind"] == "model"):
        _, _, a_train = graph(root, inst["cv_dataset"])
        a_ho = heldout_graph(root, inst["cv_dataset"], x.shape[0])
        cv = partitions(cands, a_train, x, k_cls, use_cv=True, root=root)
        se_cv = {n: se2(a_ho, cv[n]) for n in names}
        picks["edge_cv"] = min(names, key=se_cv.get)
        f = StandardScaler().fit_transform(x)
        if f.shape[1] > 64:
            f = PCA(64, random_state=0).fit_transform(f)
        feat_lab = kmeans(f, k_cls, 0)
        se_feat, se_rand = se2(a_ho, feat_lab), se2(a_ho, np.random.default_rng(0).permutation(feat_lab))
        gate = (se_rand - se_feat) / max(se_rand - min(se_cv.values()), 1e-9)
        rank_cv = {n: i for i, n in enumerate(sorted(names, key=se_cv.get))}
        picks["gated_cv"] = (min(names, key=lambda n: (rank_cv[n] + rank_x[n], rank_cv[n])) if gate > 0
                             else picks["edge_cv"])
    oracle = max(nmi.values())
    row = {"id": inst["id"], "set": inst["set"], "dataset": inst["dataset"], "seed": inst["seed"],
           "oracle_nmi": round(oracle, 4), "oracle": max(names, key=nmi.get)}
    if se_cv:
        row["feature_gate_r"] = round(float(gate), 4)
    for r in RULES:
        if r in picks:
            row[f"{r}_pick"] = picks[r]
            row[f"{r}_nmi"] = round(nmi[picks[r]], 4)
    for n in names:
        row[f"nmi[{n}]"] = round(nmi[n], 4)
        row[f"se[{n}]"] = round(se_a[n], 4)
        row[f"sex[{n}]"] = round(se_x[n], 4)
        if n in se_cv:
            row[f"secv[{n}]"] = round(se_cv[n], 4)
    return row


def summarize(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["set"]].append(r)
        groups["all"].append(r)
        if "edge_cv_nmi" in r:
            groups[f"{r['set']}|cv-subset"].append(r)
            groups["all|cv-subset"].append(r)
    for g, rs in sorted(groups.items()):
        for rule in RULES:
            vals = [r for r in rs if f"{rule}_nmi" in r]
            if not vals:
                continue
            reg = [r["oracle_nmi"] - r[f"{rule}_nmi"] for r in vals]
            out.append({"group": g, "rule": rule, "instances": len(vals),
                        "mean_nmi": round(float(np.mean([r[f"{rule}_nmi"] for r in vals])), 4),
                        "mean_oracle": round(float(np.mean([r["oracle_nmi"] for r in vals])), 4),
                        "mean_regret": round(float(np.mean(reg)), 4), "max_regret": round(float(np.max(reg)), 4),
                        "regret_over_0.05": int(sum(x > 0.05 for x in reg))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--spec", required=True, help="JSON file with an 'instances' list.")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    root = Path(args.root_path)
    insts = json.loads(Path(args.spec).read_text())["instances"]
    rows = []
    for inst in insts:
        row = evaluate(inst, root)
        rows.append(row)
        print({k: v for k, v in row.items() if not k.startswith(("nmi[", "se[", "sex[", "secv["))}, flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out / "instances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s, flush=True)


if __name__ == "__main__":
    main()
