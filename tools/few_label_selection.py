#!/usr/bin/env python3
"""How many labels does it take to choose well among candidate partitions?

For every decision in one or more selection specs (the E18/E19 format of tools/select_partition_rules.py), the
candidates' full-graph partitions are rebuilt once (and saved to partitions/<id>.npz). Then, for each label budget m,
`--draws` random sets of m labeled nodes are drawn uniformly from the labeled nodes; each candidate is scored by the
adjusted Rand index between its clusters and the labels on those m nodes, and the best is picked (ties count as the
mean over the tied candidates). The pick is compared with the best candidate on all labeled nodes, in NMI (the metric
reported elsewhere) and in ARI (the metric the selection uses; NMI regret can persist when the ARI-best and NMI-best
candidates differ). Selection is the only use of the m labels; no model is trained on them. Label-free picks (SE, edge CV) are merged from the
rule-benchmark outputs given with --reference for comparison.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_partition_rules import graph, partitions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[], help="instances.csv files from select_partition_rules.py")
    parser.add_argument("--budgets", default="5,10,20,50,100")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    root = Path(args.root_path)
    out = Path(args.out_dir)
    (out / "partitions").mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]
    ref = {}
    for path in args.reference:
        for r in csv.DictReader(open(path)):
            ref[r["id"]] = r
    rows = []
    for spec in args.specs:
        for inst in json.loads(Path(spec).read_text())["instances"]:
            x, y, a = graph(root, inst["dataset"])
            known = np.where(y >= 0)[0]
            k_cls = int(np.unique(y[known]).size)
            cache = out / "partitions" / (inst["id"].replace("/", "__") + ".npz")
            if cache.exists():
                z = np.load(cache)
                parts = {n: z[n] for n in z.files}
            else:
                parts = partitions(inst["candidates"], a, x, k_cls, use_cv=False, root=root)
                np.savez(cache, **parts)
            names = list(parts)
            nmi = np.array([normalized_mutual_info_score(y[known], parts[n][known]) for n in names])
            full_ari = np.array([adjusted_rand_score(y[known], parts[n][known]) for n in names])
            oracle = float(nmi.max())
            rng = np.random.default_rng(args.seed)
            base = {"id": inst["id"], "set": inst["set"], "spec": Path(spec).stem, "dataset": inst["dataset"],
                    "seed": inst["seed"], "n_labeled": int(known.size), "oracle_nmi": round(oracle, 4)}
            r0 = ref.get(inst["id"], {})
            for rule in ("se", "edge_cv"):
                if r0.get(f"{rule}_nmi"):
                    base[f"{rule}_regret"] = round(oracle - float(r0[f"{rule}_nmi"]), 4)
                    base[f"{rule}_ari_regret"] = round(float(full_ari.max() - full_ari[names.index(r0[f"{rule}_pick"])]), 4)
            for m in budgets:
                if m > known.size:
                    continue
                regrets, ari_regrets = [], []
                for _ in range(args.draws):
                    idx = rng.choice(known, m, replace=False)
                    score = np.array([adjusted_rand_score(y[idx], parts[n][idx]) for n in names])
                    tied = np.isclose(score, score.max())
                    regrets.append(oracle - float(nmi[tied].mean()))
                    ari_regrets.append(float(full_ari.max() - full_ari[tied].mean()))
                regrets, ari_regrets = np.array(regrets), np.array(ari_regrets)
                rows.append({**base, "budget": m, "mean_regret": round(float(regrets.mean()), 4),
                             "p_regret_over_0.05": round(float((regrets > 0.05).mean()), 4),
                             "p90_regret": round(float(np.quantile(regrets, 0.9)), 4),
                             "mean_ari_regret": round(float(ari_regrets.mean()), 4),
                             "p_ari_regret_over_0.05": round(float((ari_regrets > 0.05).mean()), 4)})
            print(inst["id"], "done", flush=True)
    with open(out / "instances.csv", "w", newline="") as fh:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        for g in ("all", r["spec"], f"{r['spec']}|{r['set']}"):
            groups[(g, r["budget"])].append(r)
    summary = []
    for (g, m), rs in sorted(groups.items()):
        rec = {"group": g, "budget": m, "decisions": len(rs),
               "mean_regret": round(float(np.mean([r["mean_regret"] for r in rs])), 4),
               "mean_ari_regret": round(float(np.mean([r["mean_ari_regret"] for r in rs])), 4),
               "mean_p_regret_over_0.05": round(float(np.mean([r["p_regret_over_0.05"] for r in rs])), 4),
               "decisions_mean_regret_over_0.05": int(sum(r["mean_regret"] > 0.05 for r in rs))}
        for rule in ("se", "edge_cv"):
            vals = [r[f"{rule}_regret"] for r in rs if f"{rule}_regret" in r]
            if vals:
                rec[f"{rule}_mean_regret_same_decisions"] = round(float(np.mean(vals)), 4)
                rec[f"{rule}_mean_ari_regret_same_decisions"] = round(float(np.mean([r[f"{rule}_ari_regret"] for r in rs if f"{rule}_ari_regret" in r])), 4)
                rec[f"{rule}_decisions"] = len(vals)
        summary.append(rec)
    with open(out / "summary.csv", "w", newline="") as fh:
        fields = []
        for r in summary:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        if s["group"] == "all" or "|" not in s["group"]:
            print(s, flush=True)


if __name__ == "__main__":
    main()
