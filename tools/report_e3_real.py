#!/usr/bin/env python3
"""Real-graph table for the README: dual-start DSE runs vs classic floors vs the pre-breakthrough numbers.

For each (model, dataset, seed) the runner trained `<model>_warm` and `<model>_nowarm`; the reported run is the
one with the lower final SE training loss (best_train_loss_mean), which needs no labels.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# 60-epoch numbers from results/mainline_evidence/ep_compare_v1 (README of 2026-04-16). Urban was scored with
# `unknown` as a class there, so that value is not comparable.
OLD_B45 = {"cora": 0.164, "photo": 0.002, "dblp_magnn_author_v2": 0.170, "entities_aifb": 0.082,
           "urban_paris_plot_v3bsjg": None}
CLASSIC = ("kmeans_x", "spectral_a", "louvain_a", "sgc2_ax")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_dir", default="results/breakthrough_v1/e3_real_dual")
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    runs = defaultdict(dict)
    for r in csv.DictReader(open(exp_dir / "runs.csv")):
        if r["status"] not in ("ok", "cached") or not r.get("nmi_mean"):
            continue
        model, start = r["tag"].rsplit("_", 1)
        runs[(model, r["dataset"], r["seed"])][start] = r
    classic = defaultdict(list)
    for r in csv.DictReader(open(exp_dir / "classic_baselines.csv")):
        classic[(r["method"], r["dataset"])].append((float(r["nmi"]), int(r["n_clusters"])))

    picked = defaultdict(list)
    detail = []
    for (model, ds, seed), starts in sorted(runs.items()):
        cands = {s: r for s, r in starts.items()}
        chosen = min(cands, key=lambda s: float(cands[s]["best_train_loss_mean"]))
        for s, r in cands.items():
            detail.append({"model": model, "dataset": ds, "seed": seed, "start": s, "chosen": s == chosen,
                           "nmi": float(r["nmi_mean"]), "best_train_loss": float(r["best_train_loss_mean"]),
                           "modularity": float(r["modularity_mean"]), "warm_pseudo_nmi": r.get("warm_pseudo_nmi", "")})
        picked[(model, ds)].append((float(cands[chosen]["nmi_mean"]), chosen))
        for s in cands:
            picked[(model + f"[{s}]", ds)].append((float(cands[s]["nmi_mean"]), s))
    with open(exp_dir / "dual_start_detail.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    def cell(key):
        vals = picked.get(key, [])
        if not vals:
            return "—"
        nmi = [v for v, _ in vals]
        starts = sorted({s for _, s in vals})
        tag = "" if "[" in key[0] else f" ({'/'.join(starts)})"
        return f"{np.mean(nmi):.3f}±{np.std(nmi):.3f}{tag}" if len(nmi) > 1 else f"{nmi[0]:.3f}{tag}"

    datasets = ["cora", "photo", "dblp_magnn_author_v2", "entities_aifb", "urban_paris_plot_v3bsjg"]
    md = ["| dataset | old B45 (60 ep) | " + " | ".join(CLASSIC) + " | baseline (SE-selected) | B45 (SE-selected) | baseline warm / nowarm | B45 warm / nowarm |",
          "|---|---|" + "---|" * len(CLASSIC) + "---|---|---|---|"]
    for ds in datasets:
        cl = []
        for m in CLASSIC:
            v = classic.get((m, ds))
            cl.append(f"{v[0][0]:.3f}" + (f" (k={v[0][1]})" if m == "louvain_a" else "") if v else "—")
        old = OLD_B45.get(ds)
        md.append(f"| {ds} | {'n/a' if old is None else f'{old:.3f}'} | " + " | ".join(cl) + " | "
                  + f"{cell(('baseline', ds))} | {cell(('b45', ds))} | "
                  + f"{cell(('baseline[warm]', ds))} / {cell(('baseline[nowarm]', ds))} | "
                  + f"{cell(('b45[warm]', ds))} / {cell(('b45[nowarm]', ds))} |")
    (exp_dir / "real_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
