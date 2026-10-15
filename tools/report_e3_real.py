#!/usr/bin/env python3
"""Real-graph table for the README: dual-start DSE runs vs classic floors vs the pre-breakthrough numbers.

For each (model, dataset, seed) the runner trained `<model>_warm` and `<model>_nowarm`. The start is chosen by the
edge-free baseline's best-checkpoint SE training loss for that (dataset, seed) and applied to every model;
no class identities are used for this choice.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# 60-epoch numbers from results/mainline_evidence/ep_compare_v1 (README of 2026-04-16). Urban was scored with
# `unknown` as a class there, so that value is not comparable.
OLD_B45 = {"cora": 0.164, "photo": 0.002, "dblp_magnn_author_v2": 0.170, "entities_aifb": 0.082,
           "urban_paris_plot_v3bsjg": None, "acm_multiplex_v2": None}
CLASSIC = ("kmeans_x", "spectral_a", "louvain_a", "sgc2_ax")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_dir", default="results/breakthrough_v1/e3_real_dual",
                        help="Source experiment directory; its runs.csv and classic_baselines.csv are always read.")
    parser.add_argument("--extra_dirs", default="", help="Comma-separated experiment dirs whose runs are merged in.")
    parser.add_argument("--extra_classic", default="", help="Comma-separated extra classic-baseline CSVs.")
    parser.add_argument("--datasets", default="cora,photo,dblp_magnn_author_v2,entities_aifb,urban_paris_plot_v3bsjg")
    parser.add_argument("--out_dir", default="", help="Where to write the tables (default: --exp_dir).")
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(dict)
    run_rows = list(csv.DictReader(open(exp_dir / "runs.csv")))
    for extra in [d for d in args.extra_dirs.split(",") if d]:
        run_rows += list(csv.DictReader(open(Path(extra) / "runs.csv")))
    for r in run_rows:
        if "_" not in r["tag"] or r["tag"].rsplit("_", 1)[1] not in ("warm", "nowarm"):
            continue
        if r["status"] not in ("ok", "cached") or not r.get("nmi_mean"):
            continue
        model, start = r["tag"].rsplit("_", 1)
        runs[(model, r["dataset"], r["seed"])][start] = r
    classic = defaultdict(list)
    classic_files = [exp_dir / "classic_baselines.csv"] + [Path(c) for c in args.extra_classic.split(",") if c]
    for path in classic_files:
        for r in csv.DictReader(open(path)):
            classic[(r["method"], r["dataset"])].append((float(r["nmi"]), int(r["n_clusters"])))

    # Start rule: the edge-free baseline's SE loss is comparable across starts (same graph), while edge-aware models
    # compute their loss on their own reweighted graph. So the baseline picks the start per (dataset, seed) and every
    # model uses that start. Still label-free. Falls back to the model's own loss if the baseline run is missing.
    baseline_start = {}
    for (model, ds, seed), starts in runs.items():
        if model == "baseline" and len(starts) == 2:
            baseline_start[(ds, seed)] = min(starts, key=lambda s: float(starts[s]["best_train_loss_mean"]))

    picked = defaultdict(list)
    detail = []
    for (model, ds, seed), starts in sorted(runs.items()):
        cands = {s: r for s, r in starts.items()}
        chosen = baseline_start.get((ds, seed))
        if chosen not in cands:
            chosen = min(cands, key=lambda s: float(cands[s]["best_train_loss_mean"]))
        for s, r in cands.items():
            detail.append({"model": model, "dataset": ds, "seed": seed, "start": s, "chosen": s == chosen,
                           "nmi": float(r["nmi_mean"]), "best_train_loss": float(r["best_train_loss_mean"]),
                           "modularity": float(r["modularity_mean"]), "warm_pseudo_nmi": r.get("warm_pseudo_nmi", "")})
        picked[(model, ds)].append((float(cands[chosen]["nmi_mean"]), chosen))
        for s in cands:
            picked[(model + f"[{s}]", ds)].append((float(cands[s]["nmi_mean"]), s))
    with open(out_dir / "dual_start_detail.csv", "w", newline="") as fh:
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

    datasets = [d for d in args.datasets.split(",") if d]
    models = [m for m in ("baseline", "b31", "b31fr", "b31si", "b31nores", "b31ctx", "b31frctx", "b45", "b45nrctx") if any(k[0] == m for k in picked)]
    md = ["| dataset | old B45 (60 ep) | " + " | ".join(CLASSIC) + " | " + " | ".join(f"{m} (selected)" for m in models)
          + " | " + " | ".join(f"{m} warm / nowarm" for m in models) + " |",
          "|---|---|" + "---|" * len(CLASSIC) + "---|" * (2 * len(models))]
    for ds in datasets:
        cl = []
        for m in CLASSIC:
            v = classic.get((m, ds))
            cl.append(f"{v[0][0]:.3f}" + (f" (k={v[0][1]})" if m == "louvain_a" else "") if v else "—")
        old = OLD_B45.get(ds)
        md.append(f"| {ds} | {'n/a' if old is None else f'{old:.3f}'} | " + " | ".join(cl) + " | "
                  + " | ".join(cell((m, ds)) for m in models) + " | "
                  + " | ".join(f"{cell((m + '[warm]', ds))} / {cell((m + '[nowarm]', ds))}" for m in models) + " |")
    (out_dir / "real_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
