#!/usr/bin/env python3
"""Turn the synth_ctx_v1 runs into the edge-mechanism table used in the README.

Effects are differences of mean NMI between datasets that share graph, features and labels:
  architecture   model(permEA)      - baseline         (edge rows shuffled: no edge information at all)
  descriptive    model(none)        - model(permEA)    (aligned nuisance channels, e.g. endpoint feature cosine)
  direct         model(direct)      - model(none)      (only channel 0 differs)
  contextual     model(contextual)  - model(none)      (only channel 0 differs; needs endpoint context)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(path: Path):
    by_key = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["status"] not in ("ok", "cached") or not r.get("nmi_mean"):
            continue
        by_key[(r["tag"], r["dataset"])].append(float(r["nmi_mean"]))
    return by_key


def load_classic(path: Path):
    out = defaultdict(list)
    if path.exists():
        for r in csv.DictReader(open(path)):
            out[(r["method"], r["dataset"])].append(float(r["nmi"]))
    return out


def fmt(vals):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_dir", default="results/breakthrough_v1/e2_synth_ctx_warm")
    parser.add_argument("--classic_csv", default="results/breakthrough_v1/e2_synth_ctx_warm/classic_baselines.csv")
    parser.add_argument("--prefix", default="synth_ctx_v1")
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    runs = load_runs(exp_dir / "runs.csv")
    classic = load_classic(Path(args.classic_csv))
    tags = sorted({t for t, _ in runs if t != "baseline"})
    rows = []
    md = ["| h | model | permEA | none | direct | contextual | arch | descr | direct eff | context eff |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for h in ("h45", "h55"):
        ds = {m: f"{args.prefix}_{h}_{m}_ds00" for m in ("none", "direct", "contextual")}
        base = runs.get(("baseline", ds["none"]), [])
        sgc = classic.get(("sgc2_ax", ds["none"]), [])
        oracle_d = classic.get(("oracle_sgc2_ax_attr0", ds["direct"]), [])
        oracle_c = classic.get(("oracle_sgc2_ax_attr0", ds["contextual"]), [])
        md.append(f"| {h} | baseline (no edges) | | {fmt(base)} | | | | | | |")
        md.append(f"| {h} | SGC2+KMeans / oracle | | {fmt(sgc)} | {fmt(oracle_d)} | {fmt(oracle_c)} | | | | |")
        for tag in tags:
            v = {
                "permEA": runs.get((tag, ds["direct"] + "_permEA"), []) + runs.get((tag, ds["contextual"] + "_permEA"), []),
                "none": runs.get((tag, ds["none"]), []),
                "direct": runs.get((tag, ds["direct"]), []),
                "contextual": runs.get((tag, ds["contextual"]), []),
            }
            mean = {k: (np.mean(x) if x else np.nan) for k, x in v.items()}
            eff = {
                "architecture": mean["permEA"] - (np.mean(base) if base else np.nan),
                "descriptive": mean["none"] - mean["permEA"],
                "direct": mean["direct"] - mean["none"],
                "contextual": mean["contextual"] - mean["none"],
            }
            rows.append({"h": h, "model": tag, **{f"nmi_{k}": mean[k] for k in mean}, **{f"effect_{k}": eff[k] for k in eff}})
            md.append(f"| {h} | {tag} | {fmt(v['permEA'])} | {fmt(v['none'])} | {fmt(v['direct'])} | {fmt(v['contextual'])} | "
                      + " | ".join("—" if np.isnan(e) else f"{e:+.3f}" for e in eff.values()) + " |")
    with open(exp_dir / "mechanism_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (exp_dir / "mechanism_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
