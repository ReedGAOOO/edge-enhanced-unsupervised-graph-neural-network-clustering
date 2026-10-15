#!/usr/bin/env python3
"""Label-free decoder choice for runs trained on a reweighted graph: the DSE assignment versus spectral clustering and
SGC2+KMeans on the same weighted graph, at the model's cluster count and at the class count (--spectral reg uses
regularized spectral clustering).

The candidate with the lowest 2-level structural entropy on the run's weighted input graph (<dataset>_edge_weight.npy,
uniform when absent) is selected. Labels only score the candidates. Writes one row per run.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from inspect_edge_factors import configs_from_cmd  # noqa: E402
from run_classic_baselines import kmeans, load_graph, sgc, spectral, spectral_reg, sym_adj  # noqa: E402
from utils.warm_start import structural_entropy_2level  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--spectral", choices=["plain", "reg"], default="plain",
                        help="reg: regularized spectral clustering, independent of the eigensolver start.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    from data import load_data
    from exp import Exp
    rows = []
    for run in args.run_dirs:
        rd = Path(run)
        cfg = configs_from_cmd((rd / "stdout.txt").read_text().splitlines()[0])
        if args.cpu:
            cfg.use_gpu = False
        exp = Exp(cfg)
        data = load_data(cfg).to(exp.device)
        model = exp.build_model(data)
        model.load_state_dict(torch.load(REPO / "checkpoints" / cfg.save_path, map_location=exp.device))
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            emb, clu = model.get_cluster_results(data)
            pred = model.fix_cluster_results(clu[1], emb, cfg.epsInt).cpu().numpy()
        x, y, ei, _, _ = load_graph(Path(cfg.root_path), cfg.dataset)
        w_path = Path(cfg.root_path) / cfg.dataset / f"{cfg.dataset}_edge_weight.npy"
        w = np.load(w_path).astype(np.float64) if w_path.exists() else None
        a = sym_adj(x.shape[0], ei, w)
        known = y >= 0
        k_cls, k_m = int(np.unique(y[known]).size), int(np.unique(pred).size)
        cands = {"dse": pred}
        for k in sorted({k_cls, k_m}):
            cands[f"spectral_k{k}"] = spectral_reg(a, k, 0) if args.spectral == "reg" else spectral(a, k, 0)
            cands[f"sgc_k{k}"] = kmeans(sgc(a, x), k, 0)
        scored = {name: (structural_entropy_2level(a, lab), normalized_mutual_info_score(y[known], lab[known]))
                  for name, lab in cands.items()}
        pick = min(scored, key=lambda n: scored[n][0])
        row = {"run": rd.name, "dataset": cfg.dataset, "seed": cfg.seed, "dse_nmi": round(scored["dse"][1], 4),
               "dse_se": round(scored["dse"][0], 4), "selected": pick, "selected_nmi": round(scored[pick][1], 4),
               **{f"{n}_nmi": round(v[1], 4) for n, v in scored.items() if n != "dse"},
               **{f"{n}_se": round(v[0], 4) for n, v in scored.items() if n != "dse"}}
        rows.append(row)
        print(row, flush=True)
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
