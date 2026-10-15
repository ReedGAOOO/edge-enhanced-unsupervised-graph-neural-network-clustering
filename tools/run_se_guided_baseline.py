#!/usr/bin/env python3
"""Label-free SE-guided edge reweighting followed by a classic clusterer: the "simple baseline" a learned edge head
has to beat, and the search stage of the search-then-train pipeline.

Candidate weightings stay in the learned-factor range of B31 (factor exp(+-0.7 * tanh(z)), z a standardised score):
  F0  uniform;
  F1  one edge-attribute channel (native, or the generic cosine/L2/L1/log-degree attributes), plus multiplex layer
      emphasis (one <layer>_member layer at 2.0, the others at 0.5);
  F2  one endpoint-context channel z_i[d] * z_j[d] (data.build_endpoint_context_attr, the --edge_ctx_dim input);
  F3  one edge channel times one endpoint-context channel (a context-dependent edge reading);
  F4  (--order2) one edge channel times two endpoint-context channels, enumerated exhaustively;
  Br  (--beam B) beam search: the B lowest-SE candidates of order r-1 (starting from F3) are each multiplied by one
      more endpoint-context channel, up to --max_order. B * ctx_dim * 2 candidates per order instead of all
      combinations. A context whose lower-order products carry no signal (XOR-like roles) gives the beam no foothold.
Every weighting is clustered with spectral clustering (--spectral plain, or reg for regularized spectral clustering,
which does not depend on the eigensolver start on graphs with many components) and SGC2+KMeans at k = number of
classes. The partition with the lowest 2-level structural entropy on its own weighted graph is selected, without
labels, within F0+F1 ("noctx"), F0-F3 ("ctx"), F0-F4 ("ctx2", with --order2) and F0-F3 plus Br ("beam", with
--beam). The best NMI in each family is reported as a label-selected upper bound only.
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
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from utils.warm_start import structural_entropy_2level  # noqa: E402
from run_classic_baselines import kmeans, load_graph, sgc, spectral, spectral_reg, sym_adj  # noqa: E402
from screen_edge_headroom import generic_edge_attr  # noqa: E402

LOGCLIP = 0.7
BATCH = 64
_G = {}


def factor(score, sign):
    z = (score - score.mean()) / max(score.std(), 1e-12)
    return np.exp(sign * LOGCLIP * np.tanh(z))


def endpoint_context(x, ei, max_dim):
    import torch
    from data import build_endpoint_context_attr
    torch.manual_seed(0)
    return build_endpoint_context_attr(torch.from_numpy(x).float(), torch.from_numpy(ei).long(), max_dim).numpy()


def product_tag(sign, chan, dims):
    return f"{'+-'[sign < 0]}{chan}" + "".join(f"*ctx{d}" for d in dims)


def candidates(ea, chans, ctx, order2=False):
    yield "F0", "uniform", None
    members = {c[:-7]: ea[:, i] > 0 for i, c in enumerate(chans) if c.endswith("_member")}
    if members:
        n_layers = np.maximum(1.0, sum(m.astype(float) for m in members.values()))
        for up in members:
            yield "F1", f"layer+{up}", sum((2.0 if layer == up else 0.5) * m for layer, m in members.items()) / n_layers
    for i, name in enumerate(chans):
        if not name.endswith("_member"):
            for sign in (1, -1):
                yield "F1", f"{'+-'[sign < 0]}{name}", factor(ea[:, i], sign)
    if ctx is None:
        return
    for d in range(ctx.shape[1]):
        for sign in (1, -1):
            yield "F2", f"{'+-'[sign < 0]}ctx{d}", factor(ctx[:, d], sign)
    for i, name in enumerate(chans):
        for d in range(ctx.shape[1]):
            for sign in (1, -1):
                yield "F3", product_tag(sign, name, (d,)), factor(ea[:, i] * ctx[:, d], sign)
    if not order2:
        return
    for i, name in enumerate(chans):
        for d in range(ctx.shape[1]):
            for e in range(d + 1, ctx.shape[1]):
                for sign in (1, -1):
                    yield "F4", product_tag(sign, name, (d, e)), factor(ea[:, i] * ctx[:, d] * ctx[:, e], sign)


def evaluate(args):
    family, tag, w = args
    x, y, ei, n, k, seed, method = (_G[key] for key in ("x", "y", "ei", "n", "k", "seed", "spectral"))
    a = sym_adj(n, ei, w)
    known = y >= 0
    spec = spectral_reg(a, k, seed) if method == "reg" else spectral(a, k, seed)
    rows = []
    for name, lab in (("spectral", spec), ("sgc", kmeans(sgc(a, x), k, seed))):
        rows.append({"family": family, "weighting": tag, "method": name, "se": structural_entropy_2level(a, lab),
                     "nmi": normalized_mutual_info_score(y[known], lab[known])})
    return rows


def init_worker(state):
    _G.update(state)


def score_all(pool, cands):
    """Evaluate a candidate iterable in batches so that only BATCH weight vectors are held at once."""
    rows, batch = [], []
    for cand in cands:
        batch.append(cand)
        if len(batch) == BATCH:
            rows += [r for chunk in pool.map(evaluate, batch) for r in chunk]
            batch = []
    if batch:
        rows += [r for chunk in pool.map(evaluate, batch) for r in chunk]
    return rows


def beam_search(pool, rows, ea, chans, ctx, width, max_order):
    """Grow F3 products by one context channel per order, keeping the `width` lowest-SE candidates of each order."""
    best = {}
    for r in rows:
        best[r["weighting"]] = min(best.get(r["weighting"], np.inf), r["se"])
    specs = {}
    for i, chan in enumerate(chans):
        for d in range(ctx.shape[1]):
            for sign in (1, -1):
                specs[product_tag(sign, chan, (d,))] = (i, (d,), sign)
    seen = {(i, dims, sign) for i, dims, sign in specs.values()}
    frontier = sorted((t for t in specs if t in best), key=best.get)[:width]
    new_rows = []
    for order in range(2, max_order + 1):
        level = []
        for tag in frontier:
            i, dims, _ = specs[tag]
            for d in range(ctx.shape[1]):
                if d in dims:
                    continue
                nd = tuple(sorted(dims + (d,)))
                for sign in (1, -1):
                    if (i, nd, sign) in seen:
                        continue
                    seen.add((i, nd, sign))
                    t = product_tag(sign, chans[i], nd)
                    specs[t] = (i, nd, sign)
                    level.append(t)

        def gen(tags=level, fam=f"B{order}"):
            for t in tags:
                i, nd, sign = specs[t]
                yield fam, t, factor(ea[:, i] * np.prod(ctx[:, list(nd)], axis=1), sign)

        level_rows = score_all(pool, gen())
        new_rows += level_rows
        for r in level_rows:
            best[r["weighting"]] = min(best.get(r["weighting"], np.inf), r["se"])
        frontier = sorted(level, key=best.get)[:width]
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ctx_dim", type=int, default=32, help="Endpoint-context width (the models' --edge_ctx_dim).")
    parser.add_argument("--synth_ctx_dim", type=int, default=64, help="Context width for synth_ctx_* datasets.")
    parser.add_argument("--spectral", choices=["plain", "reg"], default="plain",
                        help="Spectral clusterer used to score candidates; reg is start-vector independent.")
    parser.add_argument("--order2", action="store_true", help="Add the exhaustive second-order context family F4.")
    parser.add_argument("--beam", type=int, default=0, help="Beam width for higher-order context products (0 = off).")
    parser.add_argument("--max_order", type=int, default=2, help="Highest context order the beam reaches.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        x, y, ei, ea, meta = load_graph(Path(args.root_path), name)
        if ea is None:
            ea = generic_edge_attr(x, ei)
            chans = ["cos", "l2", "l1", "logdeg_src", "logdeg_dst", "logdeg_gap"]
        else:
            chans = list(meta.get("edge_attr_channels") or [f"attr{i}" for i in range(ea.shape[1])])[:ea.shape[1]]
            chans += [f"attr{i}" for i in range(len(chans), ea.shape[1])]
        ctx = endpoint_context(x, ei, args.synth_ctx_dim if name.startswith("synth_ctx") else args.ctx_dim)
        n, k = x.shape[0], int(np.unique(y[y >= 0]).size)
        for seed in [int(s) for s in args.seeds.split(",")]:
            state = {"x": x, "y": y, "ei": ei, "n": n, "k": k, "seed": seed, "spectral": args.spectral}
            with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(state,)) as pool:
                rows = score_all(pool, candidates(ea, chans, ctx, args.order2))
                if args.beam > 0:
                    rows += beam_search(pool, rows, ea, chans, ctx, args.beam, args.max_order)
            with open(out / f"{name}__s{seed}_candidates.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            fams_seen = {r["family"] for r in rows}
            rec = {"dataset": name, "seed": seed, "spectral": args.spectral, "n_candidates": len(rows) // 2}
            scopes = [("noctx", {"F0", "F1"}), ("ctx", {"F0", "F1", "F2", "F3"})]
            if args.order2:
                scopes.append(("ctx2", {"F0", "F1", "F2", "F3", "F4"}))
            if args.beam > 0:
                scopes.append(("beam", {"F0", "F1", "F2", "F3"} | {f for f in fams_seen if f.startswith("B")}))
                rec["n_beam_candidates"] = sum(1 for r in rows if r["family"].startswith("B")) // 2
            for scope, fams in scopes:
                pool_rows = [r for r in rows if r["family"] in fams]
                pick = min(pool_rows, key=lambda r: r["se"])
                rec.update({f"{scope}_nmi": round(pick["nmi"], 4), f"{scope}_pick": f"{pick['weighting']}/{pick['method']}",
                            f"{scope}_oracle_nmi": round(max(r["nmi"] for r in pool_rows), 4)})
            rec["uniform_best_se_nmi"] = round(min((r for r in rows if r["family"] == "F0"), key=lambda r: r["se"])["nmi"], 4)
            summary.append(rec)
            print(rec, flush=True)
    fields = []
    for r in summary:
        fields += [k for k in r if k not in fields]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
