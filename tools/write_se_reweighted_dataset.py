#!/usr/bin/env python3
"""Write a copy of a dataset whose edge weights are an SE-guided search pick (tools/run_se_guided_baseline.py).

The pick is a weighting name from that tool's summary ("uniform", "layer+PAP", "+cos", "-attr0", "+l2*ctx1",
"+attr0*ctx32*ctx33", ...). It is rebuilt exactly as the search built it and stored as <out>_edge_weight.npy next to
the original features, labels, edge_index and edge_attr, so an edge-free DSE can train on the searched graph
("search, then train"). The pick is chosen without labels; this script only materialises it.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from run_classic_baselines import load_graph  # noqa: E402
from run_se_guided_baseline import endpoint_context, factor  # noqa: E402
from screen_edge_headroom import generic_edge_attr  # noqa: E402

GENERIC = ["cos", "l2", "l1", "logdeg_src", "logdeg_dst", "logdeg_gap"]


def pick_weight(pick, x, ei, ea, meta, ctx_dim):
    if ea is None:
        ea, chans = generic_edge_attr(x, ei), GENERIC
    else:
        chans = list(meta.get("edge_attr_channels") or [])[:ea.shape[1]]
        chans += [f"attr{i}" for i in range(len(chans), ea.shape[1])]
    if pick == "uniform":
        return np.ones(ei.shape[1])
    if pick.startswith("layer+"):
        members = {c[:-7]: ea[:, i] > 0 for i, c in enumerate(chans) if c.endswith("_member")}
        n_layers = np.maximum(1.0, sum(m.astype(float) for m in members.values()))
        return sum((2.0 if layer == pick[6:] else 0.5) * m for layer, m in members.items()) / n_layers
    sign = 1 if pick[0] == "+" else -1
    score = np.ones(ei.shape[1])
    ctx = None
    for term in pick[1:].split("*"):
        m = re.fullmatch(r"ctx(\d+)", term)
        if m:
            ctx = endpoint_context(x, ei, ctx_dim) if ctx is None else ctx
            score = score * ctx[:, int(m.group(1))]
        else:
            score = score * ea[:, chans.index(term)]
    return factor(score, sign)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pick", required=True, help="Weighting name, without the '/method' suffix.")
    parser.add_argument("--ctx_dim", type=int, default=32, help="Endpoint-context width the search used.")
    parser.add_argument("--out_name", required=True)
    args = parser.parse_args()
    root = Path(args.root_path)
    x, y, ei, ea, meta = load_graph(root, args.dataset)
    w = pick_weight(args.pick.split("/")[0], x, ei, ea, meta, args.ctx_dim)
    out = root / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.out_name}_feat.npy", x.astype(np.float32))
    # load_graph maps an "unknown" class to -1; restore the stored convention so ATsDataset sees the original labels.
    src_dir = root / args.dataset
    label_src = src_dir / f"{args.dataset}_label.npy"
    np.save(out / f"{args.out_name}_label.npy", np.load(label_src) if label_src.exists() else y)
    np.save(out / f"{args.out_name}_edge_index.npy", ei.astype(np.int64))
    np.save(out / f"{args.out_name}_edge_weight.npy", w.astype(np.float32))
    if ea is not None:
        np.save(out / f"{args.out_name}_edge_attr.npy", ea.astype(np.float32))
    meta = dict(meta)
    meta.update({"dataset": args.out_name, "source_dataset": args.dataset, "se_search_pick": args.pick,
                 "se_search_ctx_dim": args.ctx_dim})
    (out / f"{args.out_name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.out_name}: pick={args.pick} E={ei.shape[1]} weight mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}")


if __name__ == "__main__":
    main()
