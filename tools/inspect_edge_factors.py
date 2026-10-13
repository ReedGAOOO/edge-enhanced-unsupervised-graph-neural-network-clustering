#!/usr/bin/env python3
"""Reload a finished run and report the learned edge-weight factors, grouped by edge-attribute channels.

Answers "what did the edge-weighting head learn?": for multiplex graphs, the mean SI/message factor of edges in each
meta-path layer; for any labeled graph, how strongly the factor tracks the same-label indicator (diagnostic only).
The run is rebuilt from the CMD line its stdout.txt starts with, and its checkpoint is loaded from ./checkpoints.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def configs_from_cmd(cmd_line: str):
    argv = shlex.split(cmd_line.split("CMD:", 1)[1])
    argv = argv[argv.index("main.py") + 1:]
    src = (REPO / "main.py").read_text().split("configs = parser.parse_args()")[0]
    ns = {"__name__": "cfg"}
    exec(src, ns)
    return ns["parser"].parse_args(argv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True, help="results/.../runs/<run_id> containing stdout.txt")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    configs = configs_from_cmd((run_dir / "stdout.txt").read_text().splitlines()[0])

    from data import load_data
    from exp import Exp

    exp = Exp(configs)
    data = load_data(configs).to(exp.device)
    model = exp.build_model(data)
    ckpt = REPO / "checkpoints" / configs.save_path
    model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    model.eval()
    if not model._use_learnable_edge_weight_variant():
        print(f"{configs.edge_variant} has no learnable edge weights")
        return
    ea = data.edge_attr
    with torch.no_grad():
        _, factor_all, _ = model._edge_scores_and_factors_from_attr(ea, torch.ones(ea.shape[0], dtype=torch.bool, device=ea.device))
    heads = {"si": model._edge_head_index("si"), "msg": model._edge_head_index("msg")}
    factors = {h: factor_all[:, min(i, factor_all.shape[1] - 1)].cpu().numpy() for h, i in heads.items()}

    y = data.y.cpu().numpy()
    src, dst = data.edge_index.cpu().numpy()
    labeled = (y[src] >= 0) & (y[dst] >= 0)
    same = (y[src] == y[dst]).astype(float)
    print(f"run {run_dir.name}: E={ea.shape[0]}")
    for h, f in factors.items():
        corr = np.corrcoef(np.log(f[labeled]), same[labeled])[0, 1] if labeled.any() else float("nan")
        print(f"  {h:3s} factor mean={f.mean():.3f} std={f.std():.3f} min={f.min():.3f} max={f.max():.3f} "
              f"| corr(log factor, same-label)={corr:+.3f} | same-label edges {f[labeled & (same > 0)].mean():.3f} vs "
              f"cross-label {f[labeled & (same == 0)].mean():.3f}")
    meta_path = Path(configs.root_path) / configs.dataset / f"{configs.dataset}_meta.json"
    if meta_path.exists():
        chans = json.loads(meta_path.read_text()).get("edge_attr_channels", [])
        for ci, cname in enumerate(chans):
            if not cname.endswith("_member") or ci >= ea.shape[1]:
                continue
            member = (ea[:, ci] > 0).cpu().numpy()
            print(f"  layer {cname[:-7]:4s} edges={int(member.sum()):7d} hom={same[member & labeled].mean():.3f} "
                  + " ".join(f"{h}={f[member].mean():.3f}" for h, f in factors.items()))


if __name__ == "__main__":
    main()
