#!/usr/bin/env python3
"""Controlled checkpoint and paired-training interventions on B45's message gates.

Uses existing checkpoints or matched initializations without changing production
model code. Labels are diagnostic only. Training probes use a fixed SGC pseudo
partition and report fixed-epoch outcomes (not label-selected best outcomes).
"""
import argparse
import json
import random
import sys
import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from inspect_edge_factors import configs_from_cmd
from data import load_data
from exp import Exp
from modules.layers import LorentzAgg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gates(model):
    return [m for m in model.modules() if isinstance(m, LorentzAgg) and m.edge_conditioned]


def intervene(model, mode):
    for gate in gates(model):
        if mode == "off":
            gate.edge_conditioned = False
        elif mode == "unit_gate":
            gate.edge_gate_scale = 0.0
        elif mode == "no_renorm":
            gate._normalize_sparse_by_degree = lambda adj: adj
        elif mode == "gate_dropout_train":
            gate.edge_gate_mlp.train()
        elif mode == "all_train":
            model.train()


def score(model, data, seed):
    seed_all(seed)
    with torch.no_grad():
        embeddings, clusters = model.get_cluster_results(data)
        pred = model.fix_cluster_results(clusters[1], embeddings, 8).cpu().numpy()
    y = data.y.cpu().numpy()
    known = y >= 0
    _, counts = np.unique(pred, return_counts=True)
    return {"nmi": float(normalized_mutual_info_score(y[known], pred[known])),
            "max_share": float(counts.max() / counts.sum()), "k": len(counts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--root_path", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--train", action="store_true")
    p.add_argument("--warm_epochs", type=int, default=200)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--modes", nargs="+",
                   choices=["original", "off", "unit_gate", "no_renorm", "no_gate_dropout", "safe_isolates"],
                   default=["original", "off", "unit_gate", "no_renorm", "no_gate_dropout"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph_cross", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    cfg = configs_from_cmd((run / "stdout.txt").read_text().splitlines()[0])
    cfg.root_path = args.root_path
    if args.cpu:
        cfg.use_gpu = False
    exp = Exp(cfg)
    data = load_data(cfg).to(exp.device)
    if args.train:
        train_probe(args, cfg, exp, data)
        return
    state = torch.load(REPO / "checkpoints" / cfg.save_path, map_location=exp.device, weights_only=True)
    if args.graph_cross:
        model = exp.build_model(data)
        model.load_state_dict(state)
        model.set_edge_fusion_gamma(cfg.edge_fusion_gamma_end or cfg.edge_fusion_gamma)
        graph_cross_probe(model, data, Path(args.out))
        return
    rows = []
    for mode in ["original", "off", "unit_gate", "no_renorm", "gate_dropout_train", "all_train"]:
        model = exp.build_model(data)
        model.load_state_dict(state)
        model.set_edge_fusion_gamma(cfg.edge_fusion_gamma_end or cfg.edge_fusion_gamma)
        model.eval()
        intervene(model, mode)
        trials = [score(model, data, s) for s in range(args.trials)]
        row = {"mode": mode, "trials": trials,
               **{key: float(np.mean([t[key] for t in trials])) for key in trials[0]}}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_dir": str(run), "rows": rows}, indent=2) + "\n")


@torch.no_grad()
def graph_cross_probe(model, data, out):
    """Cross the graph used for leaf embeddings with the assignment graph.

    All four cells use one fixed checkpoint in eval mode. This distinguishes
    encoder covariate shift from dependence on augmentation in the assignment.
    """
    model.eval()
    captured = {}
    def capture(module, args, kwargs):
        captured["adj"] = args[1]
        captured["attr"] = kwargs.get("edge_attr")
        captured["mask"] = kwargs.get("edge_mask")
    handle = model.encoder.register_forward_pre_hook(capture, with_kwargs=True)
    seed_all(0)
    model.se_loss(data)
    handle.remove()
    graphs = {"augmented": dict(captured)}
    adj = data.adj_msg
    if model._use_learnable_edge_weight_variant():
        adj, _ = model._apply_learned_edge_weight_to_adj(data.adj_msg, adj, data.edge_attr, True, head="msg")
    graphs["base"] = {"adj": adj, "attr": data.edge_attr, "mask": None}
    use_attr = model._use_edge_attr_variant()
    embed = model.encoder.embed_leaf
    z = {name: embed(data.x, g["adj"], edge_attr=g["attr"], edge_mask=g["mask"], use_edge_attr=use_attr)
         for name, g in graphs.items()}
    y = data.y.cpu().numpy()
    known = y >= 0
    rows = []
    for emb_name, leaf in z.items():
        for ass_name, g in graphs.items():
            model.encoder.embed_leaf = lambda *args, _leaf=leaf, **kwargs: _leaf
            seed_all(999)
            _, ass, _ = model.encoder(data.x, g["adj"], edge_attr=g["attr"], edge_mask=g["mask"], use_edge_attr=use_attr)
            pred = ass[model.height].argmax(1).cpu().numpy()
            _, counts = np.unique(pred, return_counts=True)
            row = {"embedding_graph": emb_name, "assignment_graph": ass_name,
                   "nmi": float(normalized_mutual_info_score(y[known], pred[known])),
                   "max_share": float(counts.max() / counts.sum()), "k": len(counts)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    model.encoder.embed_leaf = embed
    diagnostics = {}
    for name, g in graphs.items():
        a = g["adj"].coalesce()
        mask = g["mask"]
        diagnostics[name] = {"edges": a._nnz(), "weight_sum": a.values().sum().item(),
                             "spatial_std_norm": z[name][:, 1:].std(0).norm().item()}
        if mask is not None:
            diagnostics[name].update(new_edges=int((~mask).sum()),
                                     new_edge_weight_fraction=float(a.values()[~mask].sum() / a.values().sum()))
            degree = torch.sparse.sum(data.adj_msg_raw, dim=1).to_dense()
            dst = a.indices()[1, ~mask]
            count = torch.bincount(dst, minlength=len(degree))
            top = count.topk(min(12, len(count)))
            diagnostics[name].update(
                original_isolated_nodes=int((degree == 0).sum()),
                new_edges_to_isolates_fraction=float((degree[dst] == 0).float().mean()),
                top_new_destinations=[{"node": int(i), "new_indegree": int(c), "original_degree": float(degree[i])}
                                      for i, c in zip(top.indices, top.values)])
            gate_stats = []
            for gate in gates(model):
                v = gate._apply_edge_gate_to_adj(a, g["attr"], mask, True).values()
                old_scale = gate.edge_gate_scale
                gate.edge_gate_scale = 0.0
                v_unit = gate._apply_edge_gate_to_adj(a, g["attr"], mask, True).values()
                gate.edge_gate_scale = old_scale
                gate_stats.append({"new_edge_weight_fraction": float(v[~mask].sum() / v.sum()),
                                   "unit_gate_new_edge_weight_fraction": float(v_unit[~mask].sum() / v_unit.sum())})
            diagnostics[name]["gate_stats"] = gate_stats
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "graphs": diagnostics}, indent=2) + "\n")


def train_probe(args, cfg, exp, data):
    """Common initialization and fixed SGC pseudo labels; preserve RNG consumption
    when removing gates/dropout so later backbone dropout masks also stay paired.
    No diagnostic label metric is used for optimization or checkpoint selection.
    """
    from utils.warm_start import build_pseudo_partition
    seed_all(args.seed)
    initial = copy.deepcopy(exp.build_model(data).state_dict())
    pseudo_np, info = build_pseudo_partition(data.edge_index.cpu().numpy(), data.x.cpu().numpy(),
                                           k=min(data.num_classes, cfg.max_nums[0]), method="sgc", seed=args.seed)
    pseudo = torch.as_tensor(pseudo_np, device=data.x.device, dtype=torch.long)
    y = data.y.cpu().numpy()
    known = y >= 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".pseudo.npy"), pseudo_np)
    result = {"config": vars(cfg), "seed": args.seed, "warm_epochs": args.warm_epochs,
              "epochs": args.epochs, "probe_warm_method": "sgc", "pseudo": info,
              "status": "running", "rows": []}
    start = time.monotonic()
    for mode in args.modes:
        model = exp.build_model(data)
        model.load_state_dict(initial)
        if mode == "safe_isolates":
            # Preserve a valid transformed node representation for empty rows,
            # instead of mapping isolated nodes to the off-manifold zero vector.
            def keep_isolated(module, inputs, output):
                x, adj = inputs[:2]
                degree = torch.sparse.sum(adj.coalesce(), dim=1).to_dense() if adj.is_sparse else adj.sum(1)
                return torch.where((degree == 0).unsqueeze(1), x, output)
            for module in model.modules():
                if isinstance(module, LorentzAgg):
                    module.register_forward_hook(keep_isolated)
        for gate in gates(model):
            if mode == "off":
                original = gate._apply_edge_gate_to_adj
                def consume_and_bypass(adj, edge_attr, edge_mask, use_edge_attr, fn=original):
                    fn(adj, edge_attr, edge_mask, use_edge_attr)
                    return adj
                gate._apply_edge_gate_to_adj = consume_and_bypass
            elif mode == "unit_gate":
                gate.edge_gate_scale = 0.0
            elif mode == "no_renorm":
                gate._normalize_sparse_by_degree = lambda adj: adj
            elif mode == "no_gate_dropout":
                dropout = gate.edge_gate_mlp[2]
                original = dropout.forward
                def consume_and_keep(x, fn=original):
                    fn(x)
                    return x
                dropout.forward = consume_and_keep
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
        best_loss = float("inf")
        for phase, epochs in [("warm", args.warm_epochs), ("se", args.epochs)]:
            for epoch in range(1, epochs + 1):
                seed_all(args.seed * 100000 + epoch + (10000 if phase == "se" else 0))
                model.train()
                if phase == "se":
                    model.set_edge_fusion_gamma(exp._edge_fusion_gamma_for_epoch(epoch))
                opt.zero_grad(set_to_none=True)
                loss = model.warm_start_loss(data, pseudo) if phase == "warm" else model.se_loss(data)
                loss.backward()
                opt.step()
                if phase == "se" and loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                if epoch in (1, 10, 25, 50, 100, 150, 200, epochs):
                    pred = model.last_ass_dict[model.height].detach().argmax(1).cpu().numpy()
                    model.eval()
                    row = {"mode": mode, "phase": phase, "epoch": epoch, "loss": loss.item(),
                           "train_assignment_nmi": float(normalized_mutual_info_score(y[known], pred[known])),
                           **score(model, data, 999), "elapsed": time.monotonic() - start}
                    if phase == "se":
                        row.update(best_loss=best_loss, best_epoch=best_epoch)
                    result["rows"].append(row)
                    out.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
        del model, opt, loss
        torch.cuda.empty_cache()
    result["status"] = "complete"
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
