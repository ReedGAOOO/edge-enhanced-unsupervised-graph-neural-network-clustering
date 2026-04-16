#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DATASET_MAX_NUMS = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
    "entities_aifb": 16,
    "entities_mutag": 16,
    "entities_bgs": 16,
    "entities_am": 16,
    "entities_bgs_top10k": 16,
    "entities_am_top10k": 16,
    "dblp_magnn_author": 12,
    "dblp_magnn_author_v2": 12,
    "fraud_amazon_union": 2,
    "fraud_yelp_homo": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense prior_alpha scan for B15 vs baseline")
    parser.add_argument("--dataset", type=str, default="dblp_magnn_author_v2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--exp_iters", type=int, default=1)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--alpha_start", type=float, default=0.0)
    parser.add_argument("--alpha_end", type=float, default=1.0)
    parser.add_argument("--alpha_step", type=float, default=0.05)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--out_tag", type=str, default="prior_alpha_dense_b15_vs_b0_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def float_range(start: float, end: float, step: float) -> List[float]:
    vals = []
    if step <= 0:
        raise ValueError("alpha_step must be > 0")
    cur = start
    while cur <= end + 1e-12:
        vals.append(round(cur, 6))
        cur += step
    return vals


def parse_seeds(seed_text: str) -> List[int]:
    seeds = []
    for tok in seed_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("No valid seeds found")
    return seeds


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(cwd))


def read_metrics(metrics_path: Path) -> Dict[str, float]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    out = {
        "nmi": float(m.get("nmi_mean", math.nan)),
        "ari": float(m.get("ari_mean", math.nan)),
        "acc": float(m.get("acc_mean", math.nan)),
        "si_loss": float(m.get("si_loss_mean", math.nan)),
        "modularity": float(m.get("modularity_mean", math.nan)),
        "conductance_w": float(m.get("conductance_weighted_mean", math.nan)),
        "best_epoch": float(m.get("best_epoch_mean", math.nan)),
        "best_train_loss": float(m.get("best_train_loss_mean", math.nan)),
        "pred_n_clusters": float(m.get("pred_n_clusters_mean", math.nan)),
        "pred_cluster_cv": float(m.get("pred_cluster_size_cv_mean", math.nan)),
    }
    return out


def parse_stage2_stats(log_path: Path) -> Dict[str, float]:
    if not log_path.exists():
        return {}
    re_graph_alpha = re.compile(r"graph_alpha=([0-9.]+)")
    re_edge_rel = re.compile(r"edge_rel=([0-9.]+)")
    re_edge_mix = re.compile(r"edge_mix=([0-9.]+)")
    graph_alpha_vals, edge_rel_vals, edge_mix_vals = [], [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mg = re_graph_alpha.search(line)
            mr = re_edge_rel.search(line)
            mm = re_edge_mix.search(line)
            if mg:
                graph_alpha_vals.append(float(mg.group(1)))
            if mr:
                edge_rel_vals.append(float(mr.group(1)))
            if mm:
                edge_mix_vals.append(float(mm.group(1)))

    def _mean(xs: List[float]) -> float:
        if not xs:
            return math.nan
        return float(sum(xs) / len(xs))

    def _first(xs: List[float]) -> float:
        if not xs:
            return math.nan
        return float(xs[0])

    return {
        "graph_alpha_mean_log": _mean(graph_alpha_vals),
        "graph_alpha_first_log": _first(graph_alpha_vals),
        "edge_rel_mean_log": _mean(edge_rel_vals),
        "edge_mix_mean_log": _mean(edge_mix_vals),
    }


def build_main_cmd(
    dataset: str,
    seed: int,
    version: str,
    gpu: int,
    epochs: int,
    eval_freq: int,
    train_log_interval: int,
    hid_dim: int,
    exp_iters: int,
    n_cluster_trials: int,
    edge_args: Dict[str, str],
) -> List[str]:
    if dataset.lower() not in DATASET_MAX_NUMS:
        raise ValueError(f"Unknown dataset for max_nums: {dataset}")
    max_nums = DATASET_MAX_NUMS[dataset.lower()]
    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--eval_freq",
        str(eval_freq),
        "--train_log_interval",
        str(train_log_interval),
        "--exp_iters",
        str(exp_iters),
        "--n_cluster_trials",
        str(n_cluster_trials),
        "--hid_dim",
        str(hid_dim),
        "--max_nums",
        str(max_nums),
        "--seed",
        str(seed),
        "--version",
        version,
        "--gpu",
        str(gpu),
        "--save_path",
        f"{version}.pt",
    ]
    for k, v in edge_args.items():
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            cmd.append(k)
        else:
            cmd += [k, str(v)]
    return cmd


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan
    return float(sum(xs) / len(xs))


def std(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(v))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results" / args.out_tag
    results_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    alphas = float_range(args.alpha_start, args.alpha_end, args.alpha_step)
    print(f"[info] dataset={args.dataset}, seeds={seeds}, alpha_count={len(alphas)}")

    baseline_edge_args = {
        "--edge_variant": "V1",
        "--edge_hybrid_alpha": "0.5",
        "--edge_feat_temp": "1.0",
        "--edge_input_prior_alpha": "0.0",
        "--edge_fusion_gamma": "1.0",
        "--edge_fusion_gamma_sched_epochs": "0",
        "--edge_confidence_quantile": "0.0",
        "--edge_adaptive_alpha_strength": "2.0",
        "--edge_adaptive_alpha_bias": "0.0",
        "--edge_reliability_temp": "1.0",
        "--edge_attr_hidden_dim": "64",
        "--edge_attr_fusion_scale": "1.0",
        "--edge_attr_weight_blend": "0.0",
        "--edge_attr_weight_temp": "1.0",
        "--edge_attr_weight_apply_to": "si_only",
    }
    b15_common_edge_args = {
        "--edge_variant": "V12",
        "--edge_hybrid_alpha": "0.7",
        "--edge_feat_temp": "1.0",
        "--edge_fusion_gamma": "1.0",
        "--edge_fusion_gamma_start": "0.2",
        "--edge_fusion_gamma_end": "1.2",
        "--edge_fusion_gamma_sched_epochs": "100",
        "--edge_confidence_quantile": "0.0",
        "--edge_adaptive_alpha_strength": "2.0",
        "--edge_adaptive_alpha_bias": "0.0",
        "--edge_reliability_temp": "1.0",
        "--edge_attr_hidden_dim": "64",
        "--edge_attr_fusion_scale": "0.7",
        "--edge_attr_weight_blend": "0.0",
        "--edge_attr_weight_temp": "1.0",
        "--edge_attr_weight_apply_to": "si_only",
        "--edge_adaptive_alpha": "",
        "--edge_attr_hierarchical": "",
    }

    rows = []

    for seed in seeds:
        b0_version = f"{args.out_tag}_B0_s{seed}"
        b0_metrics = repo_root / "results" / b0_version / f"{args.dataset}_metrics.json"
        if (not b0_metrics.exists()) or (not args.resume):
            cmd = build_main_cmd(
                dataset=args.dataset,
                seed=seed,
                version=b0_version,
                gpu=args.gpu,
                epochs=args.epochs,
                eval_freq=args.eval_freq,
                train_log_interval=args.train_log_interval,
                hid_dim=args.hid_dim,
                exp_iters=args.exp_iters,
                n_cluster_trials=args.n_cluster_trials,
                edge_args=baseline_edge_args,
            )
            run_command(cmd, repo_root, args.dry_run)

        if b0_metrics.exists():
            met = read_metrics(b0_metrics)
            met.update(parse_stage2_stats(repo_root / "results" / b0_version / f"{args.dataset}.log"))
            rows.append(
                {
                    "model": "B0",
                    "seed": seed,
                    "prior_alpha": 0.0,
                    "version": b0_version,
                    **met,
                }
            )

    for seed in seeds:
        for prior_alpha in alphas:
            alpha_tag = str(prior_alpha).replace(".", "p")
            v = f"{args.out_tag}_B15_a{alpha_tag}_s{seed}"
            metrics_path = repo_root / "results" / v / f"{args.dataset}_metrics.json"
            if (not metrics_path.exists()) or (not args.resume):
                edge_args = dict(b15_common_edge_args)
                edge_args["--edge_input_prior_alpha"] = f"{prior_alpha:.6f}"
                cmd = build_main_cmd(
                    dataset=args.dataset,
                    seed=seed,
                    version=v,
                    gpu=args.gpu,
                    epochs=args.epochs,
                    eval_freq=args.eval_freq,
                    train_log_interval=args.train_log_interval,
                    hid_dim=args.hid_dim,
                    exp_iters=args.exp_iters,
                    n_cluster_trials=args.n_cluster_trials,
                    edge_args=edge_args,
                )
                run_command(cmd, repo_root, args.dry_run)

            if metrics_path.exists():
                met = read_metrics(metrics_path)
                met.update(parse_stage2_stats(repo_root / "results" / v / f"{args.dataset}.log"))
                rows.append(
                    {
                        "model": "B15",
                        "seed": seed,
                        "prior_alpha": prior_alpha,
                        "version": v,
                        **met,
                    }
                )

    runs_csv = results_root / "runs.csv"
    fieldnames = [
        "model",
        "seed",
        "prior_alpha",
        "version",
        "nmi",
        "ari",
        "acc",
        "si_loss",
        "modularity",
        "conductance_w",
        "best_epoch",
        "best_train_loss",
        "pred_n_clusters",
        "pred_cluster_cv",
        "graph_alpha_mean_log",
        "graph_alpha_first_log",
        "edge_rel_mean_log",
        "edge_mix_mean_log",
    ]
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, math.nan) for k in fieldnames})

    b0_by_seed = {int(r["seed"]): r for r in rows if r["model"] == "B0"}
    b15_rows = [r for r in rows if r["model"] == "B15"]
    delta_rows = []
    for r in b15_rows:
        seed = int(r["seed"])
        if seed not in b0_by_seed:
            continue
        b0 = b0_by_seed[seed]
        delta_rows.append(
            {
                "seed": seed,
                "prior_alpha": r["prior_alpha"],
                "version_b15": r["version"],
                "version_b0": b0["version"],
                "nmi_b15": r["nmi"],
                "nmi_b0": b0["nmi"],
                "delta_nmi": r["nmi"] - b0["nmi"],
                "ari_b15": r["ari"],
                "ari_b0": b0["ari"],
                "delta_ari": r["ari"] - b0["ari"],
                "acc_b15": r["acc"],
                "acc_b0": b0["acc"],
                "delta_acc": r["acc"] - b0["acc"],
                "delta_si_loss": r["si_loss"] - b0["si_loss"],
                "delta_modularity": r["modularity"] - b0["modularity"],
                "delta_conductance_w": r["conductance_w"] - b0["conductance_w"],
                "best_epoch_b15": r["best_epoch"],
                "best_epoch_b0": b0["best_epoch"],
                "graph_alpha_mean_log": r.get("graph_alpha_mean_log", math.nan),
                "edge_rel_mean_log": r.get("edge_rel_mean_log", math.nan),
                "edge_mix_mean_log": r.get("edge_mix_mean_log", math.nan),
            }
        )

    deltas_csv = results_root / "delta_vs_baseline.csv"
    if delta_rows:
        with open(deltas_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(delta_rows)

    by_alpha: Dict[float, List[Dict[str, float]]] = {}
    for r in delta_rows:
        by_alpha.setdefault(float(r["prior_alpha"]), []).append(r)

    summary_rows = []
    for alpha in sorted(by_alpha.keys()):
        rs = by_alpha[alpha]
        d_nmi = [float(x["delta_nmi"]) for x in rs]
        d_ari = [float(x["delta_ari"]) for x in rs]
        d_acc = [float(x["delta_acc"]) for x in rs]
        d_mod = [float(x["delta_modularity"]) for x in rs]
        d_cond = [float(x["delta_conductance_w"]) for x in rs]
        g_alpha = [float(x.get("graph_alpha_mean_log", math.nan)) for x in rs]
        edge_rel = [float(x.get("edge_rel_mean_log", math.nan)) for x in rs]
        edge_mix = [float(x.get("edge_mix_mean_log", math.nan)) for x in rs]
        summary_rows.append(
            {
                "prior_alpha": alpha,
                "n": len(rs),
                "delta_nmi_mean": mean(d_nmi),
                "delta_nmi_std": std(d_nmi),
                "delta_ari_mean": mean(d_ari),
                "delta_ari_std": std(d_ari),
                "delta_acc_mean": mean(d_acc),
                "delta_modularity_mean": mean(d_mod),
                "delta_conductance_w_mean": mean(d_cond),
                "win_rate_nmi": mean([1.0 if x > 0 else 0.0 for x in d_nmi]),
                "win_rate_ari": mean([1.0 if x > 0 else 0.0 for x in d_ari]),
                "graph_alpha_mean_log": mean(g_alpha),
                "edge_rel_mean_log": mean(edge_rel),
                "edge_mix_mean_log": mean(edge_mix),
            }
        )

    summary_csv = results_root / "summary_by_alpha.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    best_by_nmi = None
    best_by_ari = None
    if summary_rows:
        best_by_nmi = max(summary_rows, key=lambda x: x["delta_nmi_mean"])
        best_by_ari = max(summary_rows, key=lambda x: x["delta_ari_mean"])

    summary_json = {
        "dataset": args.dataset,
        "seeds": seeds,
        "alphas": alphas,
        "epochs": args.epochs,
        "eval_freq": args.eval_freq,
        "train_log_interval": args.train_log_interval,
        "runs_csv": str(runs_csv),
        "delta_csv": str(deltas_csv),
        "summary_csv": str(summary_csv),
        "best_by_delta_nmi_mean": best_by_nmi,
        "best_by_delta_ari_mean": best_by_ari,
    }
    with open(results_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("[ok] wrote:")
    print(" -", runs_csv)
    print(" -", deltas_csv)
    print(" -", summary_csv)
    print(" -", results_root / "summary.json")


if __name__ == "__main__":
    main()
