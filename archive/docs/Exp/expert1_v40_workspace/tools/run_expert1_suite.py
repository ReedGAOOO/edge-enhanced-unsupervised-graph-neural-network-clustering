#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PRESETS = [
    "baseline_v1",
    "g20_se_consistent_main",
    "b45_v31_msgcond_gs050",
    "b50_v40_relation_state",
]

DEFAULT_DATASETS = [
    "entities_aifb",
    "entities_mutag",
    "dblp_magnn_author_v2",
]

DEFAULT_SEEDS = [0, 1, 2]

METRIC_FIELDS = [
    "acc_mean",
    "nmi_mean",
    "ari_mean",
    "si_loss_mean",
    "modularity_mean",
    "conductance_mean",
    "conductance_weighted_mean",
    "pred_n_clusters_mean",
    "pred_cluster_size_cv_mean",
    "best_epoch_mean",
    "best_train_loss_mean",
    "final_edge_factor_mean_mean",
    "final_edge_factor_msg_mean",
    "final_edge_factor_si_mean",
    "final_msg_gate_factor_mean",
    "final_edge_mix_beta_mean",
    "final_hier_edge_nonzero_ratio_mean",
]


def parse_csv_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run expert1 real-data comparison suite")
    parser.add_argument("--presets", type=str, default=",".join(DEFAULT_PRESETS))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--n_cluster_trials", type=int, default=3)
    parser.add_argument("--exp_iters", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--version_prefix", type=str, default="expert1_real_e80_t3")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    presets = parse_csv_list(args.presets)
    datasets = parse_csv_list(args.datasets)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    out_dir = repo_root / "results" / args.version_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "suite_runs.csv"
    json_path = out_dir / "suite_runs.json"

    rows = []
    total = len(presets) * len(datasets) * len(seeds)
    idx = 0
    for preset in presets:
        for dataset in datasets:
            for seed in seeds:
                idx += 1
                version = f"{args.version_prefix}__{preset}__{dataset}__s{seed}"
                metrics_path = repo_root / "results" / version / f"{dataset}_metrics.json"
                if metrics_path.exists() and not args.rerun:
                    status = "cached"
                    elapsed_sec = None
                else:
                    cmd = [
                        sys.executable,
                        "tools/run_preset.py",
                        "--preset", preset,
                        "--dataset", dataset,
                        "--seed", str(seed),
                        "--gpu", str(args.gpu),
                        "--epochs", str(args.epochs),
                        "--eval_freq", str(args.eval_freq),
                        "--train_log_interval", str(args.train_log_interval),
                        "--n_cluster_trials", str(args.n_cluster_trials),
                        "--exp_iters", str(args.exp_iters),
                        "--version", version,
                    ]
                    print(f"[{idx}/{total}] running {preset} | {dataset} | seed={seed}")
                    print("CMD:", " ".join(cmd))
                    start = time.time()
                    subprocess.check_call(cmd, cwd=str(repo_root))
                    elapsed_sec = time.time() - start
                    status = "ran"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
                metrics = load_metrics(metrics_path)
                row = {
                    "preset": preset,
                    "dataset": dataset,
                    "seed": seed,
                    "version": version,
                    "status": status,
                    "elapsed_sec": elapsed_sec,
                    "metrics_path": str(metrics_path),
                }
                for field in METRIC_FIELDS:
                    row[field] = metrics.get(field)
                rows.append(row)

                ensure_parent(csv_path)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
