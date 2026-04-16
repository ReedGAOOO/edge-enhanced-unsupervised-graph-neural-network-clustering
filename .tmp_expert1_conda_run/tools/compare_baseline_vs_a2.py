#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DATASET_MAX_NUMS = {
    "cora": 10,
    "citeseer": 9,
    "pubmed": 5,
    "computers": 12,
    "photo": 10,
}

PRESETS = {
    "baseline": "baseline_v1",
    "a2": "a2_u2_no_adapt",
}


def parse_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_seeds(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_one(repo: Path, dataset: str, seed: int, preset: str, gpu: int, epochs: int, eval_freq: int, train_log_interval: int, hid_dim: int, dry_run: bool):
    version = f"cmp_{preset}_{dataset}_s{seed}"
    cmd = [
        sys.executable,
        "tools/run_preset.py",
        "--preset",
        preset,
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--epochs",
        str(epochs),
        "--eval_freq",
        str(eval_freq),
        "--train_log_interval",
        str(train_log_interval),
        "--hid_dim",
        str(hid_dim),
        "--version",
        version,
    ]
    if dry_run:
        cmd.append("--dry_run")
    subprocess.check_call(cmd, cwd=str(repo))
    metrics_path = repo / "results" / version / f"{dataset}_metrics.json"
    if dry_run or (not metrics_path.exists()):
        return {"nmi": float("nan"), "ari": float("nan"), "metrics_path": str(metrics_path)}
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "nmi": float(m.get("nmi_mean", float("nan"))),
        "ari": float(m.get("ari_mean", float("nan"))),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare baseline V1 vs A2(no-adapt) on selected datasets/seeds")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_csv", type=str, default="results/compare_baseline_vs_a2.csv")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    datasets = [d.lower() for d in parse_list(args.datasets)]
    seeds = parse_seeds(args.seeds)

    for d in datasets:
        if d not in DATASET_MAX_NUMS:
            raise ValueError(f"Unknown dataset: {d}")

    rows = []
    for d in datasets:
        for s in seeds:
            res_base = run_one(repo, d, s, PRESETS["baseline"], args.gpu, args.epochs, args.eval_freq, args.train_log_interval, args.hid_dim, args.dry_run)
            res_a2 = run_one(repo, d, s, PRESETS["a2"], args.gpu, args.epochs, args.eval_freq, args.train_log_interval, args.hid_dim, args.dry_run)
            rows.append(
                {
                    "dataset": d,
                    "seed": s,
                    "baseline_nmi": res_base["nmi"],
                    "a2_nmi": res_a2["nmi"],
                    "delta_nmi": (res_a2["nmi"] - res_base["nmi"]) if (res_a2["nmi"] == res_a2["nmi"] and res_base["nmi"] == res_base["nmi"]) else float("nan"),
                    "baseline_ari": res_base["ari"],
                    "a2_ari": res_a2["ari"],
                    "delta_ari": (res_a2["ari"] - res_base["ari"]) if (res_a2["ari"] == res_a2["ari"] and res_base["ari"] == res_base["ari"]) else float("nan"),
                }
            )

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = repo / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "baseline_nmi", "a2_nmi", "delta_nmi", "baseline_ari", "a2_ari", "delta_ari"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
