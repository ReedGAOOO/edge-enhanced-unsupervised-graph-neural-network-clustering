#!/usr/bin/env python3
import argparse
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

PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "v5_mid_adaptive_u2": "configs/presets/v5_mid_adaptive_u2.json",
    "a2_u2_no_adapt": "configs/presets/a2_u2_no_adapt.json",
}


def load_preset(repo_root: Path, name_or_path: str) -> dict:
    p = Path(name_or_path)
    if p.exists():
        target = p
    elif name_or_path in PRESET_MAP:
        target = repo_root / PRESET_MAP[name_or_path]
    else:
        raise FileNotFoundError(f"Unknown preset: {name_or_path}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run DSE preset (baseline/V5/A2)")
    parser.add_argument("--preset", type=str, default="a2_u2_no_adapt")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_freq", type=int, default=20)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_cluster_trials", type=int, default=1)
    parser.add_argument("--exp_iters", type=int, default=1)
    parser.add_argument("--max_nums", type=int, default=-1, help="Override max_nums; -1 means auto by dataset")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    if args.max_nums > 0:
        max_nums = args.max_nums
    else:
        if dataset_key not in DATASET_MAX_NUMS:
            raise ValueError(f"Unknown dataset for auto max_nums: {args.dataset}")
        max_nums = DATASET_MAX_NUMS[dataset_key]

    preset = load_preset(repo_root, args.preset)

    if args.version.strip():
        version = args.version.strip()
    else:
        version = f"{Path(args.preset).stem}_{dataset_key}_s{args.seed}"

    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--eval_freq",
        str(args.eval_freq),
        "--train_log_interval",
        str(args.train_log_interval),
        "--exp_iters",
        str(args.exp_iters),
        "--n_cluster_trials",
        str(args.n_cluster_trials),
        "--hid_dim",
        str(args.hid_dim),
        "--max_nums",
        str(max_nums),
        "--seed",
        str(args.seed),
        "--version",
        version,
        "--gpu",
        str(args.gpu),
        "--edge_variant",
        str(preset.get("edge_variant", "V1")),
        "--edge_hybrid_alpha",
        str(preset.get("edge_hybrid_alpha", 0.5)),
        "--edge_feat_temp",
        str(preset.get("edge_feat_temp", 1.0)),
        "--edge_fusion_gamma",
        str(preset.get("edge_fusion_gamma", 1.0)),
        "--edge_fusion_gamma_sched_epochs",
        str(preset.get("edge_fusion_gamma_sched_epochs", 0)),
        "--edge_confidence_quantile",
        str(preset.get("edge_confidence_quantile", 0.0)),
        "--edge_adaptive_alpha_strength",
        str(preset.get("edge_adaptive_alpha_strength", 2.0)),
        "--edge_adaptive_alpha_bias",
        str(preset.get("edge_adaptive_alpha_bias", 0.0)),
        "--edge_reliability_temp",
        str(preset.get("edge_reliability_temp", 1.0)),
    ]

    if preset.get("edge_fusion_gamma_start", None) is not None:
        cmd += ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])]
    if preset.get("edge_fusion_gamma_end", None) is not None:
        cmd += ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])]
    if bool(preset.get("edge_adaptive_alpha", False)):
        cmd.append("--edge_adaptive_alpha")

    print("CMD:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.check_call(cmd, cwd=str(repo_root))
    metrics_path = repo_root / "results" / version / f"{args.dataset}_metrics.json"
    if metrics_path.exists():
        print(f"[ok] metrics: {metrics_path}")
    else:
        print(f"[warn] metrics file not found: {metrics_path}")


if __name__ == "__main__":
    main()
