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

PRESET_MAP = {
    "baseline_v1": "configs/presets/baseline_v1.json",
    "v5_mid_adaptive_u2": "configs/presets/v5_mid_adaptive_u2.json",
    "a2_u2_no_adapt": "configs/presets/a2_u2_no_adapt.json",
    # ECHF family (new canonical names).
    "b15_echf_branch": "configs/presets/b15_echf_branch.json",
    "b15_echf_branch_s60": "configs/presets/b15_echf_branch_s60.json",
    "g15_echf_main": "configs/presets/g15_echf_main.json",
    "g15_echf_noadapt": "configs/presets/g15_echf_noadapt.json",
    "g17_v5_temp15": "configs/presets/g17_v5_temp15.json",
    "g13_edge_lorentz_l1": "configs/presets/g13_edge_lorentz_l1.json",
    "g13_edge_lorentz_l2": "configs/presets/g13_edge_lorentz_l2.json",
    "g13_edge_lorentz_noadapt": "configs/presets/g13_edge_lorentz_noadapt.json",
    "g20_se_consistent_main": "configs/presets/g20_se_consistent_main.json",
    "b40_v31_msgcond": "configs/presets/b40_v31_msgcond.json",
    "b43_v31_msgcond_matchonly": "configs/presets/b43_v31_msgcond_matchonly.json",
    "b44_v31_msgcond_gs020": "configs/presets/b44_v31_msgcond_gs020.json",
    "b45_v31_msgcond_gs050": "configs/presets/b45_v31_msgcond_gs050.json",
    "b46_v31_msgcond_confgate": "configs/presets/b46_v31_msgcond_confgate.json",
    "b47_v31_msgcond_gs050_matchonly": "configs/presets/b47_v31_msgcond_gs050_matchonly.json",
    "b48_v31_msgcond_gs050_confgate": "configs/presets/b48_v31_msgcond_gs050_confgate.json",
    "b30_dualscalar": "configs/presets/b30_dualscalar.json",
    "b31_dualscalar_assign": "configs/presets/b31_dualscalar_assign.json",
    "b32_dualscalar_assign_hier": "configs/presets/b32_dualscalar_assign_hier.json",
    "b33_dualscalar_assign_hier_aug": "configs/presets/b33_dualscalar_assign_hier_aug.json",
    "b34_v33_augsmall": "configs/presets/b34_v33_augsmall.json",
    "b35_v33_augpositive": "configs/presets/b35_v33_augpositive.json",
    "b36_v33_augpositive_small": "configs/presets/b36_v33_augpositive_small.json",
    "b37_v32_hardhier": "configs/presets/b37_v32_hardhier.json",
    "b38_v32_topk3": "configs/presets/b38_v32_topk3.json",
    "b50_v40_relation_state": "configs/presets/b50_v40_relation_state.json",
    # Backward-compatible aliases.
    "b15_pathb_v12_hier": "configs/presets/b15_pathb_v12_hier.json",
    "g17_temp1p5_mainline": "configs/presets/g17_temp1p5_mainline.json",
    "g15_default_hetero": "configs/presets/g15_default_hetero.json",
    "g15_noadapt_hetero": "configs/presets/g15_noadapt_hetero.json",
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

    parser = argparse.ArgumentParser(description="Run edge-fusion preset")
    parser.add_argument("--preset", type=str, default="b45_v31_msgcond_gs050")
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
    parser.add_argument("--list_presets", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for k in sorted(PRESET_MAP.keys()):
            print(f"- {k}: {PRESET_MAP[k]}")
        return

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
        "--save_path",
        f"{version}.pt",
        "--gpu",
        str(args.gpu),
        "--edge_variant",
        str(preset.get("edge_variant", "V1")),
        "--edge_hybrid_alpha",
        str(preset.get("edge_hybrid_alpha", 0.5)),
        "--edge_feat_temp",
        str(preset.get("edge_feat_temp", 1.0)),
        "--edge_input_prior_alpha",
        str(preset.get("edge_input_prior_alpha", 0.0)),
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
        "--edge_attr_hidden_dim",
        str(preset.get("edge_attr_hidden_dim", 64)),
        "--edge_attr_fusion_scale",
        str(preset.get("edge_attr_fusion_scale", 1.0)),
        "--edge_relation_channels",
        str(preset.get("edge_relation_channels", 4)),
        "--edge_relation_hidden_dim",
        str(preset.get("edge_relation_hidden_dim", 64)),
        "--edge_relation_assign_scale",
        str(preset.get("edge_relation_assign_scale", 1.0)),
        "--edge_relation_reg_lambda",
        str(preset.get("edge_relation_reg_lambda", 1e-3)),
        "--edge_attr_pool_topk",
        str(preset.get("edge_attr_pool_topk", 1)),
        "--edge_msg_gate_scale",
        str(preset.get("edge_msg_gate_scale", 0.35)),
        "--edge_msg_confidence_temp",
        str(preset.get("edge_msg_confidence_temp", 1.0)),
        "--edge_attr_weight_blend",
        str(preset.get("edge_attr_weight_blend", 0.0)),
        "--edge_attr_weight_temp",
        str(preset.get("edge_attr_weight_temp", 1.0)),
        "--edge_attr_weight_apply_to",
        str(preset.get("edge_attr_weight_apply_to", "si_only")),
        "--edge_weight_learn_reg_lambda",
        str(preset.get("edge_weight_learn_reg_lambda", 0.02)),
        "--edge_weight_learn_logclip",
        str(preset.get("edge_weight_learn_logclip", 0.8)),
        "--edge_weight_learn_temp",
        str(preset.get("edge_weight_learn_temp", 1.0)),
        "--edge_weight_learn_apply_to",
        str(preset.get("edge_weight_learn_apply_to", "both")),
        "--edge_aug_prior_scale",
        str(preset.get("edge_aug_prior_scale", 0.0)),
        "--edge_aug_prior_mode",
        str(preset.get("edge_aug_prior_mode", "raw")),
    ]

    if preset.get("edge_fusion_gamma_start", None) is not None:
        cmd += ["--edge_fusion_gamma_start", str(preset["edge_fusion_gamma_start"])]
    if preset.get("edge_fusion_gamma_end", None) is not None:
        cmd += ["--edge_fusion_gamma_end", str(preset["edge_fusion_gamma_end"])]
    if bool(preset.get("edge_adaptive_alpha", False)):
        cmd.append("--edge_adaptive_alpha")
    if bool(preset.get("edge_attr_hierarchical", False)):
        cmd.append("--edge_attr_hierarchical")
    if bool(preset.get("edge_msg_conditioned", False)):
        cmd.append("--edge_msg_conditioned")
    if bool(preset.get("edge_msg_matched_only", False)):
        cmd.append("--edge_msg_matched_only")
    if bool(preset.get("edge_msg_confidence_gate", False)):
        cmd.append("--edge_msg_confidence_gate")
    if bool(preset.get("edge_attr_pool_confidence", False)):
        cmd.append("--edge_attr_pool_confidence")
    if bool(preset.get("append_generic_edge_attr", False)):
        cmd.append("--append_generic_edge_attr")

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
