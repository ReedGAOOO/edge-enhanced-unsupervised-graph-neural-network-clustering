"""
Ultra-light merged current mainline code for expert review.
Generated from the current repository state.
Included files:
  - main.py
  - exp.py
  - modules/model.py
  - modules/layers.py
  - modules/dsi.py
  - tools/run_preset.py
"""


# ========================================================================================
# FILE: main.py
# ========================================================================================

import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict


def set_seed(seed, deterministic=True):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass


parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='KarateClub')
parser.add_argument('--task', type=str, default='Clustering',
                    choices=['Clustering'])
parser.add_argument('--root_path', type=str, default='data')
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/FootBall.log")

parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=1e-2)
parser.add_argument('--max_nums', type=int, nargs='+', default=[4], help="such as [50, 10]")
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--nonlin', type=str, default="leaky_relu")
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--n_cluster_trials', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--knn', type=int, default=8)
parser.add_argument('--knn_mode', type=str, default='auto', choices=['auto', 'dense', 'edge'],
                    help='KNN augmentation mode in se_loss. dense: full pairwise; edge: score only existing edges; auto: edge for large graphs.')
parser.add_argument('--knn_auto_threshold', type=int, default=20000,
                    help='When knn_mode=auto and num_nodes exceeds this threshold, switch to edge mode.')
parser.add_argument("--epsInt", type=int, default=8)
parser.add_argument('--edge_variant', type=str, default='V1', choices=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20', 'V30', 'V31', 'V32', 'V33'],
                    help='V1: plain adjacency; V2: structural pre-weight; '
                         'V3: feature-similarity pre-weight; V4: hybrid pre-weight; '
                         'V5: hybrid + attention-stage edge fusion; '
                         'V6: edge-attr gated fusion; V7: edge-attr gated fusion + alignment residual; '
                         'V8: calibrated mixture of structural/edge-attr fusion; '
                         'V12: V5 residual fusion with calibrated edge-attr correction; '
                         'V13: Lorentzized edge-attr residual fusion on assignment score; '
                         'V20: SE-consistent learnable edge weighting with bounded regularization; '
                         'V30: dual message/SI edge scalarization; '
                         'V31: V30 + assignment residual; '
                         'V32: V31 + hierarchical edge-state pooling; '
                         'V33: V32 + edge-aware augment prior.')
parser.add_argument('--edge_hybrid_alpha', type=float, default=0.5,
                    help='Feature weight in hybrid edge variant V4/V5.')
parser.add_argument('--edge_feat_temp', type=float, default=1.0,
                    help='Temperature for feature-similarity edge weighting.')
parser.add_argument('--edge_input_prior_alpha', type=float, default=0.0,
                    help='Blend ratio for dataset-provided edge weights when using V2/V3/V4/V5.')
parser.add_argument('--edge_fusion_gamma', type=float, default=1.0,
                    help='Fusion strength for V5 attention-stage edge fusion.')
parser.add_argument('--edge_fusion_gamma_start', type=float, default=None,
                    help='Optional start value for scheduled V5 fusion gamma.')
parser.add_argument('--edge_fusion_gamma_end', type=float, default=None,
                    help='Optional end value for scheduled V5 fusion gamma.')
parser.add_argument('--edge_fusion_gamma_sched_epochs', type=int, default=0,
                    help='Warmup epochs for linearly scheduling V5 fusion gamma.')
parser.add_argument('--edge_confidence_quantile', type=float, default=0.0,
                    help='Optional quantile filtering (0~1) for V5 edge-fusion confidence.')
parser.add_argument('--edge_adaptive_alpha', action='store_true',
                    help='Enable graph-adaptive scaling for V5 edge-fusion term.')
parser.add_argument('--edge_adaptive_alpha_strength', type=float, default=2.0,
                    help='Strength of graph-adaptive edge-fusion scaling.')
parser.add_argument('--edge_adaptive_alpha_bias', type=float, default=0.0,
                    help='Bias of graph-adaptive edge-fusion scaling.')
parser.add_argument('--edge_reliability_temp', type=float, default=1.0,
                    help='Temperature for per-edge reliability in V5 edge fusion.')
parser.add_argument('--edge_attr_hidden_dim', type=int, default=64,
                    help='Hidden size for edge-attribute encoder in V6/V7.')
parser.add_argument('--edge_attr_fusion_scale', type=float, default=1.0,
                    help='Fusion scale for edge-attribute terms in V6/V7.')
parser.add_argument('--append_generic_edge_attr', action='store_true',
                    help='Append generic edge features to provided edge_attr (for custom datasets).')
parser.add_argument('--edge_attr_weight_blend', type=float, default=0.0,
                    help='Path-A: blend ratio for edge-attr-derived weights into SI graph (0~1).')
parser.add_argument('--edge_attr_weight_temp', type=float, default=1.0,
                    help='Path-A: temperature when mapping edge_attr to weights.')
parser.add_argument('--edge_attr_weight_apply_to', type=str, default='si_only', choices=['si_only', 'both'],
                    help='Path-A: apply edge-attr-derived weights to SI graph only, or both SI and message graph.')
parser.add_argument('--edge_attr_hierarchical', action='store_true',
                    help='Path-B: propagate/coarsen edge attributes across hierarchy levels.')
parser.add_argument('--edge_weight_learn_reg_lambda', type=float, default=0.02,
                    help='V20: regularization strength for learnable edge-weight log-ratio.')
parser.add_argument('--edge_weight_learn_logclip', type=float, default=0.8,
                    help='V20: absolute clip on learned edge log-ratio before exp mapping.')
parser.add_argument('--edge_weight_learn_temp', type=float, default=1.0,
                    help='V20: temperature on learned edge score before tanh clipping.')
parser.add_argument('--edge_weight_learn_apply_to', type=str, default='both', choices=['si_only', 'both'],
                    help='V20: apply learned edge weighting to SI graph only, or both SI/message graphs.')
parser.add_argument('--edge_attr_pool_topk', type=int, default=1,
                    help='For hierarchical edge-state pooling: use top-k assignment parents per endpoint (1 = hard argmax).')
parser.add_argument('--edge_msg_conditioned', action='store_true',
                    help='Enable edge-conditioned message gating inside leaf Lorentz aggregation.')
parser.add_argument('--edge_msg_gate_scale', type=float, default=0.35,
                    help='Bounded log-scale for edge-conditioned message gating.')
parser.add_argument('--edge_msg_matched_only', action='store_true',
                    help='Apply message gating only on edges with aligned/original edge attributes.')
parser.add_argument('--edge_msg_confidence_gate', action='store_true',
                    help='Shrink message gating on low-confidence edge-gate scores.')
parser.add_argument('--edge_msg_confidence_temp', type=float, default=1.0,
                    help='Temperature for confidence-gated message conditioning.')
parser.add_argument('--edge_attr_pool_confidence', action='store_true',
                    help='Use assignment-confidence weighting during hierarchical edge-state pooling.')
parser.add_argument('--edge_attr_pool_conf_power', type=float, default=1.0,
                    help='Exponent applied to assignment-confidence weights in hierarchical edge-state pooling.')
parser.add_argument('--edge_aug_prior_scale', type=float, default=0.0,
                    help='For V33: additive scale of edge-attr prior on augment-graph candidate scores.')
parser.add_argument('--edge_aug_prior_mode', type=str, default='raw', choices=['raw', 'positive', 'tanh'],
                    help='For V33: transform mode for augment prior head before adding to candidate scores.')
parser.add_argument('--known_only_eval', action='store_true',
                    help='For datasets with explicit unknown label mapping, remap unknown labels to -1 during evaluation.')
parser.add_argument('--train_log_interval', type=int, default=1,
                    help='Epoch interval for Stage2 training loss logs.')
parser.add_argument('--amp_bf16', action='store_true',
                    help='Enable CUDA autocast with bfloat16 for lower memory usage.')

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')
parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                    help='Enable deterministic algorithms for better reproducibility.')
parser.add_argument('--non_deterministic', dest='deterministic', action='store_false',
                    help='Disable deterministic algorithms for maximum throughput.')

# GPU
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true',
                    help='Use GPU when available (default).')
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false',
                    help='Force CPU execution.')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')
parser.add_argument('--seed', type=int, default=3047)
parser.set_defaults(use_gpu=True, deterministic=True)


configs = parser.parse_args()
set_seed(configs.seed, deterministic=configs.deterministic)
# with open(f'./configs/{configs.dataset}.json', 'wt') as f:
#     json.dump(vars(configs), f, indent=4)

# configs_dict = vars(configs)
# with open(f'./configs/{configs.dataset}.json', 'rt') as f:
#     configs_dict.update(json.load(f))
# configs = DotDict(configs_dict)
# f.close()

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.dataset}"):
    os.mkdir(f"./results/{configs.dataset}")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
metrics = exp.train()
metrics_path = f"./results/{configs.version}/{configs.dataset}_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
logger.info(f"Saved metrics to {metrics_path}")
torch.cuda.empty_cache()


# ========================================================================================
# FILE: exp.py
# ========================================================================================

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from torch.optim import AdamW

from data import load_data
from logger import create_logger
from modules.dsi import DSI
from utils.eval_utils import cluster_metrics


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            gpu_id = int(getattr(self.configs, "gpu", 0))
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                gpu_id = 0
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.amp_bf16 = bool(getattr(self.configs, "amp_bf16", False)) and self.device.type == "cuda"

    @staticmethod
    def _mean_std(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(valid.mean()), "std": float(valid.std())}

    def _edge_branch_diagnostics(self, edge_stats: Dict[str, float]) -> Dict[str, float]:
        variant = str(getattr(self.configs, "edge_variant", "V1")).upper()
        diag = {
            "diag_factor_live": float("nan"),
            "diag_dual_live": float("nan"),
            "diag_msg_live": float("nan"),
            "diag_assign_live": float("nan"),
            "diag_hier_live": float("nan"),
            "diag_aug_live": float("nan"),
            "diag_factor_shift": float("nan"),
            "diag_dual_gap": float("nan"),
            "diag_msg_gate_shift": float("nan"),
            "diag_aug_bias_abs": float("nan"),
            "diag_dead_branch_count": 0.0,
            "diag_required_branch_count": 0.0,
            "diag_all_required_live": 1.0,
        }

        required = 0
        dead = 0

        if variant in {"V20", "V30", "V31", "V32", "V33"}:
            factor_shift = abs(float(edge_stats.get("edge_factor_mean", 1.0)) - 1.0)
            factor_sigma = abs(float(edge_stats.get("edge_factor_std", 0.0)))
            edge_reg = abs(float(edge_stats.get("edge_reg", 0.0)))
            factor_live = float((factor_shift > 5e-3) or (factor_sigma > 5e-3) or (edge_reg > 1e-4))
            diag["diag_factor_live"] = factor_live
            diag["diag_factor_shift"] = factor_shift
            required += 1
            dead += int(factor_live < 0.5)

        if variant in {"V30", "V31", "V32", "V33"}:
            factor_msg = float(edge_stats.get("edge_factor_msg_mean", 1.0))
            factor_si = float(edge_stats.get("edge_factor_si_mean", 1.0))
            dual_gap = abs(
                factor_msg - factor_si
            )
            dual_live = float(
                (dual_gap > 2e-3)
                or ((abs(factor_msg - 1.0) > 5e-3) and (abs(factor_si - 1.0) > 5e-3))
            )
            diag["diag_dual_live"] = dual_live
            diag["diag_dual_gap"] = dual_gap
            required += 1
            dead += int(dual_live < 0.5)

        if bool(getattr(self.configs, "edge_msg_conditioned", False)):
            msg_shift = abs(float(edge_stats.get("msg_gate_factor_mean", 1.0)) - 1.0)
            msg_sigma = abs(float(edge_stats.get("msg_gate_factor_std", 0.0)))
            msg_live = float((msg_shift > 5e-3) or (msg_sigma > 5e-3))
            diag["diag_msg_live"] = msg_live
            diag["diag_msg_gate_shift"] = msg_shift
            required += 1
            dead += int(msg_live < 0.5)

        if variant in {"V31", "V32", "V33"}:
            graph_alpha = float(edge_stats.get("graph_alpha_mean", 1.0))
            edge_rel = float(edge_stats.get("edge_reliability_mean", 1.0))
            edge_mix = float(edge_stats.get("edge_mix_beta_mean", 0.0))
            assign_live = float(
                (abs(graph_alpha - 1.0) > 5e-2)
                or (abs(edge_rel - 1.0) > 5e-2)
                or (abs(edge_mix) > 5e-2)
            )
            diag["diag_assign_live"] = assign_live
            required += 1
            dead += int(assign_live < 0.5)

        if variant in {"V32", "V33"} and bool(getattr(self.configs, "edge_attr_hierarchical", False)):
            hier_levels = float(edge_stats.get("hier_edge_levels_active_ratio", 0.0))
            hier_nonzero = float(edge_stats.get("hier_edge_nonzero_ratio", 0.0))
            hier_mean_abs = float(edge_stats.get("hier_edge_mean_abs", 0.0))
            hier_live = float((hier_levels > 0.0) and ((hier_nonzero > 5e-2) or (hier_mean_abs > 1e-4)))
            diag["diag_hier_live"] = hier_live
            required += 1
            dead += int(hier_live < 0.5)

        if variant == "V33" and float(getattr(self.configs, "edge_aug_prior_scale", 0.0)) > 0.0:
            aug_mean = float(edge_stats.get("edge_aug_bias_mean", 0.0))
            aug_std = float(edge_stats.get("edge_aug_bias_std", 0.0))
            aug_live = float((abs(aug_mean) > 1e-4) or (aug_std > 1e-3))
            diag["diag_aug_live"] = aug_live
            diag["diag_aug_bias_abs"] = abs(aug_mean)
            required += 1
            dead += int(aug_live < 0.5)

        diag["diag_dead_branch_count"] = float(dead)
        diag["diag_required_branch_count"] = float(required)
        diag["diag_all_required_live"] = float(dead == 0) if required > 0 else 1.0
        return diag

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs).to(device)
        edge_attr_dim = 1
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.dim() == 2:
            edge_attr_dim = int(data.edge_attr.shape[1])

        if hasattr(data, "y") and data.y is not None and data.y.numel() > 0:
            known_ratio = float((data.y >= 0).float().mean().item())
            logger.info(
                f"[Data] known_label_ratio={known_ratio:.4f}, "
                f"known_only_eval={bool(getattr(self.configs, 'known_only_eval', False))}"
            )

        run_stats: List[Dict[str, float]] = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            model = DSI(
                in_dim=data.x.shape[1],
                hid_dim=self.configs.hid_dim,
                num_nodes=data.x.shape[0],
                temperature=self.configs.temperature,
                dropout=self.configs.dropout,
                nonlin_str=self.configs.nonlin,
                max_nums=self.configs.max_nums,
                alpha=self.configs.alpha,
                knn=self.configs.knn,
                edge_variant=getattr(self.configs, "edge_variant", "V1"),
                edge_fusion_gamma=getattr(self.configs, "edge_fusion_gamma", 1.0),
                edge_confidence_quantile=getattr(self.configs, "edge_confidence_quantile", 0.0),
                edge_adaptive_alpha=bool(getattr(self.configs, "edge_adaptive_alpha", False)),
                edge_adaptive_alpha_strength=float(getattr(self.configs, "edge_adaptive_alpha_strength", 2.0)),
                edge_adaptive_alpha_bias=float(getattr(self.configs, "edge_adaptive_alpha_bias", 0.0)),
                edge_reliability_temp=float(getattr(self.configs, "edge_reliability_temp", 1.0)),
                edge_attr_hidden_dim=int(getattr(self.configs, "edge_attr_hidden_dim", 64)),
                edge_attr_fusion_scale=float(getattr(self.configs, "edge_attr_fusion_scale", 1.0)),
                edge_attr_dim=edge_attr_dim,
                edge_attr_hierarchical=bool(getattr(self.configs, "edge_attr_hierarchical", False)),
                edge_attr_pool_topk=int(getattr(self.configs, "edge_attr_pool_topk", 1)),
                edge_msg_conditioned=bool(getattr(self.configs, "edge_msg_conditioned", False)),
                edge_msg_gate_scale=float(getattr(self.configs, "edge_msg_gate_scale", 0.35)),
                edge_msg_matched_only=bool(getattr(self.configs, "edge_msg_matched_only", False)),
                edge_msg_confidence_gate=bool(getattr(self.configs, "edge_msg_confidence_gate", False)),
                edge_msg_confidence_temp=float(getattr(self.configs, "edge_msg_confidence_temp", 1.0)),
                edge_attr_pool_confidence=bool(getattr(self.configs, "edge_attr_pool_confidence", False)),
                edge_attr_pool_conf_power=float(getattr(self.configs, "edge_attr_pool_conf_power", 1.0)),
                edge_weight_learn_reg_lambda=float(getattr(self.configs, "edge_weight_learn_reg_lambda", 0.02)),
                edge_weight_learn_logclip=float(getattr(self.configs, "edge_weight_learn_logclip", 0.8)),
                edge_weight_learn_temp=float(getattr(self.configs, "edge_weight_learn_temp", 1.0)),
                edge_weight_learn_apply_to=str(getattr(self.configs, "edge_weight_learn_apply_to", "both")),
                edge_aug_prior_scale=float(getattr(self.configs, "edge_aug_prior_scale", 0.0)),
                edge_aug_prior_mode=str(getattr(self.configs, "edge_aug_prior_mode", "raw")),
                knn_mode=str(getattr(self.configs, "knn_mode", "auto")),
                knn_auto_threshold=int(getattr(self.configs, "knn_auto_threshold", 20000)),
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == "Clustering":
                stat = self.train_clu(data, model, optimizer, logger)
                run_stats.append(stat)

        if self.configs.task != "Clustering":
            return {}
        if not run_stats:
            return {}

        nmi_stats = self._mean_std([s["final_nmi"] for s in run_stats])
        ari_stats = self._mean_std([s["final_ari"] for s in run_stats])
        acc_stats = self._mean_std([s["final_acc"] for s in run_stats])
        nmi_trial_std_stats = self._mean_std([s["final_nmi_trial_std"] for s in run_stats])
        ari_trial_std_stats = self._mean_std([s["final_ari_trial_std"] for s in run_stats])
        stability_pair_stats = self._mean_std([s["stability_pair_nmi"] for s in run_stats])
        si_loss_stats = self._mean_std([s["si_loss"] for s in run_stats])
        modularity_stats = self._mean_std([s["modularity"] for s in run_stats])
        conductance_stats = self._mean_std([s["conductance_mean"] for s in run_stats])
        conductance_weighted_stats = self._mean_std([s["conductance_weighted"] for s in run_stats])
        pred_k_stats = self._mean_std([s["pred_n_clusters"] for s in run_stats])
        pred_cv_stats = self._mean_std([s["pred_cluster_size_cv"] for s in run_stats])
        best_epoch_stats = self._mean_std([s["best_epoch"] for s in run_stats])
        best_loss_stats = self._mean_std([s["best_train_loss"] for s in run_stats])
        graph_alpha_stats = self._mean_std([s.get("final_graph_alpha", float("nan")) for s in run_stats])
        edge_rel_stats = self._mean_std([s.get("final_edge_reliability", float("nan")) for s in run_stats])
        edge_mix_stats = self._mean_std([s.get("final_edge_mix_beta", float("nan")) for s in run_stats])
        edge_factor_mean_stats = self._mean_std([s.get("final_edge_factor_mean", float("nan")) for s in run_stats])
        edge_factor_std_stats = self._mean_std([s.get("final_edge_factor_std", float("nan")) for s in run_stats])
        edge_factor_msg_stats = self._mean_std([s.get("final_edge_factor_msg_mean", float("nan")) for s in run_stats])
        edge_factor_msg_std_stats = self._mean_std([s.get("final_edge_factor_msg_std", float("nan")) for s in run_stats])
        edge_factor_si_stats = self._mean_std([s.get("final_edge_factor_si_mean", float("nan")) for s in run_stats])
        edge_factor_si_std_stats = self._mean_std([s.get("final_edge_factor_si_std", float("nan")) for s in run_stats])
        msg_gate_factor_stats = self._mean_std([s.get("final_msg_gate_factor_mean", float("nan")) for s in run_stats])
        msg_gate_factor_std_stats = self._mean_std([s.get("final_msg_gate_factor_std", float("nan")) for s in run_stats])
        edge_aug_stats = self._mean_std([s.get("final_edge_aug_bias_mean", float("nan")) for s in run_stats])
        edge_aug_std_stats = self._mean_std([s.get("final_edge_aug_bias_std", float("nan")) for s in run_stats])
        edge_reg_stats = self._mean_std([s.get("final_edge_reg", float("nan")) for s in run_stats])
        hier_levels_stats = self._mean_std([s.get("final_hier_edge_levels_active_ratio", float("nan")) for s in run_stats])
        hier_nonzero_stats = self._mean_std([s.get("final_hier_edge_nonzero_ratio", float("nan")) for s in run_stats])
        hier_abs_stats = self._mean_std([s.get("final_hier_edge_mean_abs", float("nan")) for s in run_stats])
        diag_factor_live_stats = self._mean_std([s.get("diag_factor_live", float("nan")) for s in run_stats])
        diag_dual_live_stats = self._mean_std([s.get("diag_dual_live", float("nan")) for s in run_stats])
        diag_msg_live_stats = self._mean_std([s.get("diag_msg_live", float("nan")) for s in run_stats])
        diag_assign_live_stats = self._mean_std([s.get("diag_assign_live", float("nan")) for s in run_stats])
        diag_hier_live_stats = self._mean_std([s.get("diag_hier_live", float("nan")) for s in run_stats])
        diag_aug_live_stats = self._mean_std([s.get("diag_aug_live", float("nan")) for s in run_stats])
        diag_dead_count_stats = self._mean_std([s.get("diag_dead_branch_count", float("nan")) for s in run_stats])
        diag_all_live_stats = self._mean_std([s.get("diag_all_required_live", float("nan")) for s in run_stats])

        logger.info(
            f"NMI: {nmi_stats['mean']}+-{nmi_stats['std']}, "
            f"ARI: {ari_stats['mean']}+-{ari_stats['std']}, "
            f"SI-loss: {si_loss_stats['mean']}+-{si_loss_stats['std']}"
        )

        return {
            "dataset": self.configs.dataset,
            "amp_bf16": bool(self.amp_bf16),
            "known_only_eval": bool(getattr(self.configs, "known_only_eval", False)),
            "known_label_ratio": float(getattr(data, "known_label_ratio", float("nan"))),
            "edge_variant": getattr(self.configs, "edge_variant", "V1"),
            "edge_fusion_gamma": float(getattr(self.configs, "edge_fusion_gamma", 1.0)),
            "edge_fusion_gamma_start": getattr(self.configs, "edge_fusion_gamma_start", None),
            "edge_fusion_gamma_end": getattr(self.configs, "edge_fusion_gamma_end", None),
            "edge_fusion_gamma_sched_epochs": int(getattr(self.configs, "edge_fusion_gamma_sched_epochs", 0)),
            "edge_confidence_quantile": float(getattr(self.configs, "edge_confidence_quantile", 0.0)),
            "edge_adaptive_alpha": bool(getattr(self.configs, "edge_adaptive_alpha", False)),
            "edge_adaptive_alpha_strength": float(getattr(self.configs, "edge_adaptive_alpha_strength", 2.0)),
            "edge_adaptive_alpha_bias": float(getattr(self.configs, "edge_adaptive_alpha_bias", 0.0)),
            "edge_reliability_temp": float(getattr(self.configs, "edge_reliability_temp", 1.0)),
            "edge_attr_hidden_dim": int(getattr(self.configs, "edge_attr_hidden_dim", 64)),
            "edge_attr_fusion_scale": float(getattr(self.configs, "edge_attr_fusion_scale", 1.0)),
            "edge_attr_hierarchical": bool(getattr(self.configs, "edge_attr_hierarchical", False)),
            "edge_attr_pool_topk": int(getattr(self.configs, "edge_attr_pool_topk", 1)),
            "edge_msg_conditioned": bool(getattr(self.configs, "edge_msg_conditioned", False)),
            "edge_msg_gate_scale": float(getattr(self.configs, "edge_msg_gate_scale", 0.35)),
            "edge_msg_matched_only": bool(getattr(self.configs, "edge_msg_matched_only", False)),
            "edge_msg_confidence_gate": bool(getattr(self.configs, "edge_msg_confidence_gate", False)),
            "edge_msg_confidence_temp": float(getattr(self.configs, "edge_msg_confidence_temp", 1.0)),
            "edge_attr_pool_confidence": bool(getattr(self.configs, "edge_attr_pool_confidence", False)),
            "edge_attr_pool_conf_power": float(getattr(self.configs, "edge_attr_pool_conf_power", 1.0)),
            "edge_weight_learn_reg_lambda": float(getattr(self.configs, "edge_weight_learn_reg_lambda", 0.02)),
            "edge_weight_learn_logclip": float(getattr(self.configs, "edge_weight_learn_logclip", 0.8)),
            "edge_weight_learn_temp": float(getattr(self.configs, "edge_weight_learn_temp", 1.0)),
            "edge_weight_learn_apply_to": str(getattr(self.configs, "edge_weight_learn_apply_to", "both")),
            "edge_aug_prior_scale": float(getattr(self.configs, "edge_aug_prior_scale", 0.0)),
            "edge_aug_prior_mode": str(getattr(self.configs, "edge_aug_prior_mode", "raw")),
            "edge_attr_weight_blend": float(getattr(self.configs, "edge_attr_weight_blend", 0.0)),
            "edge_attr_weight_temp": float(getattr(self.configs, "edge_attr_weight_temp", 1.0)),
            "edge_attr_weight_apply_to": str(getattr(self.configs, "edge_attr_weight_apply_to", "si_only")),
            "knn_mode": str(getattr(self.configs, "knn_mode", "auto")),
            "knn_auto_threshold": int(getattr(self.configs, "knn_auto_threshold", 20000)),
            "acc_mean": acc_stats["mean"],
            "acc_std": acc_stats["std"],
            "nmi_mean": nmi_stats["mean"],
            "nmi_std": nmi_stats["std"],
            "ari_mean": ari_stats["mean"],
            "ari_std": ari_stats["std"],
            "nmi_trial_std_mean": nmi_trial_std_stats["mean"],
            "nmi_trial_std_std": nmi_trial_std_stats["std"],
            "ari_trial_std_mean": ari_trial_std_stats["mean"],
            "ari_trial_std_std": ari_trial_std_stats["std"],
            "stability_pair_nmi_mean": stability_pair_stats["mean"],
            "stability_pair_nmi_std": stability_pair_stats["std"],
            "si_loss_mean": si_loss_stats["mean"],
            "si_loss_std": si_loss_stats["std"],
            "modularity_mean": modularity_stats["mean"],
            "modularity_std": modularity_stats["std"],
            "conductance_mean": conductance_stats["mean"],
            "conductance_std": conductance_stats["std"],
            "conductance_weighted_mean": conductance_weighted_stats["mean"],
            "conductance_weighted_std": conductance_weighted_stats["std"],
            "pred_n_clusters_mean": pred_k_stats["mean"],
            "pred_n_clusters_std": pred_k_stats["std"],
            "pred_cluster_size_cv_mean": pred_cv_stats["mean"],
            "pred_cluster_size_cv_std": pred_cv_stats["std"],
            "best_epoch_mean": best_epoch_stats["mean"],
            "best_epoch_std": best_epoch_stats["std"],
            "best_train_loss_mean": best_loss_stats["mean"],
            "best_train_loss_std": best_loss_stats["std"],
            "final_graph_alpha_mean": graph_alpha_stats["mean"],
            "final_graph_alpha_std": graph_alpha_stats["std"],
            "final_edge_reliability_mean": edge_rel_stats["mean"],
            "final_edge_reliability_std": edge_rel_stats["std"],
            "final_edge_mix_beta_mean": edge_mix_stats["mean"],
            "final_edge_mix_beta_std": edge_mix_stats["std"],
            "final_edge_factor_mean_mean": edge_factor_mean_stats["mean"],
            "final_edge_factor_mean_std": edge_factor_mean_stats["std"],
            "final_edge_factor_std_mean": edge_factor_std_stats["mean"],
            "final_edge_factor_std_std": edge_factor_std_stats["std"],
            "final_edge_factor_msg_mean": edge_factor_msg_stats["mean"],
            "final_edge_factor_msg_std": edge_factor_msg_stats["std"],
            "final_edge_factor_msg_sigma_mean": edge_factor_msg_std_stats["mean"],
            "final_edge_factor_msg_sigma_std": edge_factor_msg_std_stats["std"],
            "final_edge_factor_si_mean": edge_factor_si_stats["mean"],
            "final_edge_factor_si_std": edge_factor_si_stats["std"],
            "final_edge_factor_si_sigma_mean": edge_factor_si_std_stats["mean"],
            "final_edge_factor_si_sigma_std": edge_factor_si_std_stats["std"],
            "final_msg_gate_factor_mean": msg_gate_factor_stats["mean"],
            "final_msg_gate_factor_std": msg_gate_factor_stats["std"],
            "final_msg_gate_factor_sigma_mean": msg_gate_factor_std_stats["mean"],
            "final_msg_gate_factor_sigma_std": msg_gate_factor_std_stats["std"],
            "final_edge_aug_bias_mean": edge_aug_stats["mean"],
            "final_edge_aug_bias_std": edge_aug_std_stats["mean"],
            "final_edge_aug_bias_sigma_std": edge_aug_std_stats["std"],
            "final_edge_reg_mean": edge_reg_stats["mean"],
            "final_edge_reg_std": edge_reg_stats["std"],
            "final_hier_edge_levels_active_ratio_mean": hier_levels_stats["mean"],
            "final_hier_edge_levels_active_ratio_std": hier_levels_stats["std"],
            "final_hier_edge_nonzero_ratio_mean": hier_nonzero_stats["mean"],
            "final_hier_edge_nonzero_ratio_std": hier_nonzero_stats["std"],
            "final_hier_edge_mean_abs_mean": hier_abs_stats["mean"],
            "final_hier_edge_mean_abs_std": hier_abs_stats["std"],
            "diag_factor_live_mean": diag_factor_live_stats["mean"],
            "diag_dual_live_mean": diag_dual_live_stats["mean"],
            "diag_msg_live_mean": diag_msg_live_stats["mean"],
            "diag_assign_live_mean": diag_assign_live_stats["mean"],
            "diag_hier_live_mean": diag_hier_live_stats["mean"],
            "diag_aug_live_mean": diag_aug_live_stats["mean"],
            "diag_dead_branch_count_mean": diag_dead_count_stats["mean"],
            "diag_dead_branch_count_std": diag_dead_count_stats["std"],
            "diag_all_required_live_mean": diag_all_live_stats["mean"],
            "diag_all_required_live_std": diag_all_live_stats["std"],
            "selection_rule": "min_train_loss",
            "exp_iters": int(self.configs.exp_iters),
            "epochs": int(self.configs.epochs),
            "eval_freq": int(self.configs.eval_freq),
            "seed": int(self.configs.seed),
        }

    def _edge_fusion_gamma_for_epoch(self, epoch: int) -> float:
        base = float(getattr(self.configs, "edge_fusion_gamma", 1.0))
        start = getattr(self.configs, "edge_fusion_gamma_start", None)
        end = getattr(self.configs, "edge_fusion_gamma_end", None)
        if start is None and end is None:
            return base
        start_v = float(base if start is None else start)
        end_v = float(base if end is None else end)
        sched_epochs = int(getattr(self.configs, "edge_fusion_gamma_sched_epochs", 0))
        if sched_epochs <= 0:
            return end_v
        if sched_epochs == 1:
            return end_v
        ratio = min(1.0, max(0.0, float(epoch - 1) / float(sched_epochs - 1)))
        return start_v + ratio * (end_v - start_v)

    @staticmethod
    def _has_valid_labels(data) -> bool:
        if (not hasattr(data, "y")) or data.y is None:
            return False
        if data.y.numel() == 0:
            return False
        return bool(torch.any(data.y >= 0).item())

    def _evaluate_clustering_detailed(self, model, data):
        if not self._has_valid_labels(data):
            return {
                "acc_mean": float("nan"),
                "nmi_mean": float("nan"),
                "ari_mean": float("nan"),
                "acc_std": float("nan"),
                "nmi_std": float("nan"),
                "ari_std": float("nan"),
                "stability_pair_nmi": float("nan"),
                "predicts": [],
            }

        n_cluster_trials = max(1, int(getattr(self.configs, "n_cluster_trials", 1)))
        trues = data.y.detach().cpu().numpy()
        acc_list: List[float] = []
        nmi_list: List[float] = []
        ari_list: List[float] = []
        predicts_list: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            embed_dict, clu_mat_dict = model.get_cluster_results(data)
            for _ in range(n_cluster_trials):
                predicts = model.fix_cluster_results(clu_mat_dict[1], embed_dict, self.configs.epsInt).cpu().numpy()
                predicts_list.append(predicts)
                cm = cluster_metrics(trues, predicts)
                acc_, nmi_, ari_ = cm.evaluateFromLabel(use_acc=True)
                acc_list.append(acc_)
                nmi_list.append(nmi_)
                ari_list.append(ari_)

        pair_nmi: List[float] = []
        if len(predicts_list) >= 2:
            for i in range(len(predicts_list)):
                for j in range(i + 1, len(predicts_list)):
                    pair_nmi.append(sk_metrics.normalized_mutual_info_score(predicts_list[i], predicts_list[j]))

        return {
            "acc_mean": float(np.mean(acc_list)),
            "nmi_mean": float(np.mean(nmi_list)),
            "ari_mean": float(np.mean(ari_list)),
            "acc_std": float(np.std(acc_list)),
            "nmi_std": float(np.std(nmi_list)),
            "ari_std": float(np.std(ari_list)),
            "stability_pair_nmi": float(np.mean(pair_nmi)) if pair_nmi else float("nan"),
            "predicts": predicts_list,
        }

    @staticmethod
    def _compute_partition_structure_metrics(data, predicts: np.ndarray) -> Dict[str, float]:
        if predicts is None or predicts.size == 0:
            return {
                "modularity": float("nan"),
                "conductance_mean": float("nan"),
                "conductance_weighted": float("nan"),
                "pred_n_clusters": float("nan"),
                "pred_cluster_size_cv": float("nan"),
            }

        adj = getattr(data, "adj_si", getattr(data, "adj_msg_raw", data.adj))
        adj = adj.coalesce()
        idx = adj.indices()
        w = adj.values().float()
        src = idx[0].long()
        dst = idx[1].long()
        num_nodes = int(data.x.shape[0])

        pred = torch.as_tensor(predicts, dtype=torch.long, device=src.device)
        if pred.numel() != num_nodes:
            return {
                "modularity": float("nan"),
                "conductance_mean": float("nan"),
                "conductance_weighted": float("nan"),
                "pred_n_clusters": float("nan"),
                "pred_cluster_size_cv": float("nan"),
            }

        cluster_ids, inv = torch.unique(pred, sorted=True, return_inverse=True)
        k = int(cluster_ids.numel())
        if k <= 0:
            return {
                "modularity": float("nan"),
                "conductance_mean": float("nan"),
                "conductance_weighted": float("nan"),
                "pred_n_clusters": float("nan"),
                "pred_cluster_size_cv": float("nan"),
            }

        deg = torch.zeros(num_nodes, dtype=w.dtype, device=w.device)
        deg.index_add_(0, src, w)
        total_weight = deg.sum().clamp_min(1e-12)

        src_cluster = inv[src]
        same_cluster = (pred[src] == pred[dst])
        intra_weight = w[same_cluster].sum()

        vol_per_cluster = torch.zeros(k, dtype=w.dtype, device=w.device)
        vol_per_cluster.index_add_(0, inv, deg)

        cut_weight = w * (~same_cluster).to(w.dtype)
        cut_per_cluster = torch.zeros(k, dtype=w.dtype, device=w.device)
        cut_per_cluster.index_add_(0, src_cluster, cut_weight)

        den = torch.minimum(vol_per_cluster, total_weight - vol_per_cluster)
        valid = den > 1e-12
        phi = torch.zeros_like(den)
        phi[valid] = cut_per_cluster[valid] / den[valid]

        modularity = intra_weight / total_weight - torch.sum((vol_per_cluster / total_weight) ** 2)

        cluster_sizes = torch.bincount(inv, minlength=k).float()
        cluster_size_cv = cluster_sizes.std(unbiased=False) / cluster_sizes.mean().clamp_min(1e-12)

        if valid.any():
            conductance_mean = phi[valid].mean()
            conductance_weighted = (phi[valid] * vol_per_cluster[valid]).sum() / vol_per_cluster[valid].sum().clamp_min(1e-12)
        else:
            conductance_mean = torch.tensor(float("nan"), dtype=w.dtype, device=w.device)
            conductance_weighted = torch.tensor(float("nan"), dtype=w.dtype, device=w.device)

        return {
            "modularity": float(modularity.detach().cpu().item()),
            "conductance_mean": float(conductance_mean.detach().cpu().item()),
            "conductance_weighted": float(conductance_weighted.detach().cpu().item()),
            "pred_n_clusters": float(k),
            "pred_cluster_size_cv": float(cluster_size_cv.detach().cpu().item()),
        }

    def _si_loss_no_grad(self, model, data) -> float:
        model.eval()
        with torch.no_grad():
            loss = model.se_loss(data)
        return float(loss.detach().item())

    def train_clu(self, data, model, optimizer, logger):
        logger.info("--------------------------Training Start-------------------------")
        epochs = int(self.configs.epochs)
        eval_freq = max(1, int(getattr(self.configs, "eval_freq", 1)))
        if eval_freq > epochs:
            logger.warning(
                f"eval_freq ({eval_freq}) > epochs ({epochs}); final-epoch evaluation will still be executed."
            )
        train_log_interval = max(1, int(getattr(self.configs, "train_log_interval", 1)))
        patience = max(0, int(getattr(self.configs, "patience", 0)))
        best_loss = float("inf")
        best_epoch = 0
        best_gamma = self._edge_fusion_gamma_for_epoch(1)
        no_improve = 0
        checkpoint_path = f"./checkpoints/{self.configs.save_path}"

        for epoch in range(1, epochs + 1):
            model.train()
            curr_gamma = self._edge_fusion_gamma_for_epoch(epoch)
            if hasattr(model, "set_edge_fusion_gamma"):
                model.set_edge_fusion_gamma(curr_gamma)

            if self.amp_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model.se_loss(data)
            else:
                loss = model.se_loss(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().item())

            if loss_value < best_loss - 1e-10:
                best_loss = loss_value
                best_epoch = epoch
                best_gamma = curr_gamma
                no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                no_improve += 1

            adaptive_stats = model.get_edge_adaptive_stats() if hasattr(model, "get_edge_adaptive_stats") else {
                "graph_alpha_mean": 1.0,
                "edge_reliability_mean": 1.0,
                "edge_mix_beta_mean": 0.0,
            }
            if epoch == 1 or epoch == epochs or epoch % train_log_interval == 0:
                edge_factor = adaptive_stats.get("edge_factor_mean", 1.0)
                edge_reg = adaptive_stats.get("edge_reg", 0.0)
                edge_factor_msg = adaptive_stats.get("edge_factor_msg_mean", edge_factor)
                edge_factor_si = adaptive_stats.get("edge_factor_si_mean", edge_factor)
                msg_gate = adaptive_stats.get("msg_gate_factor_mean", 1.0)
                edge_aug_prior = adaptive_stats.get("edge_aug_bias_mean", 0.0)
                hier_ratio = adaptive_stats.get("hier_edge_nonzero_ratio", 0.0)
                logger.info(
                    f"[Stage2] Epoch {epoch}: loss={loss_value:.4f}, edge_fusion_gamma={curr_gamma:.4f}, "
                    f"graph_alpha={adaptive_stats['graph_alpha_mean']:.4f}, "
                    f"edge_rel={adaptive_stats['edge_reliability_mean']:.4f}, "
                    f"edge_mix={adaptive_stats.get('edge_mix_beta_mean', 0.0):.4f}, "
                    f"edge_factor={edge_factor:.4f}, msg/si={edge_factor_msg:.4f}/{edge_factor_si:.4f}, "
                    f"msg_gate={msg_gate:.4f}, edge_aug={edge_aug_prior:.4f}, "
                    f"edge_reg={edge_reg:.6f}, hier_nonzero={hier_ratio:.4f}"
                )

            if (epoch % eval_freq == 0) or (epoch == epochs):
                logger.info("-----------------------Evaluation Start---------------------")
                eval_stats = self._evaluate_clustering_detailed(model, data)
                if np.isnan(eval_stats["acc_mean"]):
                    logger.info(f"Epoch {epoch}: labels unavailable, skip ACC/NMI/ARI.")
                else:
                    logger.info(
                        f"Epoch {epoch}: ACC: {eval_stats['acc_mean'] * 100: .2f}, "
                        f"NMI: {eval_stats['nmi_mean'] * 100: .2f}, "
                        f"ARI: {eval_stats['ari_mean'] * 100: .2f}, "
                        f"trial-std(NMI/ARI): {eval_stats['nmi_std']:.4g}/{eval_stats['ari_std']:.4g}"
                    )
                logger.info("-------------------------------------------------------------------------")

            if patience > 0 and no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(patience={patience}, best_loss={best_loss:.6f}, best_epoch={best_epoch})."
                )
                break

        if best_epoch > 0:
            model.load_state_dict(torch.load(checkpoint_path, map_location=data.x.device))
        if hasattr(model, "set_edge_fusion_gamma"):
            model.set_edge_fusion_gamma(best_gamma)

        final_eval = self._evaluate_clustering_detailed(model, data)
        final_pred = final_eval["predicts"][0] if final_eval["predicts"] else np.array([])
        struct = self._compute_partition_structure_metrics(data, final_pred)
        si_loss = self._si_loss_no_grad(model, data)
        edge_stats = model.get_edge_adaptive_stats() if hasattr(model, "get_edge_adaptive_stats") else {
            "graph_alpha_mean": 1.0,
            "edge_reliability_mean": 1.0,
            "edge_mix_beta_mean": 0.0,
            "edge_factor_mean": 1.0,
            "edge_factor_std": 0.0,
            "edge_factor_msg_mean": 1.0,
            "edge_factor_msg_std": 0.0,
            "edge_factor_si_mean": 1.0,
            "edge_factor_si_std": 0.0,
            "msg_gate_factor_mean": 1.0,
            "msg_gate_factor_std": 0.0,
            "edge_aug_bias_mean": 0.0,
            "edge_aug_bias_std": 0.0,
            "hier_edge_levels_active_ratio": 0.0,
            "hier_edge_nonzero_ratio": 0.0,
            "hier_edge_mean_abs": 0.0,
            "edge_reg": 0.0,
        }
        branch_diag = self._edge_branch_diagnostics(edge_stats)

        if np.isnan(final_eval["acc_mean"]):
            logger.info(
                f"Final model (selected by min train loss @ epoch {best_epoch}, loss={best_loss:.6f}) "
                f"evaluated without labels. SI-loss={si_loss:.6f}"
            )
        else:
            logger.info(
                f"Final model (selected by min train loss @ epoch {best_epoch}, loss={best_loss:.6f}): "
                f"ACC: {final_eval['acc_mean'] * 100: .2f}, "
                f"NMI: {final_eval['nmi_mean'] * 100: .2f}, "
                f"ARI: {final_eval['ari_mean'] * 100: .2f}, "
                f"SI-loss: {si_loss:.6f}, "
                f"Modularity: {struct['modularity']:.6f}, "
                f"Conductance(mean/w): {struct['conductance_mean']:.6f}/{struct['conductance_weighted']:.6f}, "
                f"Stability(pair-NMI): {final_eval['stability_pair_nmi']:.6f}"
            )
        logger.info(
            "[Diag] "
            f"factor={branch_diag.get('diag_factor_live', float('nan'))}, "
            f"dual={branch_diag.get('diag_dual_live', float('nan'))}, "
            f"msg={branch_diag.get('diag_msg_live', float('nan'))}, "
            f"assign={branch_diag.get('diag_assign_live', float('nan'))}, "
            f"hier={branch_diag.get('diag_hier_live', float('nan'))}, "
            f"aug={branch_diag.get('diag_aug_live', float('nan'))}, "
            f"dead={branch_diag.get('diag_dead_branch_count', 0.0)}/"
            f"{branch_diag.get('diag_required_branch_count', 0.0)}"
        )

        return {
            "final_acc": float(final_eval["acc_mean"]),
            "final_nmi": float(final_eval["nmi_mean"]),
            "final_ari": float(final_eval["ari_mean"]),
            "final_nmi_trial_std": float(final_eval["nmi_std"]),
            "final_ari_trial_std": float(final_eval["ari_std"]),
            "stability_pair_nmi": float(final_eval["stability_pair_nmi"]),
            "si_loss": float(si_loss),
            "modularity": float(struct["modularity"]),
            "conductance_mean": float(struct["conductance_mean"]),
            "conductance_weighted": float(struct["conductance_weighted"]),
            "pred_n_clusters": float(struct["pred_n_clusters"]),
            "pred_cluster_size_cv": float(struct["pred_cluster_size_cv"]),
            "best_epoch": float(best_epoch),
            "best_train_loss": float(best_loss),
            "final_graph_alpha": float(edge_stats.get("graph_alpha_mean", 1.0)),
            "final_edge_reliability": float(edge_stats.get("edge_reliability_mean", 1.0)),
            "final_edge_mix_beta": float(edge_stats.get("edge_mix_beta_mean", 0.0)),
            "final_edge_factor_mean": float(edge_stats.get("edge_factor_mean", 1.0)),
            "final_edge_factor_std": float(edge_stats.get("edge_factor_std", 0.0)),
            "final_edge_factor_msg_mean": float(edge_stats.get("edge_factor_msg_mean", edge_stats.get("edge_factor_mean", 1.0))),
            "final_edge_factor_msg_std": float(edge_stats.get("edge_factor_msg_std", edge_stats.get("edge_factor_std", 0.0))),
            "final_edge_factor_si_mean": float(edge_stats.get("edge_factor_si_mean", edge_stats.get("edge_factor_mean", 1.0))),
            "final_edge_factor_si_std": float(edge_stats.get("edge_factor_si_std", edge_stats.get("edge_factor_std", 0.0))),
            "final_msg_gate_factor_mean": float(edge_stats.get("msg_gate_factor_mean", 1.0)),
            "final_msg_gate_factor_std": float(edge_stats.get("msg_gate_factor_std", 0.0)),
            "final_edge_aug_bias_mean": float(edge_stats.get("edge_aug_bias_mean", 0.0)),
            "final_edge_aug_bias_std": float(edge_stats.get("edge_aug_bias_std", 0.0)),
            "final_hier_edge_levels_active_ratio": float(edge_stats.get("hier_edge_levels_active_ratio", 0.0)),
            "final_hier_edge_nonzero_ratio": float(edge_stats.get("hier_edge_nonzero_ratio", 0.0)),
            "final_hier_edge_mean_abs": float(edge_stats.get("hier_edge_mean_abs", 0.0)),
            "final_edge_reg": float(edge_stats.get("edge_reg", 0.0)),
            "diag_factor_live": float(branch_diag.get("diag_factor_live", float("nan"))),
            "diag_dual_live": float(branch_diag.get("diag_dual_live", float("nan"))),
            "diag_msg_live": float(branch_diag.get("diag_msg_live", float("nan"))),
            "diag_assign_live": float(branch_diag.get("diag_assign_live", float("nan"))),
            "diag_hier_live": float(branch_diag.get("diag_hier_live", float("nan"))),
            "diag_aug_live": float(branch_diag.get("diag_aug_live", float("nan"))),
            "diag_dead_branch_count": float(branch_diag.get("diag_dead_branch_count", 0.0)),
            "diag_required_branch_count": float(branch_diag.get("diag_required_branch_count", 0.0)),
            "diag_all_required_live": float(branch_diag.get("diag_all_required_live", 1.0)),
        }


# ========================================================================================
# FILE: modules/model.py
# ========================================================================================

import torch
import torch.nn as nn
from modules.layers import LSENetLayer, LorentzGraphConvolution
from utils.model_utils import select_activation


class LSENet(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, max_nums,
                 temperature=0.2, dropout=0.5, nonlin_str='relu',
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1,
                 edge_attr_hierarchical=False, edge_attr_pool_topk=1,
                 edge_msg_conditioned=False, edge_msg_gate_scale=0.35,
                 edge_msg_matched_only=False, edge_msg_confidence_gate=False, edge_msg_confidence_temp=1.0,
                 edge_attr_pool_confidence=False, edge_attr_pool_conf_power=1.0):
        super(LSENet, self).__init__()
        assert max_nums is not None
        self.manifold = manifold
        self.max_nums = max_nums  # [N_{H-1}, ..., N_1]
        self.height = len(max_nums) + 1
        self.edge_attr_hierarchical = bool(edge_attr_hierarchical)
        self.edge_attr_pool_topk = max(1, int(edge_attr_pool_topk))
        self.edge_msg_conditioned = bool(edge_msg_conditioned)
        self.edge_msg_gate_scale = float(edge_msg_gate_scale)
        self.edge_msg_matched_only = bool(edge_msg_matched_only)
        self.edge_msg_confidence_gate = bool(edge_msg_confidence_gate)
        self.edge_msg_confidence_temp = float(max(1e-3, edge_msg_confidence_temp))
        self.edge_attr_pool_confidence = bool(edge_attr_pool_confidence)
        self.edge_attr_pool_conf_power = float(max(0.0, edge_attr_pool_conf_power))
        self.last_hier_edge_levels_active_ratio = 0.0
        self.last_hier_edge_nonzero_ratio = 0.0
        self.last_hier_edge_mean_abs = 0.0
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0

        # Project input to Lorentz space (d+1)
        self.input_proj = LorentzGraphConvolution(manifold, in_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str),
                                                  edge_conditioned=self.edge_msg_conditioned,
                                                  edge_attr_dim=edge_attr_dim,
                                                  edge_attr_hidden_dim=edge_attr_hidden_dim,
                                                  edge_gate_scale=self.edge_msg_gate_scale,
                                                  edge_matched_only=self.edge_msg_matched_only,
                                                  edge_confidence_gate=self.edge_msg_confidence_gate,
                                                  edge_confidence_temp=self.edge_msg_confidence_temp)
        self.input_proj2 = LorentzGraphConvolution(manifold, hid_dim + 1, hid_dim + 1,
                                                  True, dropout, False,
                                                  select_activation(nonlin_str),
                                                  edge_conditioned=self.edge_msg_conditioned,
                                                  edge_attr_dim=edge_attr_dim,
                                                  edge_attr_hidden_dim=edge_attr_hidden_dim,
                                                  edge_gate_scale=self.edge_msg_gate_scale,
                                                  edge_matched_only=self.edge_msg_matched_only,
                                                  edge_confidence_gate=self.edge_msg_confidence_gate,
                                                  edge_confidence_temp=self.edge_msg_confidence_temp)
        self.dropout = nn.Dropout(dropout)

        # Build layers bottom-up: layer[0] = leaf → level H-1; layer[-1] = level 1 → root
        self.layers = nn.ModuleList()
        curr_dim = hid_dim + 1  # +1 for time-like coordinate
        for i in range(self.height - 1):
            self.layers.append(LSENetLayer(
                manifold, curr_dim, hid_dim + 1, max_nums[i],
                dropout=dropout, temperature=temperature,
                nonlin=select_activation(nonlin_str),
                edge_variant=edge_variant,
                edge_fusion_gamma=edge_fusion_gamma,
                edge_confidence_quantile=edge_confidence_quantile,
                edge_adaptive_alpha=edge_adaptive_alpha,
                edge_adaptive_alpha_strength=edge_adaptive_alpha_strength,
                edge_adaptive_alpha_bias=edge_adaptive_alpha_bias,
                edge_reliability_temp=edge_reliability_temp,
                edge_attr_hidden_dim=edge_attr_hidden_dim,
                edge_attr_fusion_scale=edge_attr_fusion_scale,
                edge_attr_dim=edge_attr_dim,
            ))
            curr_dim = hid_dim + 1  # parent embedding dim

    def set_edge_fusion_gamma(self, gamma: float):
        for layer in self.layers:
            if hasattr(layer, "assigner") and hasattr(layer.assigner, "set_edge_fusion_gamma"):
                layer.assigner.set_edge_fusion_gamma(gamma)

    def get_edge_adaptive_stats(self):
        alphas = []
        rels = []
        mixes = []
        for layer in self.layers:
            assigner = getattr(layer, "assigner", None)
            if assigner is None:
                continue
            if hasattr(assigner, "last_graph_alpha"):
                alphas.append(float(assigner.last_graph_alpha))
            if hasattr(assigner, "last_reliability_mean"):
                rels.append(float(assigner.last_reliability_mean))
            if hasattr(assigner, "last_mix_beta"):
                mixes.append(float(assigner.last_mix_beta))
        msg_means = []
        msg_stds = []
        for proj in (self.input_proj, self.input_proj2):
            agg = getattr(proj, "agg", None)
            if agg is None:
                continue
            msg_means.append(float(getattr(agg, "last_msg_gate_factor_mean", 1.0)))
            msg_stds.append(float(getattr(agg, "last_msg_gate_factor_std", 0.0)))
        if msg_means:
            self.last_msg_gate_factor_mean = float(sum(msg_means) / len(msg_means))
            self.last_msg_gate_factor_std = float(sum(msg_stds) / len(msg_stds))
        else:
            self.last_msg_gate_factor_mean = 1.0
            self.last_msg_gate_factor_std = 0.0
        return {
            "graph_alpha_mean": float(sum(alphas) / len(alphas)) if len(alphas) > 0 else 1.0,
            "edge_reliability_mean": float(sum(rels) / len(rels)) if len(rels) > 0 else 1.0,
            "edge_mix_beta_mean": float(sum(mixes) / len(mixes)) if len(mixes) > 0 else 0.0,
            "msg_gate_factor_mean": float(self.last_msg_gate_factor_mean),
            "msg_gate_factor_std": float(self.last_msg_gate_factor_std),
            "hier_edge_levels_active_ratio": float(self.last_hier_edge_levels_active_ratio),
            "hier_edge_nonzero_ratio": float(self.last_hier_edge_nonzero_ratio),
            "hier_edge_mean_abs": float(self.last_hier_edge_mean_abs),
        }

    def embed_leaf(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        # Map raw features to Lorentz leaf embedding
        o = torch.zeros_like(x[:, :1])                # (N, 1)
        x = torch.cat([o, x], dim=1)           # (N, d+1)
        x = self.manifold.expmap0(x)                  # project to Lorentz
        msg_use_edge_attr = bool(use_edge_attr) and self.edge_msg_conditioned
        x = self.input_proj(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=msg_use_edge_attr)  # (N, d + 1)
        x = self.input_proj2(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=msg_use_edge_attr)
        return x

    @staticmethod
    def _to_sparse_coalesced(adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            return adj.coalesce()
        return adj.to_sparse().coalesce()

    @classmethod
    def _coarsen_edge_attr_hard(
        cls,
        adj_curr: torch.Tensor,
        edge_attr_curr: torch.Tensor,
        ass: torch.Tensor,
        adj_par: torch.Tensor,
    ) -> torch.Tensor | None:
        if edge_attr_curr is None or edge_attr_curr.numel() == 0:
            return None

        adj_curr_sp = cls._to_sparse_coalesced(adj_curr)
        idx = adj_curr_sp.indices()
        val = adj_curr_sp.values()
        if edge_attr_curr.shape[0] != idx.shape[1]:
            return None

        parent = ass.argmax(dim=1)
        num_parent = int(ass.shape[1])
        src_p = parent[idx[0]]
        dst_p = parent[idx[1]]
        key = src_p.long() * num_parent + dst_p.long()
        sorted_key, order = torch.sort(key)

        attr = edge_attr_curr.float()
        weighted_attr = (attr * val.unsqueeze(1).to(attr.dtype))[order]
        val_sorted = val[order].to(attr.dtype)
        uniq, inv = torch.unique(sorted_key, sorted=True, return_inverse=True)

        agg_attr = torch.zeros((uniq.numel(), attr.shape[1]), dtype=attr.dtype, device=attr.device)
        agg_attr.index_add_(0, inv, weighted_attr)
        agg_denom = torch.zeros((uniq.numel(),), dtype=attr.dtype, device=attr.device)
        agg_denom.index_add_(0, inv, val_sorted)
        agg_attr = agg_attr / agg_denom.clamp_min(1e-6).unsqueeze(1)

        adj_par_sp = cls._to_sparse_coalesced(adj_par)
        idx_par = adj_par_sp.indices()
        target_key = idx_par[0].long() * num_parent + idx_par[1].long()
        pos = torch.searchsorted(uniq, target_key)
        valid = pos < uniq.shape[0]
        matched = torch.zeros_like(valid, dtype=torch.bool)
        matched[valid] = uniq[pos[valid]] == target_key[valid]

        out_attr = torch.zeros((idx_par.shape[1], attr.shape[1]), dtype=attr.dtype, device=attr.device)
        if matched.any():
            out_attr[matched] = agg_attr[pos[matched]]
        return out_attr

    @classmethod
    def _align_parent_edge_attr(
        cls,
        num_parent: int,
        key: torch.Tensor,
        pair_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        adj_par: torch.Tensor,
    ) -> torch.Tensor | None:
        if key.numel() == 0 or pair_weight.numel() == 0:
            return None
        order = torch.argsort(key)
        key = key[order]
        pair_weight = pair_weight[order]
        weighted_attr = (edge_attr * pair_weight.unsqueeze(1))[order]
        uniq, inv = torch.unique(key, sorted=True, return_inverse=True)

        agg_attr = torch.zeros((uniq.numel(), edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
        agg_attr.index_add_(0, inv, weighted_attr)
        agg_denom = torch.zeros((uniq.numel(),), dtype=edge_attr.dtype, device=edge_attr.device)
        agg_denom.index_add_(0, inv, pair_weight)
        agg_attr = agg_attr / agg_denom.clamp_min(1e-6).unsqueeze(1)

        adj_par_sp = cls._to_sparse_coalesced(adj_par)
        idx_par = adj_par_sp.indices()
        target_key = idx_par[0].long() * num_parent + idx_par[1].long()
        pos = torch.searchsorted(uniq, target_key)
        valid = pos < uniq.shape[0]
        matched = torch.zeros_like(valid, dtype=torch.bool)
        matched[valid] = uniq[pos[valid]] == target_key[valid]

        out_attr = torch.zeros((idx_par.shape[1], edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
        if matched.any():
            out_attr[matched] = agg_attr[pos[matched]]
        return out_attr

    @classmethod
    def _coarsen_edge_attr_soft_topk(
        cls,
        adj_curr: torch.Tensor,
        edge_attr_curr: torch.Tensor,
        ass: torch.Tensor,
        adj_par: torch.Tensor,
        topk: int,
        use_confidence: bool = False,
        conf_power: float = 1.0,
    ) -> torch.Tensor | None:
        if edge_attr_curr is None or edge_attr_curr.numel() == 0:
            return None
        if topk <= 1:
            return cls._coarsen_edge_attr_hard(adj_curr, edge_attr_curr, ass, adj_par)

        adj_curr_sp = cls._to_sparse_coalesced(adj_curr)
        idx = adj_curr_sp.indices()
        val = adj_curr_sp.values().float()
        if edge_attr_curr.shape[0] != idx.shape[1]:
            return None

        attr = edge_attr_curr.float()
        num_parent = int(ass.shape[1])
        k = min(int(topk), num_parent)
        prob_topk, idx_topk = torch.topk(ass, k=k, dim=1)
        prob_topk = prob_topk / prob_topk.sum(dim=1, keepdim=True).clamp_min(1e-6)
        conf = None
        if bool(use_confidence):
            if k <= 1:
                conf = torch.ones((ass.shape[0],), dtype=prob_topk.dtype, device=prob_topk.device)
            else:
                entropy = -(prob_topk * torch.log(prob_topk.clamp_min(1e-8))).sum(dim=1)
                norm = torch.log(torch.tensor(float(k), dtype=prob_topk.dtype, device=prob_topk.device)).clamp_min(1e-6)
                conf = (1.0 - entropy / norm).clamp(0.0, 1.0)
                if float(conf_power) != 1.0:
                    conf = conf.pow(float(conf_power))

        src = idx[0].long()
        dst = idx[1].long()
        src_prob = prob_topk[src]
        dst_prob = prob_topk[dst]
        src_parent = idx_topk[src]
        dst_parent = idx_topk[dst]

        key_parts = []
        weight_parts = []
        attr_parts = []
        for a in range(k):
            for b in range(k):
                pw = (src_prob[:, a] * dst_prob[:, b] * val).to(attr.dtype)
                if conf is not None:
                    pw = pw * conf[src].to(attr.dtype) * conf[dst].to(attr.dtype)
                keep = pw > 1e-8
                if not torch.any(keep):
                    continue
                key_parts.append(src_parent[:, a][keep].long() * num_parent + dst_parent[:, b][keep].long())
                weight_parts.append(pw[keep])
                attr_parts.append(attr[keep])

        if len(key_parts) == 0:
            return cls._coarsen_edge_attr_hard(adj_curr, edge_attr_curr, ass, adj_par)

        key = torch.cat(key_parts, dim=0)
        pair_weight = torch.cat(weight_parts, dim=0)
        attr_cat = torch.cat(attr_parts, dim=0)
        return cls._align_parent_edge_attr(num_parent, key, pair_weight, attr_cat, adj_par)

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        """
        Args:
            x: raw node features (N, D)
            adj: sparse adjacency
        """
        z = self.embed_leaf(x, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        tree_coord_dict = {self.height: z}
        ass_dict = {}
        adj_dict = {self.height: adj}

        current_z = z
        current_adj = adj
        current_edge_attr = edge_attr if bool(use_edge_attr) and edge_attr is not None else None
        hier_level_active = []
        hier_nonzero = []
        hier_mean_abs = []

        for i, layer in enumerate(self.layers):
            layer_use_edge_attr = bool(use_edge_attr) and current_edge_attr is not None
            layer_edge_attr = current_edge_attr if layer_use_edge_attr else None
            z_par, adj_par, ass, z_curr = layer(
                current_z,
                current_adj,
                edge_attr=layer_edge_attr,
                use_edge_attr=layer_use_edge_attr,
            )

            level_curr = self.height - i
            level_par = self.height - i - 1
            tree_coord_dict[level_par] = z_par
            ass_dict[level_curr] = ass
            adj_dict[level_par] = adj_par

            if layer_use_edge_attr and self.edge_attr_hierarchical:
                next_edge_attr = self._coarsen_edge_attr_soft_topk(
                    current_adj,
                    current_edge_attr,
                    ass,
                    adj_par,
                    topk=self.edge_attr_pool_topk,
                    use_confidence=self.edge_attr_pool_confidence,
                    conf_power=self.edge_attr_pool_conf_power,
                )
                current_edge_attr = next_edge_attr
                if next_edge_attr is None or next_edge_attr.numel() == 0:
                    hier_level_active.append(0.0)
                    hier_nonzero.append(0.0)
                    hier_mean_abs.append(0.0)
                else:
                    hier_level_active.append(1.0)
                    row_strength = next_edge_attr.abs().sum(dim=1)
                    hier_nonzero.append(float((row_strength > 1e-8).float().mean().item()))
                    hier_mean_abs.append(float(next_edge_attr.abs().mean().item()))
            else:
                current_edge_attr = None
            current_z = z_par
            current_adj = adj_par

        self.last_hier_edge_levels_active_ratio = float(sum(hier_level_active) / len(hier_level_active)) if len(hier_level_active) > 0 else 0.0
        self.last_hier_edge_nonzero_ratio = float(sum(hier_nonzero) / len(hier_nonzero)) if len(hier_nonzero) > 0 else 0.0
        self.last_hier_edge_mean_abs = float(sum(hier_mean_abs) / len(hier_mean_abs)) if len(hier_mean_abs) > 0 else 0.0

        # Root (level 0) is Frechet mean of level 1
        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device)

        return tree_coord_dict, ass_dict, adj_dict


# ========================================================================================
# FILE: modules/layers.py
# ========================================================================================

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import math
from utils.model_utils import gumbel_softmax, graph_top_K, normalize_adj, givens_rot_mat


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None,
                 edge_conditioned=False, edge_attr_dim=1, edge_attr_hidden_dim=64, edge_gate_scale=0.35,
                 edge_matched_only=False, edge_confidence_gate=False, edge_confidence_temp=1.0):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(
            manifold,
            out_dim,
            dropout,
            use_att,
            edge_conditioned=edge_conditioned,
            edge_attr_dim=edge_attr_dim,
            edge_attr_hidden_dim=edge_attr_hidden_dim,
            edge_gate_scale=edge_gate_scale,
            edge_matched_only=edge_matched_only,
            edge_confidence_gate=edge_confidence_gate,
            edge_confidence_temp=edge_confidence_temp,
        )

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        h = self.linear(x)
        h = self.agg(h, adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_dim,
                 out_dim,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_dim
        self.out_features = out_dim
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_dim, dropout, use_att,
                 edge_conditioned=False, edge_attr_dim=1, edge_attr_hidden_dim=64, edge_gate_scale=0.35,
                 edge_matched_only=False, edge_confidence_gate=False, edge_confidence_temp=1.0):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        self.edge_conditioned = bool(edge_conditioned)
        self.edge_gate_scale = float(edge_gate_scale)
        self.edge_matched_only = bool(edge_matched_only)
        self.edge_confidence_gate = bool(edge_confidence_gate)
        self.edge_confidence_temp = float(max(1e-3, edge_confidence_temp))
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))
        if self.edge_conditioned:
            hidden = max(8, int(edge_attr_hidden_dim))
            self.edge_gate_mlp = nn.Sequential(
                nn.Linear(int(max(1, edge_attr_dim)), hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    @staticmethod
    def _normalize_sparse_by_degree(adj_k):
        coo = adj_k.coalesce()
        idx = coo.indices()
        val = coo.values()
        row = idx[0].long()
        col = idx[1].long()
        num_nodes = int(coo.size(0))
        deg = torch.zeros(num_nodes, dtype=val.dtype, device=val.device)
        deg.index_add_(0, row, val)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(idx, norm_val, size=coo.size(), device=coo.device).coalesce()

    def _apply_edge_gate_to_adj(self, adj, edge_attr, edge_mask, use_edge_attr: bool):
        self.last_msg_gate_factor_mean = 1.0
        self.last_msg_gate_factor_std = 0.0
        if (not self.edge_conditioned) or (not bool(use_edge_attr)) or edge_attr is None:
            return adj

        adj_sp = adj.coalesce() if adj.is_sparse else adj.to_sparse().coalesce()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        if edge_attr.shape[0] != adj_sp.indices().shape[1]:
            return adj

        gate_raw = self.edge_gate_mlp(edge_attr.to(dtype=adj_sp.values().dtype, device=adj_sp.values().device)).squeeze(-1)
        gate_raw = gate_raw - gate_raw.mean()
        gate_raw = gate_raw / gate_raw.std(unbiased=False).clamp_min(1e-6)
        if self.edge_confidence_gate:
            conf = torch.sigmoid((gate_raw.abs() - 1.0) / self.edge_confidence_temp)
            gate_raw = conf * gate_raw
        gate_raw = torch.tanh(gate_raw) * float(self.edge_gate_scale)
        factor = torch.exp(gate_raw)
        eff_mask = None
        if edge_mask is not None:
            eff_mask = edge_mask.to(dtype=torch.bool, device=factor.device)
        if self.edge_matched_only and eff_mask is not None:
            factor = torch.where(eff_mask, factor, torch.ones_like(factor))
        gated = torch.sparse_coo_tensor(
            adj_sp.indices(),
            adj_sp.values() * factor.to(dtype=adj_sp.values().dtype, device=adj_sp.values().device),
            size=adj_sp.size(),
            device=adj_sp.device,
        ).coalesce()
        gated = self._normalize_sparse_by_degree(gated)
        if eff_mask is not None and torch.any(eff_mask):
            stat_factor = factor[eff_mask]
        else:
            stat_factor = factor
        self.last_msg_gate_factor_mean = float(stat_factor.detach().mean().item())
        self.last_msg_gate_factor_std = float(stat_factor.detach().std(unbiased=False).item())
        return gated

    def forward(self, x, adj, edge_attr=None, edge_mask=None, use_edge_attr=False):
        adj_eff = self._apply_edge_gate_to_adj(adj, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr)
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj_eff.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            # CUDA sparse addmm currently does not support bfloat16.
            # Keep sparse aggregation in fp32 and cast back to preserve AMP flow.
            if adj_eff.is_sparse and torch.is_autocast_enabled():
                out_dtype = x.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    support_t = torch.matmul(adj_eff.float(), x.float())
                support_t = support_t.to(dtype=out_dtype)
            else:
                support_t = torch.matmul(adj_eff, x)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, temperature=0.2, edge_variant='V1',
                 edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.edge_variant = edge_variant
        self.edge_fusion_gamma = edge_fusion_gamma
        self.edge_confidence_quantile = float(max(0.0, min(1.0, edge_confidence_quantile)))
        self.edge_adaptive_alpha = bool(edge_adaptive_alpha)
        self.edge_adaptive_alpha_strength = float(edge_adaptive_alpha_strength)
        self.edge_adaptive_alpha_bias = float(edge_adaptive_alpha_bias)
        self.edge_reliability_temp = float(max(1e-3, edge_reliability_temp))
        self.edge_attr_fusion_scale = float(edge_attr_fusion_scale)
        self.last_graph_alpha = 1.0
        self.last_reliability_mean = 1.0
        self.last_mix_beta = 0.0
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(int(max(1, edge_attr_dim)), max(8, int(edge_attr_hidden_dim))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, int(edge_attr_hidden_dim)), 2),
        )
        self.edge_attr_lorentz_encoder = nn.Sequential(
            nn.Linear(int(max(1, edge_attr_dim)), max(8, int(edge_attr_hidden_dim))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, int(edge_attr_hidden_dim)), max(1, int(hid_dim) - 1)),
        )

    def set_edge_fusion_gamma(self, gamma: float):
        self.edge_fusion_gamma = float(gamma)

    def _graph_alpha(self, signal: torch.Tensor, fallback_dtype: torch.dtype, fallback_device: torch.device):
        if not self.edge_adaptive_alpha:
            return torch.tensor(1.0, dtype=fallback_dtype, device=fallback_device)
        with torch.no_grad():
            s_mean = signal.detach().mean()
            s_std = signal.detach().std(unbiased=False)
            raw = s_mean - s_std
            alpha = torch.sigmoid(
                self.edge_adaptive_alpha_strength * raw + self.edge_adaptive_alpha_bias
            ).clamp(0.05, 0.95)
        return alpha

    def _struct_reliability_and_log(self, edge_value: torch.Tensor):
        edge_log = torch.log(edge_value.clamp_min(1e-8))
        center = torch.median(edge_value.detach())
        spread = edge_value.detach().std(unbiased=False).clamp_min(1e-6) * self.edge_reliability_temp
        reliability = torch.sigmoid((edge_value - center) / spread)
        if self.edge_confidence_quantile > 0.0:
            qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
            threshold = torch.quantile(edge_value.detach(), qv)
            conf_mask = (edge_value >= threshold).to(edge_log.dtype)
            reliability = reliability * conf_mask
        return reliability, edge_log

    def forward(self, x, adj, edge_attr=None, use_edge_attr=False):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        adj_coo = adj.coalesce() if adj.is_sparse else adj.to_sparse().coalesce()
        edge_index = adj_coo.indices()
        edge_value = adj_coo.values()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = -score
        if self.edge_variant == 'V5':
            reliability, edge_log = self._struct_reliability_and_log(edge_value)
            if self.edge_adaptive_alpha:
                with torch.no_grad():
                    mean_w = edge_value.detach().mean()
                    cv_w = edge_value.detach().std(unbiased=False) / mean_w.abs().clamp_min(1e-6)
                    raw = mean_w - cv_w
                    graph_alpha = torch.sigmoid(
                        self.edge_adaptive_alpha_strength * raw + self.edge_adaptive_alpha_bias
                    ).clamp(0.05, 0.95)
            else:
                graph_alpha = edge_log.new_tensor(1.0)
            self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
            self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
            self.last_mix_beta = 0.0
            score = score + float(self.edge_fusion_gamma) * graph_alpha * reliability * edge_log
        elif self.edge_variant in {'V6', 'V7', 'V8', 'V12', 'V13', 'V31', 'V32', 'V33'} and bool(use_edge_attr) and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            if edge_attr.shape[0] == edge_value.shape[0]:
                edge_attr = torch.nan_to_num(
                    edge_attr.to(dtype=score.dtype, device=score.device),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                attr_out = self.edge_attr_encoder(edge_attr)
                attr_bias = attr_out[:, 0]
                attr_gate = torch.sigmoid(attr_out[:, 1])
                attr_bias = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                edge_log = torch.log(edge_value.clamp_min(1e-8))
                if self.edge_variant == 'V6':
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    attr_term = reliability * attr_bias
                    self.last_mix_beta = 1.0
                elif self.edge_variant == 'V7':
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    align = torch.tanh(attr_bias * torch.tanh(edge_log))
                    attr_term = 0.7 * reliability * attr_bias + 0.3 * align
                    self.last_mix_beta = 1.0
                elif self.edge_variant == 'V8':
                    # V8: calibrate attribute fusion by agreement with structural prior.
                    struct_rel, _ = self._struct_reliability_and_log(edge_value)
                    attr_rel = attr_gate
                    reliability = 0.5 * (struct_rel + attr_rel)
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)

                    edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
                    attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                    with torch.no_grad():
                        agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
                        mix_beta = torch.sigmoid(
                            self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias
                        ).clamp(0.05, 0.95)
                    self.last_mix_beta = float(mix_beta.detach().cpu().item())

                    struct_term = struct_rel * torch.tanh(edge_log)
                    attr_term_raw = attr_rel * torch.tanh(attr_bias)
                    attr_term = (1.0 - mix_beta) * struct_term + mix_beta * attr_term_raw
                elif self.edge_variant in {'V12', 'V31', 'V32', 'V33'}:
                    # V12: keep V5 as stable trunk and add calibrated edge-attribute residual.
                    struct_rel, _ = self._struct_reliability_and_log(edge_value)
                    attr_rel = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(struct_rel.detach(), qv)
                        keep = (struct_rel >= threshold).to(struct_rel.dtype)
                        struct_rel = struct_rel * keep
                        attr_rel = attr_rel * keep
                    graph_alpha = self._graph_alpha(struct_rel, fallback_dtype=score.dtype, fallback_device=score.device)
                    reliability = struct_rel

                    edge_z = (edge_log - edge_log.mean()) / edge_log.std(unbiased=False).clamp_min(1e-6)
                    attr_z = (attr_bias - attr_bias.mean()) / attr_bias.std(unbiased=False).clamp_min(1e-6)
                    with torch.no_grad():
                        agreement = (edge_z.detach() * attr_z.detach()).mean().clamp(-1.0, 1.0)
                        mix_beta = torch.sigmoid(
                            self.edge_adaptive_alpha_strength * agreement + self.edge_adaptive_alpha_bias
                        ).clamp(0.05, 0.95)
                    self.last_mix_beta = float(mix_beta.detach().cpu().item())

                    residual = attr_rel * attr_bias
                    residual = residual / residual.std(unbiased=False).clamp_min(1e-6)
                    attr_term = struct_rel * edge_log + mix_beta * residual
                else:
                    # V13: map edge attributes into Lorentz space and fuse as geometric residual.
                    reliability = attr_gate
                    if self.edge_confidence_quantile > 0.0:
                        qv = float(min(0.999, max(0.0, self.edge_confidence_quantile)))
                        threshold = torch.quantile(reliability.detach(), qv)
                        reliability = reliability * (reliability >= threshold).to(reliability.dtype)
                    graph_alpha = self._graph_alpha(reliability, fallback_dtype=score.dtype, fallback_device=score.device)
                    self.last_mix_beta = 1.0

                    # Build edge points in Lorentz model from edge_attr tangent vectors.
                    edge_spatial = self.edge_attr_lorentz_encoder(edge_attr)
                    edge_tangent = torch.cat(
                        [torch.zeros((edge_spatial.shape[0], 1), dtype=edge_spatial.dtype, device=edge_spatial.device), edge_spatial],
                        dim=1,
                    )
                    edge_lorentz = self.manifold.expmap0(edge_tangent)

                    # Pair node endpoints into a Lorentz midpoint surrogate.
                    node_pair = q[src] + k[dst]
                    node_denorm = (-self.manifold.inner(None, node_pair, keepdim=True)).abs().clamp_min(1e-8).sqrt()
                    node_pair = node_pair / node_denorm

                    edge_geo = 2 + 2 * self.manifold.inner(edge_lorentz, node_pair, keepdim=False)
                    edge_geo = (edge_geo - edge_geo.mean()) / edge_geo.std(unbiased=False).clamp_min(1e-6)
                    attr_term = reliability * (0.7 * torch.tanh(edge_geo) + 0.3 * torch.tanh(attr_bias))
                score = score + float(self.edge_fusion_gamma) * self.edge_attr_fusion_scale * graph_alpha * attr_term
                self.last_graph_alpha = float(graph_alpha.detach().cpu().item())
                self.last_reliability_mean = float(reliability.detach().mean().cpu().item())
            else:
                self.last_graph_alpha = 1.0
                self.last_reliability_mean = 1.0
                self.last_mix_beta = 0.0
        else:
            self.last_graph_alpha = 1.0
            self.last_reliability_mean = 1.0
            self.last_mix_beta = 0.0
        score = scatter_softmax(score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        if att.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                ass = torch.matmul(att.float(), ass.float())
            ass = ass.to(dtype=out_dtype)
        else:
            ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0, edge_attr_dim=1):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.embeder = LorentzGraphConvolution(manifold, in_dim, hid_dim,
        #                                        True, dropout, use_att, nonlin)
        self.assigner = LorentzAssignment(manifold, hid_dim,
                                          hid_dim, num_assign,
                                          dropout, bias, temperature,
                                          edge_variant=edge_variant,
                                          edge_fusion_gamma=edge_fusion_gamma,
                                          edge_confidence_quantile=edge_confidence_quantile,
                                          edge_adaptive_alpha=edge_adaptive_alpha,
                                          edge_adaptive_alpha_strength=edge_adaptive_alpha_strength,
                                          edge_adaptive_alpha_bias=edge_adaptive_alpha_bias,
                                          edge_reliability_temp=edge_reliability_temp,
                                          edge_attr_hidden_dim=edge_attr_hidden_dim,
                                          edge_attr_fusion_scale=edge_attr_fusion_scale,
                                          edge_attr_dim=edge_attr_dim)

    def forward(self, x, adj, edge_attr=None, use_edge_attr=False):
        # x = self.embeder(x, adj)
        ass = self.assigner(x, adj, edge_attr=edge_attr, use_edge_attr=use_edge_attr)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_par = support_t / denorm

        if adj.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                adj_par = ass.float().t() @ adj.float() @ ass.float()
            adj_par = adj_par.to(dtype=out_dtype)
        else:
            adj_par = ass.t() @ adj @ ass
        idx = adj_par.nonzero().t()
        adj_par = torch.sparse_coo_tensor(idx, adj_par[idx[0], idx[1]], size=adj_par.shape)
        return x_par, adj_par, ass, x


class LorentzBoost(nn.Module):
    """
    Implements a learnable Lorentz boost transformation without eigendecomposition.
    Input: x in Lorentz model L^{d} (shape: [..., d+1])
    Output: L(x) in L^{d} (same shape)
    """
    def __init__(self, in_dim):  # in_dim = d+1
        super().__init__()
        self.in_dim = in_dim
        # Parameterize boost velocity beta in R^d (spatial part)
        self.beta = nn.Parameter(torch.randn(in_dim - 1) * 0.01)  # small init

    def forward(self, x):
        """
        x: [..., d+1], assumed to be in Lorentz model (x0 > 0, <x,x>_L = -1)
        Returns L(x): [..., d+1] in Lorentz model
        """
        d = self.in_dim - 1
        # Re-parameterize to keep ||beta|| < 1 and avoid invalid Lorentz factor.
        beta = torch.tanh(self.beta)
        beta_norm = torch.norm(beta, p=2)
        max_norm = 1.0 - 1e-4
        if beta_norm > max_norm:
            beta = beta * (max_norm / (beta_norm + 1e-12))
        beta_norm_sq = (beta ** 2).sum().clamp(max=1.0 - 1e-8)
        gamma = 1.0 / torch.sqrt(1.0 - beta_norm_sq + 1e-8)  # Lorentz factor

        # Construct boost matrix L (d+1, d+1)
        L = torch.eye(self.in_dim, device=x.device)
        L[0, 0] = gamma
        L[0, 1:] = -gamma * beta
        L[1:, 0] = -gamma * beta
        L[1:, 1:] += (gamma - 1) * torch.outer(beta, beta) / (beta_norm_sq + 1e-8)

        # Apply transformation
        Lx = torch.einsum('ij,...j->...i', L, x)
        return Lx


# ========================================================================================
# FILE: modules/dsi.py
# ========================================================================================

import torch
import torch.nn as nn
from utils.model_utils import gumbel_softmax, graph_top_K
from manifold.lorentz import Lorentz
from modules.layers import LorentzBoost
from modules.model import LSENet
from torch_scatter import scatter_softmax

MIN_NORM = 1e-15
EPS = 1e-6


class DSI(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2,
                 dropout=0.5, nonlin_str='relu', tau=1.0, alpha=0.01, knn=8,
                 edge_variant='V1', edge_fusion_gamma=1.0, edge_confidence_quantile=0.0,
                 edge_adaptive_alpha=False, edge_adaptive_alpha_strength=2.0,
                 edge_adaptive_alpha_bias=0.0, edge_reliability_temp=1.0,
                 edge_attr_hidden_dim=64, edge_attr_fusion_scale=1.0,
                 edge_attr_dim=1,
                 edge_attr_hierarchical=False,
                 edge_attr_pool_topk=1,
                 edge_msg_conditioned=False,
                 edge_msg_gate_scale=0.35,
                 edge_msg_matched_only=False,
                 edge_msg_confidence_gate=False,
                 edge_msg_confidence_temp=1.0,
                 edge_attr_pool_confidence=False,
                 edge_attr_pool_conf_power=1.0,
                 edge_weight_learn_reg_lambda=0.02,
                 edge_weight_learn_logclip=0.8,
                 edge_weight_learn_temp=1.0,
                 edge_weight_learn_apply_to='both',
                 edge_aug_prior_scale=0.0,
                 edge_aug_prior_mode='raw',
                 knn_mode='auto', knn_auto_threshold=20000):
        super(DSI, self).__init__()
        self.num_nodes = num_nodes
        self.height = len(max_nums) + 1
        self.manifold = Lorentz()
        self.encoder = LSENet(
            self.manifold,
            in_dim,
            hid_dim,
            max_nums,
            temperature,
            dropout,
            nonlin_str,
            edge_variant=edge_variant,
            edge_fusion_gamma=edge_fusion_gamma,
            edge_confidence_quantile=edge_confidence_quantile,
            edge_adaptive_alpha=edge_adaptive_alpha,
            edge_adaptive_alpha_strength=edge_adaptive_alpha_strength,
            edge_adaptive_alpha_bias=edge_adaptive_alpha_bias,
            edge_reliability_temp=edge_reliability_temp,
            edge_attr_hidden_dim=edge_attr_hidden_dim,
            edge_attr_fusion_scale=edge_attr_fusion_scale,
            edge_attr_dim=edge_attr_dim,
            edge_attr_hierarchical=bool(edge_attr_hierarchical),
            edge_attr_pool_topk=int(edge_attr_pool_topk),
            edge_msg_conditioned=bool(edge_msg_conditioned),
            edge_msg_gate_scale=float(edge_msg_gate_scale),
            edge_msg_matched_only=bool(edge_msg_matched_only),
            edge_msg_confidence_gate=bool(edge_msg_confidence_gate),
            edge_msg_confidence_temp=float(edge_msg_confidence_temp),
            edge_attr_pool_confidence=bool(edge_attr_pool_confidence),
            edge_attr_pool_conf_power=float(edge_attr_pool_conf_power),
        )
        self.lorentz_proj = LorentzBoost(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau
        self.alpha = alpha
        self.knn = knn
        self.edge_variant = str(edge_variant).upper()
        self.knn_mode = str(knn_mode).lower()
        self.knn_auto_threshold = int(knn_auto_threshold)
        self.edge_weight_learn_reg_lambda = float(max(0.0, edge_weight_learn_reg_lambda))
        self.edge_weight_learn_logclip = float(max(1e-3, edge_weight_learn_logclip))
        self.edge_weight_learn_temp = float(max(1e-3, edge_weight_learn_temp))
        self.edge_weight_learn_apply_to = str(edge_weight_learn_apply_to).lower()
        if self.edge_weight_learn_apply_to not in {'si_only', 'both'}:
            self.edge_weight_learn_apply_to = 'both'
        self.edge_aug_prior_scale = float(edge_aug_prior_scale)
        self.edge_aug_prior_mode = str(edge_aug_prior_mode).lower()
        if self.edge_aug_prior_mode not in {'raw', 'positive', 'tanh'}:
            self.edge_aug_prior_mode = 'raw'

        self.last_edge_factor_mean = 1.0
        self.last_edge_factor_std = 0.0
        self.last_edge_factor_msg_mean = 1.0
        self.last_edge_factor_msg_std = 0.0
        self.last_edge_factor_si_mean = 1.0
        self.last_edge_factor_si_std = 0.0
        self.last_edge_aug_bias_mean = 0.0
        self.last_edge_aug_bias_std = 0.0
        self.last_edge_reg = 0.0

        if self._use_learnable_edge_weight_variant():
            hidden = max(8, int(edge_attr_hidden_dim))
            out_dim = 1
            if self._use_dual_edge_weight_variant():
                out_dim = 3 if self._use_edge_aug_prior_variant() else 2
            self.edge_weight_mapper = nn.Sequential(
                nn.Linear(int(max(1, edge_attr_dim)), hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.edge_weight_mapper = None

    def set_edge_fusion_gamma(self, gamma: float):
        if hasattr(self.encoder, "set_edge_fusion_gamma"):
            self.encoder.set_edge_fusion_gamma(gamma)

    def get_edge_adaptive_stats(self):
        stats = {"graph_alpha_mean": 1.0, "edge_reliability_mean": 1.0, "edge_mix_beta_mean": 0.0}
        if hasattr(self.encoder, "get_edge_adaptive_stats"):
            stats = self.encoder.get_edge_adaptive_stats()
        stats["edge_factor_mean"] = float(self.last_edge_factor_mean)
        stats["edge_factor_std"] = float(self.last_edge_factor_std)
        stats["edge_factor_msg_mean"] = float(self.last_edge_factor_msg_mean)
        stats["edge_factor_msg_std"] = float(self.last_edge_factor_msg_std)
        stats["edge_factor_si_mean"] = float(self.last_edge_factor_si_mean)
        stats["edge_factor_si_std"] = float(self.last_edge_factor_si_std)
        stats["edge_aug_bias_mean"] = float(self.last_edge_aug_bias_mean)
        stats["edge_aug_bias_std"] = float(self.last_edge_aug_bias_std)
        stats["edge_reg"] = float(self.last_edge_reg)
        return stats

    def forward(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        if self._use_learnable_edge_weight_variant():
            head = 'msg' if self._use_dual_edge_weight_variant() else 'shared'
            adj, _ = self._apply_learned_edge_weight_to_adj(
                base_adj=getattr(data, "adj_msg", data.adj),
                target_adj=adj,
                base_edge_attr=getattr(data, "edge_attr", None),
                normalize_for_message=True,
                head=head,
            )
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        tree_coord_dict, ass_dict, adj_dict = self.encoder(features, adj, edge_attr=edge_attr, edge_mask=None, use_edge_attr=use_edge_attr)
        return tree_coord_dict, ass_dict, adj_dict

    def get_cluster_results(self, data):
        features = data.x
        adj = getattr(data, "adj_msg", data.adj).clone()
        if self._use_learnable_edge_weight_variant():
            head = 'msg' if self._use_dual_edge_weight_variant() else 'shared'
            adj, _ = self._apply_learned_edge_weight_to_adj(
                base_adj=getattr(data, "adj_msg", data.adj),
                target_adj=adj,
                base_edge_attr=getattr(data, "edge_attr", None),
                normalize_for_message=True,
                head=head,
            )
        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = getattr(data, "edge_attr", None) if use_edge_attr else None
        coord_dict, ass_dict, _ = self.encoder(features, adj, edge_attr=edge_attr, edge_mask=None, use_edge_attr=use_edge_attr)
        embed_dict = {}
        for height, x in coord_dict.items():
            embed_dict[height] = x.detach()
        # Avoid constructing an [N, N] identity for large graphs.
        clu_mat_dict = {}
        running = None
        for k in range(self.height - 1, 0, -1):
            ass = ass_dict[k + 1]
            running = ass if running is None else (running @ ass)
            idx = running.max(1)[1]
            t = torch.zeros_like(running)
            t[torch.arange(t.shape[0], device=t.device), idx] = 1.
            clu_mat_dict[k] = t
        return embed_dict, clu_mat_dict

    def fix_cluster_results(self, clu_res_mat, embed_dict, epsInt: int = 7):
        clu_nums = clu_res_mat.sum(0)
        clu_res = clu_res_mat.argmax(1)
        corr_idx = clu_nums > epsInt
        if torch.all(corr_idx):
            return clu_res
        idx = torch.arange(clu_res_mat.shape[1]).to(clu_res.device)
        idx = idx[corr_idx]
        err_idx = torch.where(clu_res_mat[:, clu_nums <= epsInt] == 1.)[0]
        node = embed_dict[self.height]
        parent = embed_dict[1]
        error_node = node[err_idx]
        fixed_parent = parent[corr_idx]
        score = torch.log_softmax(2 + 2 * self.manifold.cinner(error_node, fixed_parent), dim=-1)
        fixed_res = gumbel_softmax(score, self.temperature)
        fixed_res = idx[fixed_res.argmax(1)]
        clu_res[err_idx] = fixed_res
        return clu_res

    def se_loss(self, data, eps=1e-6):
        self.last_edge_reg = 0.0
        adj_base_msg = getattr(data, "adj_msg", data.adj).clone()
        adj_base_si = getattr(data, "adj_si", adj_base_msg).clone()
        base_edge_attr = getattr(data, "edge_attr", None)
        aug_prior_bias = None
        if self._use_edge_aug_prior_variant():
            aug_prior_bias = self._edge_prior_bias_for_base_adj(adj_base_msg, base_edge_attr)

        if self._use_edge_knn_mode(data.x.shape[0]):
            # For large-graph edge mode, stop gradient through adjacency construction.
            with torch.no_grad():
                z_leaf = self.encoder.embed_leaf(
                    data.x,
                    adj_base_msg,
                    edge_attr=base_edge_attr if self._use_edge_attr_variant() else None,
                    edge_mask=None,
                    use_edge_attr=self._use_edge_attr_variant(),
                )
                z_leaf = self.lorentz_proj(z_leaf)
                adj_aug = self._edge_candidate_adj(z_leaf, adj_base_msg, self.knn, edge_bias=aug_prior_bias)
        else:
            z_leaf = self.encoder.embed_leaf(
                data.x,
                adj_base_msg,
                edge_attr=base_edge_attr if self._use_edge_attr_variant() else None,
                edge_mask=None,
                use_edge_attr=self._use_edge_attr_variant(),
            )
            z_leaf = self.lorentz_proj(z_leaf)
            neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
            score_dense = neg_dist2 / self.tau
            if aug_prior_bias is not None and float(self.edge_aug_prior_scale) > 0.0:
                prior_dense = torch.sparse_coo_tensor(
                    adj_base_msg.coalesce().indices(),
                    aug_prior_bias.to(dtype=score_dense.dtype, device=score_dense.device),
                    size=adj_base_msg.size(),
                    device=score_dense.device,
                ).to_dense()
                score_dense = score_dense + float(self.edge_aug_prior_scale) * prior_dense
            adj_aug = graph_top_K(torch.softmax(score_dense, dim=-1), k=self.knn)

        adj_train_msg = (self.alpha * adj_aug + adj_base_msg).coalesce()
        adj_train_si = (self.alpha * adj_aug + adj_base_si).coalesce()

        edge_reg_raw = adj_train_msg.values().new_tensor(0.0)
        if self._use_learnable_edge_weight_variant():
            if self._use_dual_edge_weight_variant():
                adj_train_si, reg_si = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_si,
                    target_adj=adj_train_si,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=False,
                    head='si',
                )
                adj_train_msg, reg_msg = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_msg,
                    target_adj=adj_train_msg,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=True,
                    head='msg',
                )
                edge_reg_raw = 0.5 * (reg_si + reg_msg)
            else:
                adj_train_si, reg_si = self._apply_learned_edge_weight_to_adj(
                    base_adj=adj_base_si,
                    target_adj=adj_train_si,
                    base_edge_attr=base_edge_attr,
                    normalize_for_message=False,
                    head='shared',
                )
                edge_reg_raw = reg_si
                if self.edge_weight_learn_apply_to == 'both':
                    adj_train_msg, reg_msg = self._apply_learned_edge_weight_to_adj(
                        base_adj=adj_base_msg,
                        target_adj=adj_train_msg,
                        base_edge_attr=base_edge_attr,
                        normalize_for_message=True,
                        head='shared',
                    )
                    edge_reg_raw = 0.5 * (reg_si + reg_msg)

        use_edge_attr = self._use_edge_attr_variant()
        edge_attr = None
        edge_mask = None
        if use_edge_attr:
            edge_attr_base = getattr(data, "edge_attr", None)
            edge_attr, edge_mask = self._align_edge_attr_to_adj_with_mask(adj_base_msg, edge_attr_base, adj_train_msg)
        _, ass_aug_dict, _ = self.encoder(
            data.x, adj_train_msg, edge_attr=edge_attr, edge_mask=edge_mask, use_edge_attr=use_edge_attr
        )
        adj_si_dict = self._build_hierarchy_adj_from_assign(adj_train_si, ass_aug_dict)
        loss = self._si_loss(ass_aug_dict, adj_si_dict, eps)
        if self._use_learnable_edge_weight_variant() and self.edge_weight_learn_reg_lambda > 0.0:
            edge_reg = self.edge_weight_learn_reg_lambda * edge_reg_raw
            self.last_edge_reg = float(edge_reg.detach().item())
            loss = loss + edge_reg
        return loss

    def _use_edge_attr_variant(self) -> bool:
        return self.edge_variant in {'V6', 'V7', 'V8', 'V12', 'V13', 'V31', 'V32', 'V33'}

    def _use_learnable_edge_weight_variant(self) -> bool:
        return self.edge_variant in {'V20', 'V30', 'V31', 'V32', 'V33'}

    def _use_dual_edge_weight_variant(self) -> bool:
        return self.edge_variant in {'V30', 'V31', 'V32', 'V33'}

    def _use_edge_aug_prior_variant(self) -> bool:
        return self.edge_variant in {'V33'}

    @staticmethod
    def _ass_adj_ass(ass: torch.Tensor, adj: torch.Tensor):
        if adj.is_sparse and torch.is_autocast_enabled():
            out_dtype = ass.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = ass.float().t() @ adj.float() @ ass.float()
            return out.to(dtype=out_dtype)
        return ass.t() @ adj @ ass

    def _build_hierarchy_adj_from_assign(self, adj_leaf: torch.Tensor, ass_dict: dict) -> dict:
        adj_dict = {self.height: adj_leaf}
        curr_adj = adj_leaf
        for k in range(self.height, 1, -1):
            ass = ass_dict[k]
            next_adj = self._ass_adj_ass(ass, curr_adj)
            if isinstance(next_adj, torch.Tensor) and next_adj.is_sparse:
                next_adj = next_adj.coalesce()
            adj_dict[k - 1] = next_adj
            curr_adj = next_adj
        return adj_dict

    def _use_edge_knn_mode(self, num_nodes: int) -> bool:
        mode = self.knn_mode
        if mode == 'edge':
            return True
        if mode == 'dense':
            return False
        return int(num_nodes) > int(self.knn_auto_threshold)

    def _edge_candidate_adj(self, z_leaf, adj, k: int, edge_bias: torch.Tensor | None = None):
        adj_coo = adj.coalesce()
        edge_index = adj_coo.indices()
        num_nodes = z_leaf.shape[0]

        # Ensure every node has at least one candidate edge.
        self_idx = torch.arange(num_nodes, device=z_leaf.device)
        edge_index = torch.cat([edge_index, torch.stack([self_idx, self_idx], dim=0)], dim=1)
        src, dst = edge_index[0], edge_index[1]

        # Edge-wise score; do not materialize an [E, E] matrix.
        score = 2 + 2 * self.manifold.inner(z_leaf[src], z_leaf[dst], keepdim=False)
        score = score / float(self.tau)
        if edge_bias is not None and edge_bias.numel() > 0 and float(self.edge_aug_prior_scale) > 0.0:
            edge_bias = edge_bias.to(dtype=score.dtype, device=score.device)
            if edge_bias.shape[0] == adj_coo.values().shape[0]:
                edge_bias = torch.cat([edge_bias, torch.zeros(num_nodes, dtype=edge_bias.dtype, device=edge_bias.device)], dim=0)
            score = score + float(self.edge_aug_prior_scale) * edge_bias
        edge_index, score = self._topk_edges_per_src(edge_index, score, num_nodes=num_nodes, k=int(k))
        src = edge_index[0]
        prob = scatter_softmax(score, src, dim=0)
        adj_aug = torch.sparse_coo_tensor(edge_index, prob, size=(num_nodes, num_nodes), device=z_leaf.device).coalesce()
        return adj_aug

    @staticmethod
    def _topk_edges_per_src(edge_index, score, num_nodes: int, k: int):
        if k <= 0:
            return edge_index, score
        src = edge_index[0]
        dst = edge_index[1]
        # Sorting by source lets us run a linear pass over edge segments.
        order = torch.argsort(src)
        src_s = src[order]
        dst_s = dst[order]
        score_s = score[order]
        counts = torch.bincount(src_s, minlength=num_nodes).cpu().tolist()

        keep_parts = []
        st = 0
        for c in counts:
            ed = st + int(c)
            if ed <= st:
                st = ed
                continue
            seg_len = ed - st
            if seg_len <= k:
                keep_parts.append(torch.arange(st, ed, device=score.device))
            else:
                top_local = torch.topk(score_s[st:ed], k=k, sorted=False).indices + st
                keep_parts.append(top_local)
            st = ed

        if len(keep_parts) == 0:
            return edge_index[:, :0], score[:0]
        keep = torch.cat(keep_parts, dim=0)
        edge_new = torch.stack([src_s[keep], dst_s[keep]], dim=0)
        score_new = score_s[keep]
        return edge_new, score_new

    @staticmethod
    def _align_edge_attr_to_adj(base_adj, base_edge_attr, target_adj):
        target_attr, _ = DSI._align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj)
        return target_attr

    @staticmethod
    def _align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj):
        if base_edge_attr is None:
            return None, None
        if base_edge_attr.numel() == 0:
            return base_edge_attr, None
        base = base_adj.coalesce()
        target = target_adj.coalesce()
        base_idx = base.indices()
        target_idx = target.indices()
        num_nodes = int(base.size(0))
        base_attr = base_edge_attr.float()
        if base_attr.shape[0] != base_idx.shape[1]:
            return None, None

        base_key = base_idx[0].long() * num_nodes + base_idx[1].long()
        target_key = target_idx[0].long() * num_nodes + target_idx[1].long()

        sorted_key, order = torch.sort(base_key)
        pos = torch.searchsorted(sorted_key, target_key)
        valid = pos < sorted_key.shape[0]
        matched = torch.zeros_like(valid, dtype=torch.bool)
        matched[valid] = sorted_key[pos[valid]] == target_key[valid]

        target_attr = torch.zeros(
            (target_idx.shape[1], base_attr.shape[1]),
            dtype=base_attr.dtype,
            device=base_attr.device,
        )
        if matched.any():
            mapped = order[pos[matched]]
            target_attr[matched] = base_attr[mapped]
        return target_attr, matched

    def _edge_scores_and_factors_from_attr(self, edge_attr: torch.Tensor, matched: torch.Tensor | None):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        edge_attr = edge_attr.float()
        if self.edge_weight_mapper is None:
            ones = torch.ones((edge_attr.shape[0], 1), dtype=edge_attr.dtype, device=edge_attr.device)
            zeros = torch.zeros_like(ones)
            zero = torch.zeros((), dtype=edge_attr.dtype, device=edge_attr.device)
            return zeros, ones, zero

        score = self.edge_weight_mapper(edge_attr)
        score = score / float(self.edge_weight_learn_temp)
        score = torch.tanh(score) * float(self.edge_weight_learn_logclip)

        if matched is not None and matched.any():
            score = score - score[matched].mean(dim=0, keepdim=True)
        else:
            score = score - score.mean(dim=0, keepdim=True)

        score = score.clamp(-float(self.edge_weight_learn_logclip), float(self.edge_weight_learn_logclip))
        factor = torch.exp(score)
        if matched is not None:
            factor = torch.where(matched.unsqueeze(1), factor, torch.ones_like(factor))
            eff = matched
        else:
            eff = torch.ones(score.shape[0], dtype=torch.bool, device=score.device)

        if eff.any():
            reg_anchor = torch.mean(score[eff] ** 2)
            reg_scale = torch.mean((factor[eff] - 1.0) ** 2)
            reg = reg_anchor + 0.1 * reg_scale
            factor_eff = factor[eff]
            self.last_edge_factor_mean = float(factor_eff.detach().mean().item())
            self.last_edge_factor_std = float(factor_eff.detach().std(unbiased=False).item())
            self.last_edge_factor_msg_mean = float(factor_eff[:, 0].detach().mean().item())
            self.last_edge_factor_msg_std = float(factor_eff[:, 0].detach().std(unbiased=False).item())
            if factor_eff.shape[1] >= 2:
                self.last_edge_factor_si_mean = float(factor_eff[:, 1].detach().mean().item())
                self.last_edge_factor_si_std = float(factor_eff[:, 1].detach().std(unbiased=False).item())
            else:
                self.last_edge_factor_si_mean = self.last_edge_factor_msg_mean
                self.last_edge_factor_si_std = self.last_edge_factor_msg_std
            if score.shape[1] >= 3:
                self.last_edge_aug_bias_mean = float(score[eff, 2].detach().mean().item())
                self.last_edge_aug_bias_std = float(score[eff, 2].detach().std(unbiased=False).item())
            else:
                self.last_edge_aug_bias_mean = 0.0
                self.last_edge_aug_bias_std = 0.0
        else:
            reg = factor.new_tensor(0.0)
            self.last_edge_factor_mean = 1.0
            self.last_edge_factor_std = 0.0
            self.last_edge_factor_msg_mean = 1.0
            self.last_edge_factor_msg_std = 0.0
            self.last_edge_factor_si_mean = 1.0
            self.last_edge_factor_si_std = 0.0
            self.last_edge_aug_bias_mean = 0.0
            self.last_edge_aug_bias_std = 0.0
        return score, factor, reg

    @staticmethod
    def _edge_head_index(head: str) -> int:
        if head == 'si':
            return 1
        if head == 'aug':
            return 2
        return 0

    def _edge_prior_bias_for_base_adj(self, base_adj, base_edge_attr):
        if (not self._use_edge_aug_prior_variant()) or self.edge_weight_mapper is None:
            return None
        if base_edge_attr is None or base_edge_attr.numel() == 0:
            return None
        base = base_adj.coalesce()
        edge_attr = base_edge_attr.float().to(base.values().device)
        if edge_attr.shape[0] != base.indices().shape[1]:
            return None
        raw_score, _, _ = self._edge_scores_and_factors_from_attr(edge_attr, matched=None)
        if raw_score.shape[1] <= 2:
            return None
        bias = raw_score[:, 2]
        if self.edge_aug_prior_mode == 'positive':
            bias = torch.relu(bias)
        elif self.edge_aug_prior_mode == 'tanh':
            bias = torch.tanh(bias)
        return bias

    def _apply_learned_edge_weight_to_adj(self, base_adj, target_adj, base_edge_attr, normalize_for_message: bool, head: str = 'shared'):
        edge_attr, matched = self._align_edge_attr_to_adj_with_mask(base_adj, base_edge_attr, target_adj)
        if edge_attr is None or edge_attr.numel() == 0:
            self.last_edge_factor_mean = 1.0
            self.last_edge_factor_std = 0.0
            self.last_edge_factor_msg_mean = 1.0
            self.last_edge_factor_msg_std = 0.0
            self.last_edge_factor_si_mean = 1.0
            self.last_edge_factor_si_std = 0.0
            self.last_edge_aug_bias_mean = 0.0
            self.last_edge_aug_bias_std = 0.0
            return target_adj.coalesce(), target_adj.values().new_tensor(0.0)

        tgt = target_adj.coalesce()
        matched_dev = matched.to(tgt.values().device) if matched is not None else None
        _, factor_all, reg = self._edge_scores_and_factors_from_attr(edge_attr.to(tgt.values().device), matched_dev)
        head_idx = 0 if head == 'shared' else self._edge_head_index(head)
        head_idx = min(int(head_idx), int(factor_all.shape[1] - 1))
        factor = factor_all[:, head_idx]
        new_val = tgt.values() * factor.to(dtype=tgt.values().dtype, device=tgt.values().device)
        out = torch.sparse_coo_tensor(tgt.indices(), new_val, size=tgt.size(), device=tgt.device).coalesce()
        if normalize_for_message:
            out = self._normalize_sparse_by_degree(out)
        return out, reg

    @staticmethod
    def _sparse_degree_and_diag(adj_k):
        if adj_k.is_sparse:
            coo = adj_k.coalesce()
            idx = coo.indices()
            val = coo.values()
            degree = torch.sparse.sum(coo, dim=1).to_dense()
            diag = torch.zeros(coo.size(0), dtype=val.dtype, device=val.device)
            mask = idx[0] == idx[1]
            if mask.any():
                diag.index_add_(0, idx[0][mask], val[mask])
            return degree, diag
        degree = adj_k.sum(dim=1)
        diag = torch.diagonal(adj_k, dim1=0, dim2=1)
        return degree, diag

    @staticmethod
    def _normalize_sparse_by_degree(adj_k):
        coo = adj_k.coalesce()
        idx = coo.indices()
        val = coo.values()
        row = idx[0].long()
        col = idx[1].long()
        num_nodes = int(coo.size(0))
        deg = torch.zeros(num_nodes, dtype=val.dtype, device=val.device)
        deg.index_add_(0, row, val)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(idx, norm_val, size=coo.size(), device=coo.device).coalesce()

    def _si_loss(self, ass_dict: dict, adj_dict: dict, eps: float = 1e-6):
        se_loss = 0
        vol_G = adj_dict[self.height].sum()

        for k in range(self.height, 0, -1):
            degree, diag = self._sparse_degree_and_diag(adj_dict[k])
            if k == 1:
                vol_parent = vol_G
            else:
                if adj_dict[k - 1].is_sparse:
                    vol_parent = torch.sparse.sum(adj_dict[k - 1], dim=1).to_dense()
                else:
                    vol_parent = adj_dict[k - 1].sum(dim=-1)
                vol_parent = torch.einsum('ij, j->i', ass_dict[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio_k = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss


# ========================================================================================
# FILE: tools/run_preset.py
# ========================================================================================

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
