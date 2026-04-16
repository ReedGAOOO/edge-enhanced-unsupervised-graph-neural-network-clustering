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
