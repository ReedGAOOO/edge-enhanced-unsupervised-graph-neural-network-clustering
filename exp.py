import torch
import numpy as np
from modules.dsi import DSI
from geoopt.optim import RiemannianAdam
from utils.eval_utils import cluster_metrics
from data import load_data
from logger import create_logger
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import AdamW
import math


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs).to(device)

        total_nmi = []
        total_ari = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            model = DSI(in_dim=data.x.shape[1],
                        hid_dim=self.configs.hid_dim,
                        num_nodes=data.x.shape[0],
                        temperature=self.configs.temperature,
                        dropout=self.configs.dropout,
                        nonlin_str=self.configs.nonlin,
                        max_nums=self.configs.max_nums,
                        alpha=self.configs.alpha,
                        knn=self.configs.knn,
                        edge_variant=getattr(self.configs, 'edge_variant', 'V1'),
                        edge_fusion_gamma=getattr(self.configs, 'edge_fusion_gamma', 1.0),
                        edge_confidence_quantile=getattr(self.configs, 'edge_confidence_quantile', 0.0),
                        edge_adaptive_alpha=bool(getattr(self.configs, 'edge_adaptive_alpha', False)),
                        edge_adaptive_alpha_strength=float(getattr(self.configs, 'edge_adaptive_alpha_strength', 2.0)),
                        edge_adaptive_alpha_bias=float(getattr(self.configs, 'edge_adaptive_alpha_bias', 0.0)),
                        edge_reliability_temp=float(getattr(self.configs, 'edge_reliability_temp', 1.0))).to(device)
            optimizer = AdamW(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'Clustering':
                nmi, ari = self.train_clu(data, model, optimizer, logger)
                total_nmi.append(nmi)
                total_ari.append(ari)

        if self.configs.task == 'Clustering':
            logger.info(f"NMI: {np.mean(total_nmi)}+-{np.std(total_nmi)}, "
                        f"ARI: {np.mean(total_ari)}+-{np.std(total_ari)}")
            return {
                'dataset': self.configs.dataset,
                'edge_variant': getattr(self.configs, 'edge_variant', 'V1'),
                'edge_fusion_gamma': float(getattr(self.configs, 'edge_fusion_gamma', 1.0)),
                'edge_fusion_gamma_start': getattr(self.configs, 'edge_fusion_gamma_start', None),
                'edge_fusion_gamma_end': getattr(self.configs, 'edge_fusion_gamma_end', None),
                'edge_fusion_gamma_sched_epochs': int(getattr(self.configs, 'edge_fusion_gamma_sched_epochs', 0)),
                'edge_confidence_quantile': float(getattr(self.configs, 'edge_confidence_quantile', 0.0)),
                'edge_adaptive_alpha': bool(getattr(self.configs, 'edge_adaptive_alpha', False)),
                'edge_adaptive_alpha_strength': float(getattr(self.configs, 'edge_adaptive_alpha_strength', 2.0)),
                'edge_adaptive_alpha_bias': float(getattr(self.configs, 'edge_adaptive_alpha_bias', 0.0)),
                'edge_reliability_temp': float(getattr(self.configs, 'edge_reliability_temp', 1.0)),
                'nmi_mean': float(np.mean(total_nmi)),
                'nmi_std': float(np.std(total_nmi)),
                'ari_mean': float(np.mean(total_ari)),
                'ari_std': float(np.std(total_ari)),
                'exp_iters': int(self.configs.exp_iters),
                'epochs': int(self.configs.epochs),
                'eval_freq': int(self.configs.eval_freq),
                'seed': int(self.configs.seed),
            }
        return {}

    def _edge_fusion_gamma_for_epoch(self, epoch: int) -> float:
        base = float(getattr(self.configs, 'edge_fusion_gamma', 1.0))
        start = getattr(self.configs, 'edge_fusion_gamma_start', None)
        end = getattr(self.configs, 'edge_fusion_gamma_end', None)
        if start is None and end is None:
            return base
        start_v = float(base if start is None else start)
        end_v = float(base if end is None else end)
        sched_epochs = int(getattr(self.configs, 'edge_fusion_gamma_sched_epochs', 0))
        if sched_epochs <= 0:
            return end_v
        if sched_epochs == 1:
            return end_v
        ratio = min(1.0, max(0.0, float(epoch - 1) / float(sched_epochs - 1)))
        return start_v + ratio * (end_v - start_v)

    def train_clu(self, data, model, optimizer, logger):
        best_cluster_result = {}
        best_cluster = {'nmi': 0, 'ari': 0}

        logger.info("--------------------------Training Start-------------------------")
        n_cluster_trials = self.configs.n_cluster_trials
        epoch_acc = []
        epoch_nmi = []
        epoch_ari = []

        for epoch in range(1, self.configs.epochs + 1):
            model.train()
            curr_gamma = self._edge_fusion_gamma_for_epoch(epoch)
            if hasattr(model, "set_edge_fusion_gamma"):
                model.set_edge_fusion_gamma(curr_gamma)

            loss = model.se_loss(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adaptive_stats = model.get_edge_adaptive_stats() if hasattr(model, "get_edge_adaptive_stats") else {
                "graph_alpha_mean": 1.0,
                "edge_reliability_mean": 1.0,
            }
            train_log_interval = max(1, int(getattr(self.configs, "train_log_interval", 1)))
            if epoch == 1 or epoch == self.configs.epochs or epoch % train_log_interval == 0:
                logger.info(
                    f"[Stage2] Epoch {epoch}: loss={loss.item():.4f}, edge_fusion_gamma={curr_gamma:.4f}, "
                    f"graph_alpha={adaptive_stats['graph_alpha_mean']:.4f}, "
                    f"edge_rel={adaptive_stats['edge_reliability_mean']:.4f}"
                )

            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                embed_dict, clu_mat_dict = model.get_cluster_results(data)
                predicts = model.fix_cluster_results(clu_mat_dict[1], embed_dict, self.configs.epsInt).cpu().numpy()
                trues = data.y.cpu().numpy()
                acc, nmi, ari = [], [], []
                for step in range(n_cluster_trials):
                    metrics = cluster_metrics(trues, predicts)
                    acc_, nmi_, ari_ = metrics.evaluateFromLabel(use_acc=True)
                    acc.append(acc_)
                    nmi.append(nmi_)
                    ari.append(ari_)
                acc, nmi, ari = np.mean(acc), np.mean(nmi), np.mean(ari)

                epoch_acc.append(acc)
                epoch_nmi.append(nmi)
                epoch_ari.append(ari)

                if nmi > best_cluster['nmi']:
                    best_cluster['nmi'] = nmi
                    best_cluster['ari'] = ari
                    best_cluster_result['nmi'] = [nmi, ari]
                    logger.info('------------------Saving best model-------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.save_path}")
                logger.info(
                    f"Epoch {epoch}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f}")
                logger.info(
                    "-------------------------------------------------------------------------")

        for k, result in best_cluster_result.items():
            nmi, ari = result
            logger.info(
                f"Best Results according to {k}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f} \n")
        return best_cluster['nmi'], best_cluster["ari"]
