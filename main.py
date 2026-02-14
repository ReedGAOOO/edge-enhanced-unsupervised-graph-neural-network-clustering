import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
parser.add_argument("--epsInt", type=int, default=8)
parser.add_argument('--edge_variant', type=str, default='V1', choices=['V1', 'V2', 'V3', 'V4', 'V5'],
                    help='V1: plain adjacency; V2: structural pre-weight; '
                         'V3: feature-similarity pre-weight; V4: hybrid pre-weight; '
                         'V5: hybrid + attention-stage edge fusion.')
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
parser.add_argument('--train_log_interval', type=int, default=1,
                    help='Epoch interval for Stage2 training loss logs.')

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')
parser.add_argument('--seed', type=int, default=3047)


configs = parser.parse_args()
set_seed(configs.seed)
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
