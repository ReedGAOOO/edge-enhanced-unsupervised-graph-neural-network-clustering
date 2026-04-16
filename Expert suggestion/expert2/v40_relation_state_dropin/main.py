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
parser.add_argument('--edge_variant', type=str, default='V1', choices=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V12', 'V13', 'V20', 'V30', 'V31', 'V32', 'V33', 'V40'],
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
                         'V33: V32 + edge-aware augment prior; '
                         'V40: latent relation channels + hierarchical persistence.')
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
parser.add_argument('--edge_relation_channels', type=int, default=4,
                    help='V40: number of latent relation channels for edge-state modeling.')
parser.add_argument('--edge_relation_hidden_dim', type=int, default=64,
                    help='V40: hidden size of the shared relation-state encoder.')
parser.add_argument('--edge_relation_assign_scale', type=float, default=1.0,
                    help='V40: scale of relation-aware compatibility term in assignment.')
parser.add_argument('--edge_relation_reg_lambda', type=float, default=1e-3,
                    help='V40: regularization strength for relation-state stability.')
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


