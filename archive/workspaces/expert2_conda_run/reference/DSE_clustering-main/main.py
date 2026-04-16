import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='KarateClub')
parser.add_argument('--task', type=str, default='Clustering',
                    choices=['Clustering'])
parser.add_argument('--root_path', type=str, default='datasets')
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
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adam', 'adamw', 'riemannianadam'])
parser.add_argument('--paper_faithful', action='store_true')
parser.add_argument('--leaf_use_att', action='store_true')
parser.add_argument('--leaf_att_mode', type=str, default='legacy',
                    choices=['legacy', 'paper'])
parser.add_argument('--num_input_lconvs', type=int, default=2)
parser.add_argument('--assign_att_mode', type=str, default='legacy',
                    choices=['legacy', 'paper'])
parser.add_argument('--assign_gumbel', dest='assign_gumbel', action='store_true')
parser.add_argument('--no_assign_gumbel', dest='assign_gumbel', action='store_false')
parser.set_defaults(assign_gumbel=True)
parser.add_argument('--paper_graph_fusion', action='store_true')

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')


configs = parser.parse_args()
if configs.paper_faithful:
    configs.optimizer = 'adam'
    configs.lr = 0.003
    configs.hid_dim = 2
    configs.max_nums = [10]
    configs.knn = 8
    configs.alpha = 0.01
    configs.leaf_use_att = True
    configs.leaf_att_mode = 'paper'
    configs.num_input_lconvs = 1
    configs.assign_att_mode = 'paper'
    configs.assign_gumbel = False
    configs.paper_graph_fusion = True

# with open(f'./configs/{configs.dataset}.json', 'wt') as f:
#     json.dump(vars(configs), f, indent=4)

# configs_dict = vars(configs)
# with open(f'./configs/{configs.dataset}.json', 'rt') as f:
#     configs_dict.update(json.load(f))
# configs = DotDict(configs_dict)
# f.close()

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs(f"./results/{configs.dataset}", exist_ok=True)
os.makedirs(f"./results/{configs.version}", exist_ok=True)
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()
