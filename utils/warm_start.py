"""Pseudo-partition warm start for the DSE backbone.

On graphs whose node features do not already separate the clusters, the Lorentz soft-assignment
parameterisation stays on a balanced, graph-agnostic plateau: its training loss ends above the
structural entropy of a random hard partition, even though spectral clustering finds a partition
with far lower SE (results/breakthrough_v1/e2d_plateau_diag). Pre-training the assignment head on a
cheap partition gives the SE objective a non-degenerate starting point.

Candidates are spectral clustering on A and KMeans on SGC-smoothed features S^2 X. In `auto` mode
the candidate with the lower two-level structural entropy is chosen, i.e. selection uses the same
objective DSE minimises and never looks at labels.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _sym_adj(edge_index: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def _norm_adj_with_self_loops(a: sp.csr_matrix) -> sp.csr_matrix:
    a = a + sp.eye(a.shape[0])
    deg = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg))
    return (dinv @ a @ dinv).tocsr()


def spectral_partition(a: sp.csr_matrix, k: int, seed: int) -> np.ndarray:
    _, vec = eigsh(_norm_adj_with_self_loops(a), k=k, which="LA")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True).clip(1e-12)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(vec)


def sgc_partition(a: sp.csr_matrix, x: np.ndarray, k: int, seed: int, hops: int = 2) -> np.ndarray:
    s = _norm_adj_with_self_loops(a)
    h = StandardScaler().fit_transform(x)
    for _ in range(hops):
        h = s @ h
    return KMeans(k, n_init=10, random_state=seed).fit_predict(h)


def structural_entropy_2level(a: sp.csr_matrix, labels: np.ndarray) -> float:
    deg = np.asarray(a.sum(1)).ravel()
    vol = deg.sum()
    h = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        v = deg[idx].sum()
        if v <= 0:
            continue
        cut = v - a[idx][:, idx].sum()
        h -= cut / vol * np.log2(v / vol)
        d = deg[idx][deg[idx] > 0]
        h -= np.sum(d / vol * np.log2(d / v))
    return float(h)


def build_pseudo_partition(edge_index: np.ndarray, x: np.ndarray, k: int, method: str = "auto",
                           seed: int = 0) -> Tuple[np.ndarray, Dict[str, float]]:
    a = _sym_adj(edge_index, x.shape[0])
    candidates = {}
    if method in ("auto", "spectral"):
        candidates["spectral"] = spectral_partition(a, k, seed)
    if method in ("auto", "sgc"):
        candidates["sgc"] = sgc_partition(a, x, k, seed)
    if not candidates:
        raise ValueError(f"Unknown warm start method: {method}")
    se = {name: structural_entropy_2level(a, lab) for name, lab in candidates.items()}
    chosen = min(se, key=se.get)
    info = {f"warm_se_{name}": val for name, val in se.items()}
    info["warm_chosen"] = chosen
    return candidates[chosen], info
