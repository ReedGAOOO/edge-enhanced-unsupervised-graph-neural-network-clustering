#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import DBLP


def csr_row_topk(csr: sp.csr_matrix, k: int, min_score: float = 0.0) -> sp.csr_matrix:
    k = int(max(1, k))
    n = csr.shape[0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for i in range(n):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        if s >= e:
            continue
        c = csr.indices[s:e]
        v = csr.data[s:e]
        if min_score > 0.0:
            m = v >= float(min_score)
            if not np.any(m):
                continue
            c = c[m]
            v = v[m]
        if c.size == 0:
            continue
        if c.size > k:
            idx = np.argpartition(v, -k)[-k:]
            c = c[idx]
            v = v[idx]
        order = np.argsort(-v)
        c = c[order]
        v = v[order]
        rows.append(np.full(c.size, i, dtype=np.int64))
        cols.append(c.astype(np.int64, copy=False))
        vals.append(v.astype(np.float32, copy=False))

    if not rows:
        return sp.csr_matrix(csr.shape, dtype=np.float32)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    out = sp.coo_matrix((v, (r, c)), shape=csr.shape, dtype=np.float32).tocsr()
    out.eliminate_zeros()
    return out


def cosine_sim_topk_from_csr(x_csr: sp.csr_matrix, k: int, min_score: float = 0.0) -> sp.csr_matrix:
    sim = (x_csr @ x_csr.T).tocsr()
    sim.setdiag(0.0)
    sim.eliminate_zeros()

    norms = np.sqrt(x_csr.multiply(x_csr).sum(axis=1)).A1.astype(np.float32, copy=False)
    norms = np.maximum(norms, 1e-12)

    coo = sim.tocoo()
    denom = norms[coo.row] * norms[coo.col]
    coo.data = (coo.data / np.maximum(denom, 1e-12)).astype(np.float32, copy=False)
    sim = coo.tocsr()
    sim.eliminate_zeros()

    sim_topk = csr_row_topk(sim, k=k, min_score=min_score)
    sim_topk = sim_topk.maximum(sim_topk.T).tocsr()
    sim_topk.setdiag(0.0)
    sim_topk.eliminate_zeros()
    return sim_topk


def pathsim_topk_from_bipartite(a_p: sp.csr_matrix, k: int, min_score: float = 0.0) -> sp.csr_matrix:
    # PathSim on A-P-A: 2 * |N(i)∩N(j)| / (|N(i)| + |N(j)|), robust to prolific-author bias.
    cnt = (a_p @ a_p.T).tocsr()
    cnt.setdiag(0.0)
    cnt.eliminate_zeros()
    if cnt.nnz == 0:
        return cnt

    deg = np.asarray(a_p.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    coo = cnt.tocoo()
    denom = np.maximum(deg[coo.row] + deg[coo.col], 1e-12)
    ps = (2.0 * coo.data.astype(np.float32, copy=False)) / denom
    ps = np.clip(ps, 0.0, 1.0).astype(np.float32, copy=False)

    sim = sp.csr_matrix((ps, (coo.row, coo.col)), shape=cnt.shape)
    sim.eliminate_zeros()
    sim_topk = csr_row_topk(sim, k=k, min_score=min_score)
    sim_topk = sim_topk.maximum(sim_topk.T).tocsr()
    sim_topk.setdiag(0.0)
    sim_topk.eliminate_zeros()
    return sim_topk


def tfidf_transform(x: sp.csr_matrix) -> sp.csr_matrix:
    x = x.tocsr().astype(np.float32)
    n = x.shape[0]
    if x.nnz == 0:
        return x
    # Document frequency per column.
    df = np.asarray((x > 0).sum(axis=0)).reshape(-1).astype(np.float32, copy=False)
    idf = np.log((1.0 + float(n)) / (1.0 + df)) + 1.0
    idf = idf.astype(np.float32, copy=False)
    x = x @ sp.diags(idf, offsets=0, format="csr")
    return x.tocsr()


def normalize_count_topk(count_mat: sp.csr_matrix, k: int, min_score: float = 0.0) -> sp.csr_matrix:
    count_mat = count_mat.tocsr()
    count_mat.setdiag(0.0)
    count_mat.eliminate_zeros()
    if count_mat.nnz == 0:
        return count_mat
    data = np.log1p(count_mat.data.astype(np.float32, copy=False))
    max_v = float(data.max()) if data.size > 0 else 1.0
    data = data / max(1e-12, max_v)
    out = sp.csr_matrix((data, count_mat.indices.copy(), count_mat.indptr.copy()), shape=count_mat.shape)
    out_topk = csr_row_topk(out, k=k, min_score=min_score)
    out_topk = out_topk.maximum(out_topk.T).tocsr()
    out_topk.setdiag(0.0)
    out_topk.eliminate_zeros()
    return out_topk


def build_author_graph(
    root: Path,
    mode: str,
    k_apa: int,
    k_term: int,
    k_conf: int,
    min_apa: float,
    min_term: float,
    min_conf: float,
    w_apa: float,
    w_term: float,
    w_conf: float,
):
    ds = DBLP(root=str(root))
    data = ds[0]

    num_a = int(data["author"].num_nodes)
    num_p = int(data["paper"].num_nodes)
    num_t = int(data["term"].num_nodes)
    num_c = int(data["conference"].num_nodes)

    ap = data["author", "to", "paper"].edge_index.cpu().numpy()
    pt = data["paper", "to", "term"].edge_index.cpu().numpy()
    pc = data["paper", "to", "conference"].edge_index.cpu().numpy()

    a_p = sp.coo_matrix(
        (np.ones(ap.shape[1], dtype=np.float32), (ap[0].astype(np.int64), ap[1].astype(np.int64))),
        shape=(num_a, num_p),
    ).tocsr()
    p_t = sp.coo_matrix(
        (np.ones(pt.shape[1], dtype=np.float32), (pt[0].astype(np.int64), pt[1].astype(np.int64))),
        shape=(num_p, num_t),
    ).tocsr()
    p_c = sp.coo_matrix(
        (np.ones(pc.shape[1], dtype=np.float32), (pc[0].astype(np.int64), pc[1].astype(np.int64))),
        shape=(num_p, num_c),
    ).tocsr()

    a_t = (a_p @ p_t).tocsr()
    a_c = (a_p @ p_c).tocsr()
    mode = str(mode).lower().strip()
    if mode not in {"v1", "v2"}:
        raise ValueError(f"Unknown mode: {mode}. Expected one of ['v1','v2'].")

    if mode == "v1":
        # V1: count/cosine baseline.
        apa_counts = (a_p @ a_p.T).tocsr()
        s_apa = normalize_count_topk(apa_counts, k=k_apa, min_score=min_apa)
        s_term = cosine_sim_topk_from_csr(a_t, k=k_term, min_score=min_term)
        s_conf = cosine_sim_topk_from_csr(a_c, k=k_conf, min_score=min_conf)
        channel_semantics = {
            "apa": "shared-paper affinity (log-count normalized)",
            "term": "author-term cosine similarity",
            "conf": "author-conference cosine similarity",
        }
    else:
        # V2: more principled semantics.
        # - PathSim for collaboration overlap.
        # - TF-IDF + cosine for topic/venue proximity.
        s_apa = pathsim_topk_from_bipartite(a_p, k=k_apa, min_score=min_apa)
        s_term = cosine_sim_topk_from_csr(tfidf_transform(a_t), k=k_term, min_score=min_term)
        s_conf = cosine_sim_topk_from_csr(tfidf_transform(a_c), k=k_conf, min_score=min_conf)
        channel_semantics = {
            "apa": "PathSim on A-P-A (collaboration overlap normalized by author productivity)",
            "term": "TF-IDF weighted author-term cosine (topic specialization proximity)",
            "conf": "TF-IDF weighted author-conference cosine (venue ecosystem proximity)",
        }

    w_apa = float(w_apa)
    w_term = float(w_term)
    w_conf = float(w_conf)
    weight_sum = max(1e-12, w_apa + w_term + w_conf)
    w_apa, w_term, w_conf = w_apa / weight_sum, w_term / weight_sum, w_conf / weight_sum

    w_mat = (w_apa * s_apa + w_term * s_term + w_conf * s_conf).tocsr()
    w_mat.setdiag(0.0)
    w_mat.eliminate_zeros()

    coo = w_mat.tocoo()
    rows = coo.row.astype(np.int64, copy=False)
    cols = coo.col.astype(np.int64, copy=False)
    edge_weight = coo.data.astype(np.float32, copy=False)

    # Gather per-edge multi-relation channels.
    ch_apa = np.asarray(s_apa[rows, cols]).reshape(-1).astype(np.float32, copy=False)
    ch_term = np.asarray(s_term[rows, cols]).reshape(-1).astype(np.float32, copy=False)
    ch_conf = np.asarray(s_conf[rows, cols]).reshape(-1).astype(np.float32, copy=False)
    edge_attr = np.stack([ch_apa, ch_term, ch_conf], axis=1).astype(np.float32, copy=False)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64, copy=False)

    feat = data["author"].x.cpu().numpy().astype(np.float32, copy=False)
    label = data["author"].y.cpu().numpy().astype(np.int64, copy=False)

    deg = np.bincount(rows, minlength=num_a).astype(np.int64)
    summary = {
        "source": "PyG DBLP (MAGNN split) author graph",
        "mode": mode,
        "num_nodes": int(num_a),
        "num_edges_directed": int(edge_index.shape[1]),
        "num_edges_undirected_est": int(edge_index.shape[1] // 2),
        "node_feat_dim": int(feat.shape[1]),
        "edge_attr_dim": int(edge_attr.shape[1]),
        "num_classes": int(np.unique(label).shape[0]),
        "class_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(label, return_counts=True))},
        "isolated_nodes": int((deg == 0).sum()),
        "isolated_ratio": float((deg == 0).sum() / max(1, num_a)),
        "channel_nonzero_ratio": {
            "apa": float((edge_attr[:, 0] > 0).mean()) if edge_attr.size else 0.0,
            "term": float((edge_attr[:, 1] > 0).mean()) if edge_attr.size else 0.0,
            "conf": float((edge_attr[:, 2] > 0).mean()) if edge_attr.size else 0.0,
        },
        "channel_mean": {
            "apa": float(edge_attr[:, 0].mean()) if edge_attr.size else 0.0,
            "term": float(edge_attr[:, 1].mean()) if edge_attr.size else 0.0,
            "conf": float(edge_attr[:, 2].mean()) if edge_attr.size else 0.0,
        },
        "channel_semantics": channel_semantics,
        "params": {
            "k_apa": int(k_apa),
            "k_term": int(k_term),
            "k_conf": int(k_conf),
            "min_apa": float(min_apa),
            "min_term": float(min_term),
            "min_conf": float(min_conf),
            "w_apa": float(w_apa),
            "w_term": float(w_term),
            "w_conf": float(w_conf),
        },
    }
    return edge_index, edge_weight, edge_attr, feat, label, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PyG DBLP (MAGNN) for ATsDataset format.")
    parser.add_argument("--root", type=str, default="data/pyg_dblp", help="PyG DBLP root")
    parser.add_argument("--out_root", type=str, default="data", help="Output root")
    parser.add_argument("--name", type=str, default="dblp_magnn_author", help="Output dataset name")
    parser.add_argument("--mode", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--k_apa", type=int, default=32)
    parser.add_argument("--k_term", type=int, default=32)
    parser.add_argument("--k_conf", type=int, default=12)
    parser.add_argument("--min_apa", type=float, default=0.0)
    parser.add_argument("--min_term", type=float, default=0.05)
    parser.add_argument("--min_conf", type=float, default=0.05)
    parser.add_argument("--w_apa", type=float, default=0.45)
    parser.add_argument("--w_term", type=float, default=0.40)
    parser.add_argument("--w_conf", type=float, default=0.15)
    args = parser.parse_args()

    edge_index, edge_weight, edge_attr, feat, label, summary = build_author_graph(
        root=Path(args.root),
        mode=args.mode,
        k_apa=args.k_apa,
        k_term=args.k_term,
        k_conf=args.k_conf,
        min_apa=args.min_apa,
        min_term=args.min_term,
        min_conf=args.min_conf,
        w_apa=args.w_apa,
        w_term=args.w_term,
        w_conf=args.w_conf,
    )

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.name}_edge_index.npy", edge_index)
    np.save(out_dir / f"{args.name}_edge_weight.npy", edge_weight)
    np.save(out_dir / f"{args.name}_edge_attr.npy", edge_attr)
    np.save(out_dir / f"{args.name}_feat.npy", feat)
    np.save(out_dir / f"{args.name}_label.npy", label)

    meta = {
        "source_dataset": "torch_geometric.datasets.DBLP",
        "source_variant": "MAGNN processed split",
        "mode": args.mode,
        "label_mapping": {},
        "edge_attr_channels": [
            "apa_shared_paper_affinity",
            "aptpa_term_cosine",
            "apcpa_conference_cosine",
        ],
        "edge_attr_channel_semantics": summary.get("channel_semantics", {}),
        "note": "All labels are for author nodes (4 classes).",
    }
    with open(out_dir / f"{args.name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[ok] {args.name}: nodes={summary['num_nodes']}, "
        f"edges={summary['num_edges_directed']}, iso_ratio={summary['isolated_ratio']:.4f}"
    )
    print(f"[ok] files written to: {out_dir}")


if __name__ == "__main__":
    main()
