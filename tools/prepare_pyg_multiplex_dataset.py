#!/usr/bin/env python3
"""Build multiplex (meta-path union) graphs from PyG heterographs: IMDB (movies) and HGB-ACM (papers).

Each meta-path layer becomes a homogeneous graph on the target node type. The output graph is the union of the
layers; every edge carries, per layer, a membership bit and the PathSim score 2*C_ij / (C_ii + C_jj) of that
layer (0 when the edge is absent from the layer). Layers differ in how often they connect same-class nodes, so
the edge type is a context-free signal of edge reliability — the kind of edge semantics B45 can exploit.

Layers: imdb = MDM (shared director), MAM (shared actor); acm = PAP (shared author), PSP (shared subject),
PcP (citation), PTP (shared term, top-k by PathSim because it is near-complete otherwise).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def bipartite(edge_index, n_src, n_dst):
    return sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n_src, n_dst)).tocsr()


def metapath_counts(b: sp.csr_matrix) -> sp.csr_matrix:
    c = (b @ b.T).tocsr()
    return c


def pathsim(c: sp.csr_matrix) -> sp.csr_matrix:
    diag = c.diagonal().astype(np.float64)
    coo = c.tocoo()
    keep = coo.row != coo.col
    r, cc, v = coo.row[keep], coo.col[keep], coo.data[keep]
    s = 2.0 * v / np.maximum(diag[r] + diag[cc], 1e-12)
    return sp.coo_matrix((s, (r, cc)), shape=c.shape).tocsr()


def row_topk(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(m.shape[0]):
        s, e = m.indptr[i], m.indptr[i + 1]
        if s == e:
            continue
        c, v = m.indices[s:e], m.data[s:e]
        if c.size > k:
            sel = np.argpartition(v, -k)[-k:]
            c, v = c[sel], v[sel]
        rows.append(np.full(c.size, i))
        cols.append(c)
        vals.append(v)
    out = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=m.shape).tocsr()
    return out.maximum(out.T).tocsr()


def row_sample(m: sp.csr_matrix, k: int, seed: int) -> sp.csr_matrix:
    """Keep k random neighbours per row (then symmetrise). Used for PSP, whose PathSim is 1 for every same-subject
    pair, so top-k by score would be arbitrary anyway."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(m.shape[0]):
        s, e = m.indptr[i], m.indptr[i + 1]
        if s == e:
            continue
        sel = np.arange(s, e) if e - s <= k else rng.choice(np.arange(s, e), k, replace=False)
        rows.append(np.full(sel.size, i))
        cols.append(m.indices[sel])
        vals.append(m.data[sel])
    out = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=m.shape).tocsr()
    return out.maximum(out.T).tocsr()


def load_layers(name: str, root: Path, ptp_topk: int, psp_sample_k: int = 0, seed: int = 0):
    if name == "imdb":
        from torch_geometric.datasets import IMDB
        g = IMDB(str(root / "pyg_imdb"))[0]
        n = g["movie"].num_nodes
        b_md = bipartite(g["movie", "to", "director"].edge_index.numpy(), n, g["director"].num_nodes)
        b_ma = bipartite(g["movie", "to", "actor"].edge_index.numpy(), n, g["actor"].num_nodes)
        layers = {"MDM": pathsim(metapath_counts(b_md)), "MAM": pathsim(metapath_counts(b_ma))}
        return g["movie"].x.numpy(), g["movie"].y.numpy(), layers
    if name == "acm":
        from torch_geometric.datasets import HGBDataset
        g = HGBDataset(str(root / "pyg_hgb"), name="ACM")[0]
        n = g["paper"].num_nodes
        b_pa = bipartite(g["paper", "to", "author"].edge_index.numpy(), n, g["author"].num_nodes)
        b_ps = bipartite(g["paper", "to", "subject"].edge_index.numpy(), n, g["subject"].num_nodes)
        b_pt = bipartite(g["paper", "to", "term"].edge_index.numpy(), n, g["term"].num_nodes)
        cite = g["paper", "cite", "paper"].edge_index.numpy()
        a_cite = sp.coo_matrix((np.ones(cite.shape[1]), (cite[0], cite[1])), shape=(n, n)).tocsr()
        a_cite = ((a_cite + a_cite.T) > 0).astype(np.float64).tocsr()
        a_cite.setdiag(0)
        a_cite.eliminate_zeros()
        psp = pathsim(metapath_counts(b_ps))
        if psp_sample_k > 0:
            psp = row_sample(psp, psp_sample_k, seed)
        layers = {"PAP": pathsim(metapath_counts(b_pa)), "PSP": psp, "PcP": a_cite}
        if ptp_topk > 0:
            layers["PTP"] = row_topk(pathsim(metapath_counts(b_pt)), ptp_topk)
        return g["paper"].x.numpy(), g["paper"].y.numpy(), layers
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=["imdb", "acm"])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--name", default="", help="Output dataset name (default <dataset>_multiplex_v1).")
    parser.add_argument("--ptp_topk", type=int, default=10)
    parser.add_argument("--layers", default="", help="Comma-separated subset of layers to keep (default all).")
    parser.add_argument("--psp_sample_k", type=int, default=0, help="ACM: keep k random same-subject neighbours per paper (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root_path)
    x, y, layers = load_layers(args.dataset, root, args.ptp_topk, args.psp_sample_k, args.seed)
    if args.layers:
        keep = [l.strip() for l in args.layers.split(",")]
        layers = {k: v for k, v in layers.items() if k in keep}
    name = args.name or f"{args.dataset}_multiplex_v1"
    n = x.shape[0]

    union = None
    for m in layers.values():
        b = (m > 0).astype(np.float64)
        union = b if union is None else union.maximum(b)
    union = union.tocoo()
    src, dst = union.row.astype(np.int64), union.col.astype(np.int64)
    attrs, layer_stats = [], {}
    for lname, m in layers.items():
        vals = np.asarray(m[src, dst]).ravel()
        member = (vals > 0).astype(np.float32)
        attrs += [member, vals.astype(np.float32)]
        mc = m.tocoo()
        off = mc.row != mc.col
        same = y[mc.row[off]] == y[mc.col[off]]
        layer_stats[lname] = {"edges_directed": int(off.sum()), "homophily": float(same.mean()) if same.size else None}
    edge_attr = np.stack(attrs, axis=1)
    hom_union = float(np.mean(y[src] == y[dst]))
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}_feat.npy", x.astype(np.float32))
    np.save(out / f"{name}_label.npy", y.astype(np.int64))
    np.save(out / f"{name}_edge_index.npy", np.stack([src, dst]).astype(np.int64))
    np.save(out / f"{name}_edge_weight.npy", np.ones(src.shape[0], dtype=np.float32))
    np.save(out / f"{name}_edge_attr.npy", edge_attr.astype(np.float32))
    meta = {
        "dataset": name, "source": f"torch_geometric ({'IMDB' if args.dataset == 'imdb' else 'HGBDataset ACM'})",
        "num_nodes": int(n), "num_classes": int(np.unique(y).size), "num_edges_directed": int(src.shape[0]),
        "homophily_union": hom_union, "layers": layer_stats,
        "edge_attr_channels": [f"{l}_{k}" for l in layers for k in ("member", "pathsim")],
        "ptp_topk": args.ptp_topk if args.dataset == "acm" else None,
        "psp_sample_k": args.psp_sample_k if args.dataset == "acm" else None, "label_mapping": {},
        "description": "Union of meta-path layers; per-layer membership bit and PathSim score as edge attributes.",
    }
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {name}: N={n} E={src.shape[0]} hom_union={hom_union:.3f} layers=" +
          ", ".join(f"{k}(E={v['edges_directed']}, hom={v['homophily']:.3f})" for k, v in layer_stats.items()))


if __name__ == "__main__":
    main()
