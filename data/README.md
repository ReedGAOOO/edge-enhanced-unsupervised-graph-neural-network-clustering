# Project data

Updated 2026-10-15. Dataset files are excluded from Git; this manifest is tracked. A clone contains no prepared
datasets or Urban raw city files. There is no published full-data mirror configured in this repository.

See [DATASETS.md](../DATASETS.md) for dataset meaning and experimental roles, and
[DATA_ASSETS.md](../DATA_ASSETS.md) for the original workstation's D: bind mount and F: backup. A normal local
directory works on other machines; pass `--root_path /path/to/data` to select it.

## Getting the current benchmarks

Run from the repository root with the environment described in [UV_ENVIRONMENTS.txt](../UV_ENVIRONMENTS.txt).

| Dataset | How it is obtained | Role |
|---|---|---|
| `cora`, `citeseer`, `pubmed`, `computers`, `photo` | Downloaded by the training loader when missing | Standard topology/feature controls; attributes are derived |
| `dblp_magnn_author_v2` | `tools/prepare_pyg_dblp_magnn_dataset.py` | Native multi-channel author graph |
| `acm_multiplex_v2` | `tools/prepare_pyg_multiplex_dataset.py` | PAP/PSP/PcP/PTP union; sampled PSP for the reported experiments |
| `entities_aifb` | Archived `prepare_pyg_entities_datasets.py` converter | Sparse known labels, relation attributes |
| `synth_ctx_v1_*` | `tools/prepare_synth_ctx_datasets.py` | Current weak-feature, direct/contextual edge benchmark |
| `urban_<city>_plot_v3bsjg` and hetero/v5 families | Urban builders plus separately supplied `urban_network_datasets/<city>/` | Data-construction and validity track |

```bash
.venv/bin/python tools/prepare_pyg_dblp_magnn_dataset.py --root data/pyg_dblp --out_root data --name dblp_magnn_author_v2 --mode v2
.venv/bin/python tools/prepare_pyg_multiplex_dataset.py --dataset acm --name acm_multiplex_v2 --psp_sample_k 20
.venv/bin/python tools/prepare_synth_ctx_datasets.py --root_path data --feature_sep 0.35 --homophily 0.45,0.55
.venv/bin/python archive/code/historical_builders/prepare_pyg_entities_datasets.py --help
```

The last command shows converter options; `main.py --dataset entities_aifb` expects its prepared files already present.
Legacy Fraud, Entities and Bitcoin converters remain under `archive/code/historical_builders/`.
`synth_mech_full_v1_*` and `synth_edgectrl_v1_*` are historical controls whose features reveal the labels;
do not use their scores as current edge-semantics evidence.

## Custom homogeneous graphs

Place the following files in `<root_path>/<name>/`:

```text
<name>_edge_index.npy   # required: [2, E], integer indices in [0, N)
<name>_feat.npy         # required: [N, F]
<name>_label.npy        # required: [N]; -1 for unknown labels
<name>_edge_weight.npy  # optional: [E]
<name>_edge_attr.npy    # optional: [E, D], aligned with edge_index columns
<name>_meta.json        # optional: channel names, unknown-label mapping, provenance
```

```bash
.venv/bin/python tools/run_preset.py --dataset my_graph --root_path data --max_nums 12 --seed 0
```

`max_nums` is the hierarchy's parent-node budget. The warm-start pseudo-partition size defaults to the known class count;
use `main.py --warm_start_k` to set it explicitly. Known-only evaluation is enabled by default and maps an explicitly
declared `unknown` class to `-1`. Fully unlabelled graphs cannot supply supervised evaluation metrics.
The heterogeneous Urban exports use their own schema and runners; they are not interchangeable with this sparse format.
