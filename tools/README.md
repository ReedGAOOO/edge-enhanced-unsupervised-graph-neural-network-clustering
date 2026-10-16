# Project tools

Updated 2026-10-15 (evening). Run from the repository root using `.venv/bin/python`; each Python tool provides `--help`.
The [environment declaration](../UV_ENVIRONMENTS.txt), [dataset guide](../data/README.md) and
[experiment plans](../configs/breakthrough/README.md) describe shared prerequisites.

## Training and result tables

| Tool | Purpose |
|---|---|
| `run_preset.py` | One preset run; B31 default, 2000 epochs, cold start; `--dry_run` previews the command |
| `run_breakthrough.py` | Expand JSON plans and launch concurrent training jobs on one GPU; `--num_shards/--shards` split a plan across machines |
| `run_classic_baselines.py` | KMeans(X), spectral(A), Louvain(A), SGC(A,X)+KMeans floors; also provides `spectral_reg`, regularized spectral clustering |
| `run_se_guided_baseline.py` | Label-free search over explicit edge weightings (edge channels, multiplex layers, endpoint context, products), scored by two-level SE; `--spectral reg`, `--order2`, `--beam B --max_order R` |
| `write_se_reweighted_dataset.py` | Write a search pick as `<name>_edge_weight.npy` next to a copy of the dataset, for training the edge-free DSE on it |
| `se_select_decoder.py` | For runs on a reweighted graph, compare the DSE assignment with spectral / SGC partitions by two-level SE (`--spectral reg`) |
| `make_edge_cv_split.py` | Hold out a fraction of undirected edges (no new isolated nodes); writes `<name>_cv80/` and its held-out pairs |
| `few_label_edge_semantics.py` | E21 methods 0-2 with m labeled nodes: node classification, SE top-10 + label pick, supervised edge classifiers |
| `few_label_selection.py` | How many random labeled nodes the choice among candidates needs (ARI on the labeled subset; NMI and ARI regret) |
| `run_selection_validation.py` | E19 driver: new decisions on Citeseer / Computers / PubMed and seed-1 synthetic graphs, end to end and resumable |
| `select_partition_rules.py` | Score label-free selection rules (SE, medoid, edge-CV SE, two-view SE, feature-gated edge-CV) on a JSON list of decisions |
| `report_e2_mechanism.py` | Synthetic edge-information controls and effect tables |
| `report_e3_real.py` | Merge real runs, choose starts using baseline SE, write tables; supports `--out_dir` |
| `build_results_index.py` | Utility for indexing local result files |

Neither the preset runner nor the batch runner automatically chooses between model families. The reporting script
selects starts after both runs exist. `max_nums` is a hierarchy budget, not a guaranteed final cluster count.

## Edge diagnostics

| Tool | Inputs and interpretation |
|---|---|
| `screen_edge_headroom.py` | Prepared graphs; supervised same-label edge classifier and graph reweighting, for diagnosis only |
| `check_se_alignment.py` | Prepared graphs; compares label SE with spectral/SGC partitions over candidate weights; skips `-1` labels |
| `check_spectral_stability.py` | Prepared graphs; plain vs regularized spectral NMI across eigensolver start vectors, with component counts |
| `inspect_edge_factors.py` | A run directory with `stdout.txt`, its dataset and matching checkpoint; inspects learned scalar heads |
| `diagnose_b45_message_gate.py` | A run directory and data; checkpoint interventions or controlled short paired training |
| `prepare_edge_intervention_dataset.py` | Prepared graph; creates edge-attribute controls |

```bash
# Search, write the pick, train the edge-free DSE, then compare decoders (labels only score the results)
.venv/bin/python tools/run_se_guided_baseline.py --datasets acm_multiplex_v2 --seeds 0 --spectral reg --workers 8 \
  --out_dir results/breakthrough_v1/search_acm
.venv/bin/python tools/write_se_reweighted_dataset.py --dataset acm_multiplex_v2 --pick=layer+PAP --out_name acm_sepick
.venv/bin/python tools/se_select_decoder.py --spectral reg --run_dirs results/breakthrough_v1/e14_search_then_train/runs/*acm_sepick*/ \
  --out_csv results/breakthrough_v1/e14_search_then_train/decoder_select_acm.csv

.venv/bin/python tools/check_se_alignment.py --datasets cora,photo,dblp_magnn_author_v2,acm_multiplex_v2 \
  --out_csv results/breakthrough_v1/e10_se_alignment/se_alignment.csv

# Cross embedding/assignment graphs using an existing B45 Photo checkpoint
.venv/bin/python tools/diagnose_b45_message_gate.py \
  --run_dir results/breakthrough_v1/e3_real_dual/runs/b45_warm__photo__s0 \
  --graph_cross --out results/breakthrough_v1/b45_gate_diagnosis/photo_graph_cross.json

# Fresh paired probe: common initialization, fixed SGC pseudo partition, fixed final epoch
.venv/bin/python tools/diagnose_b45_message_gate.py \
  --run_dir results/breakthrough_v1/e3_real_dual/runs/b45_warm__photo__s0 \
  --train --seed 0 --warm_epochs 200 --epochs 200 \
  --modes original off unit_gate no_renorm no_gate_dropout safe_isolates \
  --out results/breakthrough_v1/b45_gate_diagnosis/photo_paired_s0.json
```

Without `--train` or `--graph_cross`, the gate tool evaluates a fixed list of checkpoint modes and requires a checkpoint.
`--modes` configures the training probe only. `--cpu` forces CPU execution. Fresh training needs the source run's
configuration in `stdout.txt`, but does not load its trained weights. It writes JSON progress and a `.pseudo.npy`
partition; the short-budget scores are diagnostics rather than full benchmark results. The recorded `safe_isolates`
result was run separately with the same seed and initialization protocol. No hook here patches production source files.

## Dataset builders

| Family | Entry points |
|---|---|
| Current synthetic benchmark | `prepare_synth_ctx_datasets.py --feature_sep 0.35 --homophily 0.45,0.55`; `--prefix synth_ctxxor_v1 --role_mode xor2` builds the second-order (XOR-role) variant |
| DBLP author graph | `prepare_pyg_dblp_magnn_dataset.py --mode v2` |
| ACM / IMDB meta-path union | `prepare_pyg_multiplex_dataset.py`; ACM v2 uses `--psp_sample_k 20` |
| Urban V3 / V3b | `prepare_urban_plot_graph_v3.py`, `prepare_urban_plot_graph_v3b.py`, matching `run_*_series.py` |
| Urban typed graphs | `prepare_urban_plot_hetero_v1.py`, `prepare_urban_plot_hetero_v2.py`, `validate_urban_plot_hetero_v1.py`, `smoke_test_urban_plot_hetero_v2.py` |
| Urban semantic / native variants | `prepare_urban_plot_graph_v4_semantic.py`, `prepare_urban_plot_graph_v5_native.py` |
| Historical synthetic reproduction | `prepare_mechanism_synth_datasets.py`; not current edge-information evidence |

Older Entities/Fraud/Bitcoin and Urban V1/V2 builders live in
[archive/code/historical_builders/](../archive/code/historical_builders/README.md).

## Urban analysis and reports

`run_urban_data_paper_pipeline.py` and the `run_urban_*_experiments.py` tools support the Urban data track.
`analyze_raw_urban_city.py`, `analyze_urban_raw_all_cities.py`, `run_raw_urban_analysis_all.py` and related city/factor
diagnostics inspect source quality and graph construction. `plot_*.R`, `render_*.R`, `visualize_*.py`, `export_urban_*.py`
and `build_folium_cluster_map.py` produce figures, report inputs and maps. See [reports/README.md](../reports/README.md)
before reusing older Urban model comparisons. R rendering and geospatial tools may need packages outside the root
Python lock; inspect the selected script's imports first.
