# Datasets

This file describes the datasets that currently exist under `data/` in this repo, with emphasis on what is actually useful for experiments now.

Updated 2026-10-15. Dataset payloads are not included in Git; see [data/README.md](data/README.md) for construction
commands and the custom sparse format. E9 covers Cora, Photo, DBLP v2, AIFB and ACM v2; its synthetic jobs remain
uncompleted. Current comparisons and their evaluation protocol are in [README.md](README.md).

> **Location on the original workstation.** `data/` is an fstab bind mount of the D: mirror folder
> `/mnt/d/WSL Projects Data/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/data`
> (populated 2026-10-13 from the F: copy). See `DATA_ASSETS.md` for locations, backups and verification.
>
> **Classic floors.** `tools/run_classic_baselines.py` reports KMeans(X), spectral(A), Louvain(A) and SGC(A,X)+KMeans
> for any dataset here. Check a dataset against these before reading anything into a deep-model number
> (results: `results/breakthrough_v1/*/classic_baselines.csv`).

## Mainline Datasets

| Dataset | Graph Type | Edge Semantics | Labels | Current Role |
|---|---|---|---|---|
| `cora` | citation homogeneous graph | low | paper topic classes | Small sanity benchmark. Good for checking that training runs, but weak evidence for native edge semantics. |
| `citeseer` | citation homogeneous graph | low | paper topic classes | Same role as `cora`, with a more fragmented graph. Useful for robustness checks. |
| `pubmed` | citation homogeneous graph | low | paper topic classes | Larger citation benchmark. Mostly used to check scalability on a standard graph. |
| `computers` | Amazon co-purchase homogeneous graph | low | product categories | Non-citation sanity benchmark. Edges are useful, but not relation-rich in the same sense as Urban, DBLP, or KG graphs. |
| `photo` | Amazon co-purchase homogeneous graph | low | product categories | Same role as `computers`, with a smaller product-domain graph. |
| `dblp_magnn_author_v2` | author-author semantic graph | high | author area classes | Main relation-aware academic benchmark. Edges fuse collaboration, topic, and venue proximity into a native multi-channel author graph. Strong topology: spectral(A) alone reaches NMI 0.728, so any model must clear that floor. |
| `entities_aifb` | relation KG graph | medium-high | sparse known node labels, `unknown=-1` | Small knowledge-graph benchmark with explicit relation types. Good for checking relation-aware behavior under sparse labels. |
| `entities_mutag` | relation KG graph | medium-high | sparse known node labels, `unknown=-1` | Small KG benchmark, lighter than `BGS/AM`, useful for quick relation-graph experiments. |
| `entities_bgs` | relation KG graph | high | sparse known node labels, `unknown=-1` | Large KG benchmark with many relations and very low known-label ratio. Useful for harder relation-rich tests. |
| `entities_am` | relation KG graph | high | sparse known node labels, `unknown=-1` | Very large KG benchmark with many relation types and many classes. One of the harder real relation graphs in the repo. |
| `entities_bgs_top10k` | KG subgraph | medium-high | sparse known node labels, `unknown=-1` | A 10k-node cut of `entities_bgs` for cheaper repeated experiments. |
| `entities_am_top10k` | KG subgraph | medium-high | sparse known node labels, `unknown=-1` | A 10k-node cut of `entities_am` for lighter debugging and ablations. |
| `fraud_amazon_union` | fraud multi-relation graph | high | binary fraud labels | Real fraud graph derived from CARE-GNN Amazon data. Edge channels encode `homo`, `upu`, `usu`, and `uvu` relations. |
| `fraud_yelp_homo` | fraud multi-relation graph | high | binary fraud labels | Real fraud graph derived from CARE-GNN Yelp data. Edge channels encode `homo`, `rur`, `rtr`, and `rsr` relations. |
| `synth_ctx_v1_h{45,55}_{none,direct,contextual}_ds00[_permEA]` | weak-feature SBM benchmark | highest | 4 balanced synthetic classes | **Current synthetic benchmark** (`tools/prepare_synth_ctx_datasets.py`, feature_sep 0.35). KMeans(X) 0.22; h45 is below the Kesten-Stigum threshold for topology alone (spectral 0.03, SGC 0.47); h55 gives spectral 0.45, SGC 0.73. `direct`: edge channel 0 = noisy same-label sign. `contextual`: the sign is flipped when endpoint roles differ (roles are the last two feature columns), so it is marginally uninformative. `none` and `_permEA` are controls. Oracle edge-weighted SGC reaches 0.84-0.96, so edges have real headroom. |
| `acm_multiplex_v2` (and `_v1`) | paper graph, union of meta-path layers PAP/PSP/PcP/PTP | high | 3 paper classes | **Real graph with edge-type headroom** (`tools/prepare_pyg_multiplex_dataset.py`, HGB-ACM). Edge attributes are per-layer membership + PathSim. Layer homophily: PcP 0.88, PAP 0.79, PTP 0.69, PSP 0.67. SGC+KMeans on the union 0.55, on PcP alone 0.71. v1 keeps all 2.2M PSP edges (too heavy for DSE: 22 GB, 5.5 s/epoch); v2 samples 20 same-subject neighbours per paper (188k edges). |
| `imdb_multiplex_v1` | movie graph, union of MDM/MAM | — | 3 genres | Built the same way from PyG IMDB. Genre labels are unrecoverable: every classic method and every layer subset scores NMI ≤ 0.05. Not useful. |
| `synth_mech_full_v1_*`, `synth_edgectrl_v1_*` | synthetic mechanism graphs | — | synthetic community labels | **Deprecated as evidence.** Node features alone give KMeans NMI 1.000 on every instance (feature_sep 1.5 in 64 dims), and edge channel 0 is a noisy copy of the pairwise same-label bit, so these graphs cannot show whether edges help. |
| `urban_<city>_plot_v3bsjg` | plot-level urban homogeneous graph | high | land-use-derived plot labels with `unknown` | **Not a clustering benchmark.** `unknown` is stored as class 8 (84% of plots in WashingtonDC/Boston, 40% in Paris); evaluate with known-only labels (now the `main.py` default). With known-only labels every classic method scores NMI 0.02-0.07 while Louvain reaches modularity 0.90+, i.e. land-use labels are unrelated to the graph's community structure. Keep for the Urban data-paper track. Nodes are plots; edges come from shared street, junction-only context, and very sparse geometric fallback. Current instances: `bangkok`, `beijing`, `boston`, `chicago`, `johannesburg`, `madrid`, `melbourne`, `paris`, `shanghai`, `singapore`, `sydney`, `tokyo`, `washingtondc`. |
| `urban_<city>_plot_hetero_v1` | plot-centric urban heterogeneous graph | high | plot labels for evaluation | First hetero Urban export family. Keeps `plot`, `street`, `intersection`, and `building` as separate node types with typed relations, but edge attributes are still relatively light. Current instances: `beijing`, `paris`. |
| `urban_<city>_plot_hetero_v2` | plot-centric urban heterogeneous graph with typed edge attributes | high | plot labels for evaluation | Current richer hetero Urban design. It keeps the same four node types, upgrades each relation with interpretable geometry/topology/containment edge features, and separates spatial building-building edges from semantic building-building edges. Local directories exist for the same 13 cities listed for V3b; directory presence does not establish validation or model-training coverage. |

## Auxiliary Or Legacy Prepared Datasets

| Dataset | Graph Type | Edge Semantics | Labels | Current Role |
|---|---|---|---|---|
| `bitcoin_wsn_alpha` | signed weighted social/trust graph | medium-high | all unknown | Historical prepared graph from Bitcoin Alpha. Edge attributes encode signed normalized weight, magnitude, and sign. |
| `bitcoin_wsn_otc` | signed weighted social/trust graph | medium-high | all unknown | Historical prepared graph from Bitcoin OTC. Same schema as `bitcoin_wsn_alpha`. |
| `bitcoin_wsn_rfa` | signed weighted social/trust graph | medium-high | all unknown | Historical prepared graph from RFA signed interactions. Same schema as the other Bitcoin WSN variants. |
| `bitcoin_wsn_wikisigned` | signed weighted social/trust graph | medium-high | all unknown | Large signed graph built from WikiSignedNet. Useful if signed-edge experiments are resumed. |
| `bitcoin_wsn_epinion` | signed weighted social/trust graph | medium-high | all unknown | Large signed graph built from Epinion trust data. |
| `bitcoin_wsn_epinion_top5k` | signed weighted subgraph | medium-high | all unknown | A 5k-node cut of `bitcoin_wsn_epinion` used to make the very large graph easier to experiment with. |
| `bitcoin_wsn_wikisigned_top10k` | signed weighted subgraph | medium-high | all unknown | A 10k-node cut of `bitcoin_wsn_wikisigned` for lighter debugging and ablations. |

## Raw Or Source Directories

These directories are present under `data/`, but they are better understood as source material, cached downloads, or historical staging areas rather than the main experimental datasets.

| Directory | Meaning |
|---|---|
| `Bitcoin_WSN` | Source CSVs, PDF, and preparation summary for the Bitcoin WSN family. |
| `FraudAmazon` | Raw `Amazon.mat` source for `fraud_amazon_union`. |
| `FraudYelp` | Raw `YelpChi.mat` source for `fraud_yelp_homo`. |
| `Entities` | Preparation summary and cache area for the PyG Entities family. |
| `pyg_dblp` | Download/cache root used when building `dblp_magnn_author_v2`. |
| `aifb`, `am`, `bgs`, `mutag` | PyG raw/cache directories for the Entities benchmarks, not the final repo-specific datasets. |
| `Fraud` | Historical summary directory for older fraud conversions. |
| `MetroFlow` | Reference material directory, not a current experiment-ready dataset in this repo. |

## Practical Reading Order

If you only want the datasets that matter for the current research direction, read them in this order:

1. `synth_ctx_v1_*` (mechanism evidence: does a model use edges at all?)
2. `dblp_magnn_author_v2`, `entities_aifb` (native relations)
3. `cora`, `photo` (derived edges, sanity only)
4. `urban_<city>_plot_v3bsjg` / `hetero_v2` for the Urban data-paper track, not for clustering claims

If a directory only exists as a raw source or cache, prefer the processed dataset built from it instead of pointing experiments at the raw directory directly.
