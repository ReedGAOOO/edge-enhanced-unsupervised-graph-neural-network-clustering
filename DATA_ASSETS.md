# Data Assets

Last updated: 2026-10-15 (dataset-copy verification remains dated 2026-10-13)

Large or ignored assets of this repo that are not stored in Git. Each `###` block is one asset; fields are single-line
`- key: value` entries.

## Local external (Windows drives via WSL)

### prepared_datasets
- role: all prepared graph datasets (PyG caches, Cora/Photo/..., DBLP v2, Entities, Fraud, Bitcoin WSN, synth_mech, synth_ctx_v1, Urban v3bsjg/v5/hetero, urban_network_datasets raw cities, urban_factor_experiments derived graphs)
- location: data/ (fstab bind mount of /mnt/d/WSL Projects Data/home/aitx/workspace/projects/edge-enhanced-unsupervised-graph-neural-network-clustering/data, disk PC811_2 = D:)
- size: 17,699,615,134 bytes; 2,309 regular files, 322 directories, 208 symlinks
- provenance: copied 2026-10-13 from /mnt/f/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/data (rsync -rt); the D mirror path had been empty, so data/ showed nothing
- symlinks: the 208 links under urban_factor_experiments/ pointed to /mnt/e/wsl-project-data/... (already dangling on F); in the D copy they are relative links to the base datasets (../../<dataset>/<file>), 0 dangling
- verification: file count and total bytes match the source; sha256 match on 4 sample files (dblp_magnn_author_v2 edge_attr, synth_ctx_v1_h45_contextual edge_attr, urban_washingtondc_plot_v3bsjg edge_index, Cora/raw/ind.cora.x)
- added after the copy (2026-10-13, D: only, not in the F: backup): pyg_imdb/, pyg_hgb/ (PyG downloads), imdb_multiplex_v1, acm_multiplex_v1 (2.2M edges), acm_multiplex_v2 (PSP sampled to 20); all regenerable with tools/prepare_pyg_multiplex_dataset.py
- status: verified
- reproducibility: synth_ctx_v1_* regenerate with tools/prepare_synth_ctx_datasets.py; most others come from builders in tools/ or archive/code/historical_builders/ plus public downloads; Urban raw city data is not regenerable from this repo

### prepared_datasets_f_backup
- role: original copy of prepared_datasets (backup)
- location: /mnt/f/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/data (disk SN730 = F:)
- status: verified source of the 2026-10-13 copy; its urban_factor_experiments symlinks are dangling (absolute /mnt/e/wsl-project-data targets)
- notes: keep until the D copy has been used in practice; not referenced by any config

### expert_workspace_results
- role: April 2026 EP1/EP2/Expert3 run outputs and checkpoints
- location: payload at /mnt/f/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/archive/workspaces/; repo symlinks archive/workspaces/*/results and */checkpoints still point to /mnt/e/WSL Projects Data/edge-enhanced-unsupervised-graph-neural-network-clustering/archive/workspaces/ (dangling since the drive letter changed)
- status: payload present on F, symlinks unreachable
- notes: historical only (runs used an untrained backbone); repoint the symlinks if these outputs are needed again

## Local, ignored by Git

### breakthrough_results
- role: E1-E16 breakthrough-pass outputs, E9x machine checks, E10 SE-alignment screen, B45 gate diagnostics, E12/E15 search outputs (ledgers, tables, logs, metrics, candidate CSVs and probe JSON)
- location: results/breakthrough_v1/ (E1-E16, e5_headroom, e9x_machine_check, e10_se_alignment, e12_se_guided_baseline, e15_search_robustness, b45_gate_diagnosis)
- producer: tools/run_breakthrough.py with configs/breakthrough/*.json; tables from tools/report_e2_mechanism.py and tools/report_e3_real.py
- status: full raw outputs remain local with no backup recorded; headline numbers are in README.md and compact E9/E10/B45 evidence tables are tracked under configs/breakthrough/evidence/
- notes: results/breakthrough_v1/_code/ holds patched copies of the expert1/expert2 workspaces (warm start via encoder hook) used for EP1/EP2
- completion: E9 complete (92 runs; the 48 synthetic runs were trained on the ThinkBook RTX 4060 Laptop and copied back); E11 57, E13 42, E14 42, E16 12 runs; E14-E16 per-run directories were copied back from the RTX 4090 server; E9x has two runs; B45 DBLP paired training was interrupted and is excluded from conclusions

### breakthrough_remote_copies
- role: working copies used to run E9 (synthetic part), E14, E15 and E16 off the local GPU
- location: RTX 4090 server `/data/yijie/projects/edge-enhanced-unsupervised-graph-neural-network-clustering` (uv env, data subset, the `*_sepick` derived datasets, E14/E16 checkpoints of about 190 MB plus E16's); ThinkBook 16p `/root/edge-enhanced-unsupervised-graph-neural-network-clustering` (E9 run copy)
- producer: rsync of the Git-tracked files plus the needed datasets; access through `/home/aitx/workspace/tools/ual-4090-access/ssh_helper.py`
- status: result directories are copied back into results/breakthrough_v1/; the `*_sepick` datasets and E14/E16 checkpoints exist only on the 4090 copy and are regenerable from the documented picks and plans
- notes: the half-installed RTX 5090 copy was deleted on 2026-10-15

### breakthrough_checkpoints
- role: model checkpoints written during breakthrough runs
- location: checkpoints/breakthrough_v1/ (historical size: about 930 MB after E1-E8; not a current size measurement)
- status: regenerable from the plans; safe to delete
- notes: checkpoint inspection/graph-cross tools need these matching files; preserve them when reproducing the documented gate diagnosis
