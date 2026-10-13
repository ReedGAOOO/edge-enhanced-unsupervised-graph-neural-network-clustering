# Data Assets

Last updated: 2026-10-13

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
- role: E1-E4 breakthrough-pass outputs (runs.csv, summary.csv, mechanism/real tables, classic floors, per-run logs and metrics)
- location: results/breakthrough_v1/ (about 220 MB)
- producer: tools/run_breakthrough.py with configs/breakthrough/*.json; tables from tools/report_e2_mechanism.py and tools/report_e3_real.py
- status: unique local copy, no backup; headline numbers are copied into README.md
- notes: results/breakthrough_v1/_code/ holds patched copies of the expert1/expert2 workspaces (warm start via encoder hook) used for EP1/EP2

### breakthrough_checkpoints
- role: model checkpoints written during breakthrough runs
- location: checkpoints/breakthrough_v1/ (about 375 MB)
- status: regenerable from the plans; safe to delete
