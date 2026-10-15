# Compact evidence snapshot

Exported 2026-10-15 from local `results/breakthrough_v1/` outputs based on source revision `8d77f9e` plus the E9
`msg_only` implementation and diagnostic tools committed alongside this snapshot. These small tables make the
new README findings inspectable on GitHub. They are not full logs, checkpoints or a replacement for rerunning the plans.

| File | Source and protocol |
|---|---|
| `real_selected.csv` | Chosen rows of `report_e3_real.py`'s `dual_start_detail.csv`, merging E3/E6/E8/E9; five real datasets, baseline-guided starts, six models |
| `machine_check.csv` | Matching seed-0 `b31fr_warm` rows from E9 and E9x ledgers; two datasets, no pooling with the main benchmark |
| `se_alignment.csv` | Direct copy of `e10_se_alignment/se_alignment.csv`; finite candidate-weighting screen using labels for diagnosis |
| `b45_gate_interventions.csv` | Final SE-epoch rows of `b45_gate_diagnosis/summary.json`; Photo seed 0, 200 warm + 200 SE epochs, fixed final decode seed 999 |
| `b45_graph_cross.csv` | Same summary's fixed-checkpoint graph-cross rows; leaf argmax assignments, distinct from the full-decode intervention protocol |
| `b45_graph_stats.csv` | Selected graph statistics from that summary; normalized weight shares are per gate layer |

`real_selected.csv` records individual training seeds; NMI is already averaged over three decode trials within each run.
README `±` values use the population standard deviation across these seed-level means. Photo/AIFB have one training
seed; Cora/DBLP/ACM have three. All selected starts are those chosen by the matching edge-free baseline pair's lower
`best_train_loss_mean`; no label metric selects a start. Class counts and hierarchy budgets are benchmark inputs.

The graph-cross probes use one checkpoint per dataset and do not report a full training-seed distribution. The Photo
`safe_isolates` intervention was recorded in a separate paired run under the same seed/initialization protocol. The
DBLP paired-training attempt was interrupted for resource contention and is excluded from the intervention table.
Diagnostic fixes remain runtime hooks, not production model changes.

E9's ledger contains 44 real runs from its 92-job plan. The other 48 jobs are synthetic and have no recorded results.
E5 and E1–E8 headline evidence remains summarized in the root README; this snapshot focuses on the new E9/E10 and
B45 follow-up claims. Full raw outputs are local, ignored assets described in [DATA_ASSETS.md](../../../DATA_ASSETS.md).

To refresh the real table, run the report command in [the plan index](../README.md), select rows where `chosen` is
`True`, and retain the five datasets above. Refresh the other tables from their stated source CSV/JSON files while
preserving the protocol distinctions. Record a new snapshot date and source revision when results change.
