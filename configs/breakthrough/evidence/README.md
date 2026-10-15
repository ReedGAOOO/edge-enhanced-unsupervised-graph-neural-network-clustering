# Compact evidence snapshot

First exported 2026-10-15 from local `results/breakthrough_v1/` outputs (source revision `8d77f9e` plus the E9
`msg_only` implementation); refreshed the same evening for E9 (synthetic), E11-E16 at revision `2363246`. These small tables make the
new README findings inspectable on GitHub. They are not full logs, checkpoints or a replacement for rerunning the plans.

| File | Source and protocol |
|---|---|
| `real_selected.csv` | Chosen rows of `report_e3_real.py`'s `dual_start_detail.csv`, merging E3/E6/E8/E9/E11; five real datasets, baseline-guided starts, nine models |
| `synth_mechanism.csv` | `report_e2_mechanism.py` table merging E2/E6/E7/E9/E11 on `synth_ctx_v1` (arch / descriptor / direct / context effects) |
| `xor_context_learned.csv` | E13 `summary.csv`: learned heads and the edge-free baseline on `synth_ctxxor_v1` |
| `se_search_plain.csv` | E12/E13 search summaries with plain spectral (real seed 0, AIFB seeds 1-4, synthetic seeds 0-2, XOR first and second order) and the 4090 Photo rerun |
| `se_search_reg_beam.csv` | E15 search summaries with regularized spectral and a width-8 order-2 beam (real seeds 0-2; synthetic seed 0 with `--order2`) |
| `spectral_stability.csv` | E15: plain vs regularized spectral NMI over five eigensolver starts, uniform graphs |
| `selection_rules_summary.csv` | E18 summary of `tools/select_partition_rules.py` per rule and decision set (`cv-subset` = seed-0 decisions with edge-CV runs); R4 was defined after E18 |
| `selection_rules_instances.csv` | E18 per-decision picks and NMI for every rule, the best candidate and the feature-gate statistic |
| `search_then_train.csv` | E14 (plain) and E16 (regularized) decoder choice per run, with each run's tag and training loss for the start rule |
| `machine_check.csv` | Matching seed-0 `b31fr_warm` rows from E9 and E9x ledgers; two datasets, no pooling with the main benchmark |
| `se_alignment.csv` | Direct copy of `e10_se_alignment/se_alignment.csv`; finite candidate-weighting screen using labels for diagnosis |
| `b45_gate_interventions.csv` | Final SE-epoch rows of `b45_gate_diagnosis/summary.json`; Photo seed 0, 200 warm + 200 SE epochs, fixed final decode seed 999 |
| `b45_graph_cross.csv` | Same summary's fixed-checkpoint graph-cross rows; leaf argmax assignments, distinct from the full-decode intervention protocol |
| `b45_graph_stats.csv` | Selected graph statistics from that summary; normalized weight shares are per gate layer |

`real_selected.csv` records individual training seeds; NMI is already averaged over three decode trials within each run.
README `±` values use the population standard deviation across these seed-level means. Most models have three training
seeds; B31+ctx, B31+ctx msg_only, B31 msg_only and B45 have one seed on Photo/AIFB. All selected starts are those chosen by the matching edge-free baseline pair's lower
`best_train_loss_mean`; no label metric selects a start. Class counts and hierarchy budgets are benchmark inputs.

The graph-cross probes use one checkpoint per dataset and do not report a full training-seed distribution. The Photo
`safe_isolates` intervention was recorded in a separate paired run under the same seed/initialization protocol. The
DBLP paired-training attempt was interrupted for resource contention and is excluded from the intervention table.
The normalization fix was later adopted as `--no-edge_msg_gate_renorm` and validated at full budget in E11; the other
hooks remain diagnostics.

E9's 92 runs are complete (the 48 synthetic runs were trained on an RTX 4060 Laptop). E14/E16 training ran on an
RTX 4090 server; the decoder rows use `tools/se_select_decoder.py`. For the search-then-train table, take each
dataset/seed's start with the lower `best_train_loss` (synthetic runs have one start). E5 and E1–E8 headline evidence
remains summarized in the root README. Full raw outputs are local, ignored assets described in [DATA_ASSETS.md](../../../DATA_ASSETS.md).

To refresh the real table, run the report command in [the plan index](../README.md), select rows where `chosen` is
`True`, and retain the five datasets above. Refresh the other tables from their stated source CSV/JSON files while
preserving the protocol distinctions. Record a new snapshot date and source revision when results change.
