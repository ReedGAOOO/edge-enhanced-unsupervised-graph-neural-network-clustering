# Breakthrough experiment plans

Updated 2026-10-15. The [root README](../../README.md) explains findings; [evidence/](evidence/README.md) holds compact
tracked tables. Full training outputs and checkpoints are excluded from Git.

| Plan | Question and scope | Evidence status |
|---|---|---|
| `e1_backbone_gate.json` | Does the longer training budget train the backbone? | Recorded gate runs; includes historical synthetic graphs |
| `e2d_plateau_diag.json` | Why does cold training stick on weak features? | Diagnostic sweeps |
| `e2w_warm_smoke.json` | Does pseudo-partition warm start escape the plateau? | Smoke check, not the main benchmark |
| `e2_synth_ctx_warm.json` | Edge semantics versus architecture on `synth_ctx_v1` | Main E2 evidence; EP2 collapsed and was stopped early |
| `e3_real_dual.json` | Warm/cold starts on five real graphs | Original E3 tables select each model by its own SE loss |
| `e4_b45_collapse_ablation.json` | Which B45 component causes collapse? | Component ablations on Photo/AIFB |
| `e6_b31_robust.json` | Does removing the gate preserve gains and avoid collapse? | B31 synthetic/real runs, including ACM |
| `e7_ctx_edge.json` | Can endpoint products unlock contextual edges? | Synthetic context controls |
| `e8_ctx_real.json` | Does B31 plus context help real graphs? | Real-graph context follow-up |
| `e9_fixed_ruler.json` | Do message-only learned factors improve the SE objective? | 92/92 recorded; the 48 synthetic jobs ran on an RTX 4060 Laptop |
| `e9x_machine_check.json` | How much do two seed-0 runs drift on another GPU? | 2/2 recorded; current-host launcher, no remote orchestration |
| `e11_safe_gate_pathways.json` | Does a non-renormalized gate keep B45's gains without collapse? Which B31 pathway costs Photo/AIFB? | 57/57 recorded |
| `e13_xor_context.json` | Can learned edge heads read second-order (XOR-role) contexts? | 42/42 recorded |
| `e14_search_then_train.json` | Edge-free DSE trained on the plain-spectral search picks | 42/42 recorded on the RTX 4090 server |
| `e16_search_then_train_reg.json` | The same with the regularized-spectral picks (Cora, AIFB retrained) | 12/12 recorded on the RTX 4090 server |
| `e17_edge_cv.json` | Seed-0 models of the selection benchmark retrained on graphs with 20% of edges held out | 29/29 recorded on the RTX 4090 server |
| `selection_rules_spec.json` | Not a training plan: the 42 decisions scored by `tools/select_partition_rules.py` (E18) | Evaluated; results in the root README |
| `e19a_selection_validation.json`, `e19b_selection_validation.json` | Written by `tools/run_selection_validation.py`: stage 1 (baseline starts, synthetic seed-1 runs) and stage 2 (other models, edge-CV runs) for ten new decisions | 20/20 and 24/24 recorded on the RTX 4090 server |
| `selection_validation_spec.json` | The ten E19 decisions | Evaluated with the rules frozen at `876486b` |

E5 headroom, E10 SE alignment, the B45 gate diagnosis, the E12 search, and E15 (spectral stability, regularized and beam
search) are standalone tool runs, not missing numbered plan files. Their commands are documented in
[tools/README.md](../../tools/README.md) and the root README's reproduction block.

### Search picks used by E14 and E16

The picks come from `tools/run_se_guided_baseline.py` summaries (`ctx_pick`, before the `/`), chosen without labels.
Write each with `tools/write_se_reweighted_dataset.py --dataset <source> --pick=<pick> [--ctx_dim <d>] --out_name <name>`;
use `--pick=` with an equals sign because many picks start with `-` or `+`.

| out_name | source dataset | pick | ctx_dim | search |
|---|---|---|---|---|
| `cora_sepick` | Cora | `-logdeg_gap*ctx23` | 32 | E12, plain spectral |
| `cora_sepick_reg` | Cora | `+logdeg_gap` | — | E15, regularized |
| `photo_sepick` | Photo | `+logdeg_dst` | — | rerun on the 4090 (local E12 chose `+l2*ctx1`); E15 agrees |
| `dblp_sepick` | dblp_magnn_author_v2 | `+apcpa_conference_cosine` | — | E12 and E15 |
| `aifb_sepick` | entities_aifb | `+attr0*ctx5` | 32 | E12, plain spectral |
| `aifb_sepick_reg` | entities_aifb | `+attr1*ctx2` | 32 | E15, regularized (`noctx` pick `+attr1` scores higher) |
| `acm_sepick` | acm_multiplex_v2 | `layer+PAP` | — | E12 and E15 |
| `synthctx_h{45,55}_sepick` | synth_ctx_v1_h{45,55}_contextual_ds00 | `+attr0*ctx32` | 64 | E12 and E15 |
| `synthxor_h{45,55}_sepick` | synth_ctxxor_v1_h{45,55}_contextual_ds00 | `+attr0*ctx32*ctx33` | 64 | E13/E15 `--order2` |

Plain-spectral picks on graphs with more components than clusters (Cora, Photo) can differ between machines because
plain spectral clustering depends on the eigensolver start there; E15's regularized search does not.

## Launching and resuming

```bash
# Preview the 44 E9 real jobs without training
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path data \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --dry_run

# Run that subset on the current machine
.venv/bin/python tools/run_breakthrough.py --plan configs/breakthrough/e9_fixed_ruler.json --root_path data \
  --only_tags b31fr_warm,b31fr_nowarm,b31frctx_warm,b31frctx_nowarm --gpu 0 --max_parallel 4
```

Use `--only_tags b31fr_synth,b31frctx_synth` for the 48 planned E9 synthetic jobs. Their presence in a JSON plan does
not establish completion. `--dry_run` prints commands without writing output files, even when matching runs are cached.
A run with an existing `<dataset>_metrics.json` is cached without validating the code/config
revision; use `--force` or a new plan `name` after changing either. A filtered invocation rewrites that plan's `runs.csv`
and `summary.csv` for the selected jobs, although per-run directories remain. Preserve ledgers before changing the subset.

Each run writes `stdout.txt` (first line: exact command), a training log and metrics JSON. The plan root holds
`plan.json`, `runs.csv` and `summary.csv`. Checkpoints are under `checkpoints/breakthrough_v1/...`.
`--max_parallel` controls processes on one GPU. To split a plan across machines, run the same plan with
`--num_shards N --shards i,j` on each (shard = job index mod N) and merge the per-run directories afterwards; rerunning
the plan on one machine with all runs present rebuilds `runs.csv` from the cached metrics. Four or fewer jobs is a practical starting point
for larger graphs; reduce it if memory contention develops. `--threads` caps CPU library threads per process.

## Prerequisites and interpretation

- Prepare every dataset named by the selected jobs. See [data/README.md](../../data/README.md). Urban raw city files
  are external inputs. The generator options for the documented synthetic suite are `--feature_sep 0.35 --homophily 0.45,0.55`.
- EP1/EP2 jobs require patched local expert code copies under `results/breakthrough_v1/_code/`. Those ignored copies
  are not included in a clone; archived original workspaces do not contain the same warm-start hook.
- Checkpoints are selected by training loss. `oracle_best_nmi` uses labels to choose an evaluation epoch and is retained
  for reference comparison only. It is not the reported label-free checkpoint score.
- Combined E6/E8/E9 tables choose each dataset/seed's warm/cold start using the edge-free baseline's lower
  `best_train_loss_mean`, then apply that start to every model. The report falls back to a model's own loss if the
  baseline pair is missing; inspect `dual_start_detail.csv` when reporting a new subset.
- The benchmark supplies class counts/cluster budgets. E5/E10 use labels for diagnosis and do not train the models.
- E9 `msg_only` skips learned scalar SE weighting but retains augmentation, base weights, assignment residuals and
  the edge regularizer. Its SE graph is not a frozen original adjacency.
- E14/E16 train the edge-free baseline preset on reweighted copies of the datasets; `tools/se_select_decoder.py` then
  compares the DSE assignment with spectral / SGC partitions of the same weighted graph by two-level SE. Report the DSE
  column and the SE-selected column separately; the selection is not reliable across datasets.

## Regenerating the real-graph report

With the E3/E6/E8/E9/E11 output ledgers and classic-floor CSVs present:

```bash
.venv/bin/python tools/report_e3_real.py \
  --extra_dirs results/breakthrough_v1/e6_b31_robust,results/breakthrough_v1/e8_ctx_real,results/breakthrough_v1/e9_fixed_ruler,results/breakthrough_v1/e11_safe_gate_pathways \
  --extra_classic results/breakthrough_v1/e6_b31_robust/classic_acm_v2.csv \
  --datasets cora,photo,dblp_magnn_author_v2,entities_aifb,acm_multiplex_v2 \
  --out_dir results/breakthrough_v1/combined_real
```

`--out_dir` writes `real_table.md` and `dual_start_detail.csv` separately from the source experiments. It does not change
where input CSVs are read. The E9 machine check has only two warm runs and should be kept separate from this report.
