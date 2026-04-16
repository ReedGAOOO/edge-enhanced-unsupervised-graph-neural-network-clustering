# Stage Results Hub

This folder centralizes key stage summaries extracted from `results/`.
Each stage keeps raw CSV and, when needed, `*_normalized.csv` with ECHF names.

## Stage overview

| Stage | Focus | Source tag | Top line | Key takeaway |
|---|---|---|---|---|
| `stage1_branch_compare_fixed_losssel` | Re-run after fixing selection bias | `benchmark_branch_compare_v2_fixed_losssel` | `B4_V5_mid_adapt`, rank `0.2170` | Plain V5 middle-fusion is still strong under strict loss-based selection. |
| `stage2_pathab_top4_confirm180` | Path-A/Path-B top4 confirm | `benchmark_pathAB_top4_confirm180_v1` | `G17_V5_temp15` rank `0.3073`, `G15_ECHF_main` rank `0.2859` | G17 is strongest on classic sets; G15 is close and more structural on hetero graphs. |
| `stage3_b15_mainline_ablation` | B15-centered branch ablation | `benchmark_b15_mainline_ablation_v1` | `B5_V5_mid_no_adapt` rank `0.2753`, `B15_ECHF_main` rank `0.2581` | B15 is not always top NMI, but remains robust with better Path-B semantics. |
| `stage4_sched60_vs100_fair` | Direct `sched=60` vs `sched=100` | `benchmark_sched60_vs100_v2_fair` | mean NMI: `100=0.2193`, `60=0.2109` | 100 wins on average due larger gains on photo/computers. |
| `stage5_urban_known_struct_e60` | Known-only urban benchmark (multi-seed) | `urban_known_struct_g15g17_e60_v1` | `G15_ECHF_main` > `G17_V5_temp15` | G15 wins 12/13 cities in NMI/ARI and 13/13 in modularity/conductance. |
| `stage6_urban_g15_g17_detailed` | Fine-grained urban contrast set | `urban_g15_g17_detailed_v1` | `G15_ECHF_main` slight lead by mean NMI | Gap is small but consistently favors G15-family on real urban graphs. |
| `stage7_urban_sched100_vs60_quick_s0` | Urban quick check (`seed=0`) | `urban_b15_sched60_vs_g15_sched100_e60_quick_s0_v1` | mean NMI difference is tiny (`~4e-5`) | City-level winners split, showing schedule sensitivity is dataset-specific. |
| `stage8_urban_e20_vs_e60` | Training sufficiency (`20` vs `60`) | `urban_known_struct_g15g17_e20e60_compare_v1` | both improve from 20 to 60 | 60 epochs is materially more reliable for urban structural metrics. |
| `stage9_echf_smoke_v1` | Post-refactor smoke benchmark | `benchmark_echf_smoke_v1` | `B15_ECHF_s60` rank `0.2598`, `B15_ECHF_main` rank `0.2588` | New naming pipeline runs correctly and both ECHF variants beat baseline on mean NMI/ARI. |
| `stage10_echf_smoke_v2_s012` | Post-refactor multi-seed smoke | `benchmark_echf_smoke_v2_s012` | `B15_ECHF_s60` rank `0.2571`, `B15_ECHF_main` rank `0.2560` | Across seeds `0/1/2`, both ECHF variants still beat baseline; `s60` remains slightly ahead. |
| `stage11_urban_echf_quick_e20_s0` | Urban quick recheck after refactor | `urban_echf_quick_recheck_e20_s0_v1` | mean NMI `G15=0.012442`, `G17=0.012407` | E20 seed0 quick check is near-tie; `G15` wins 2/3 cities by NMI. |
| `stage12_urban_echf_recheck_e60_s01_partial` | Urban semi-full recheck (partial) | `urban_echf_recheck_e60_s01_v1` | completed `5/52` runs | Partial paired subset (`bangkok` s0/s1) supports `G15 > G17`; use stage5 as primary full evidence. |
| `stage13_urban_echf_recheck_e60_s01_full` | Urban E60 full recheck (resume completed) | `urban_echf_recheck_e60_s01_v1` | completed `52/52` runs | Full 13-city x 2-seed rerun confirms `G15 > G17` (`11/13` wins in NMI and ARI). |

## Recommended interpretation order
1. `stage1` and `stage2`: identify branch behavior on classic benchmarks.
2. `stage3` and `stage4`: understand why B15/G15 was chosen as robust mainline.
3. `stage5` to `stage8`: verify behavior on real urban graphs and training sufficiency.
4. `stage9`: validate that refactored ECHF naming and scripts reproduce expected gains.
5. `stage10` and `stage11`: multi-seed smoke and quick urban post-refactor sanity checks.
6. `stage12`: interrupted long-run subset; consistency check only.
7. `stage13`: completed long-run urban recheck; post-refactor primary evidence.

## Naming normalization
Use `*_normalized.csv` if you want canonical ECHF names:
- `B15_ECHF_main` (`B15_PATHB_v12_hier`)
- `B15_ECHF_s60` (`B15_PATHB_v12_hier_sched60`)
- `G15_ECHF_main` (`G15_default_hetero`)
- `G15_ECHF_noadapt` (`G15_noadapt_hetero`)
- `G17_V5_temp15` (`G17_temp1p5_mainline`)
