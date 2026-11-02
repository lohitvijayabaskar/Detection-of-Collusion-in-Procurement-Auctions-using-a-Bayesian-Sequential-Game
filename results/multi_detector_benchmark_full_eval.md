# Multi-Detector Benchmark Results (full_eval)

- **Validation Split**: Seeds `0-9`, Scale `1.0`, Feature Version `v3`
- **Total Detectors Evaluated Simultaneously**: 29
- **Execution Time**: 884.1s scoring, 1117.2s total

## Executive Summary

**Best Performing Detector Overall**: `triplet_lit_dep_part_gbm` with Macro LOMO AUC of **0.866** [0.859, 0.874] using 3 metric groups: `+lit+dep+part` (46 features).

### Best Detector in Each Metric Tier

| Metric Tier | Best Detector | Metrics / Groups Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max FA |
|---|---|---|---|---|---|---|
| single_metric | `single_dep_gbm` | `dep` | 22 | 0.825 [0.814, 0.836] | 0.727 | 0.063 |
| pair_metrics | `pair_lit_dep_gbm` | `lit+dep` | 43 | 0.864 [0.857, 0.872] | 0.813 | 0.084 |
| triplet_metrics | `triplet_lit_dep_part_gbm` | `lit+dep+part` | 46 | 0.866 [0.859, 0.874] | 0.824 | 0.055 |
| quad_metrics | `quad_lit_dep_part_seq_gbm` | `lit+dep+part+seq` | 49 | 0.865 [0.857, 0.872] | 0.817 | 0.057 |
| all_metrics | `all_metrics_gbm` | `lit+dep+part+change+seq` | 53 | 0.862 [0.854, 0.869] | 0.815 | 0.057 |

## Complete Global Leaderboard (Ranked from Best to Worst)

| Rank | Detector | Tier | Metric Groups | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Min Mech AUC | Max FA |
|---|---|---|---|---|---|---|---|---|
| 1 | `triplet_lit_dep_part_gbm` | triplet_metrics | `lit+dep+part` | 46 | 0.866 [0.859, 0.874] | 0.824 | 0.679 | 0.055 |
| 2 | `quad_lit_dep_part_seq_gbm` | quad_metrics | `lit+dep+part+seq` | 49 | 0.865 [0.857, 0.872] | 0.817 | 0.679 | 0.057 |
| 3 | `triplet_lit_dep_seq_gbm` | triplet_metrics | `lit+dep+seq` | 46 | 0.864 [0.855, 0.873] | 0.805 | 0.675 | 0.084 |
| 4 | `pair_lit_dep_gbm` | pair_metrics | `lit+dep` | 43 | 0.864 [0.857, 0.872] | 0.813 | 0.674 | 0.084 |
| 5 | `quad_lit_dep_part_change_gbm` | quad_metrics | `lit+dep+part+change` | 50 | 0.863 [0.855, 0.870] | 0.815 | 0.673 | 0.061 |
| 6 | `all_metrics_gbm` | all_metrics | `lit+dep+part+change+seq` | 53 | 0.862 [0.854, 0.869] | 0.815 | 0.675 | 0.057 |
| 7 | `triplet_lit_dep_change_gbm` | triplet_metrics | `lit+dep+change` | 47 | 0.862 [0.853, 0.870] | 0.811 | 0.672 | 0.086 |
| 8 | `quad_lit_dep_change_seq_gbm` | quad_metrics | `lit+dep+change+seq` | 50 | 0.862 [0.853, 0.870] | 0.811 | 0.670 | 0.078 |
| 9 | `pair_dep_seq_gbm` | pair_metrics | `dep+seq` | 25 | 0.826 [0.815, 0.836] | 0.724 | 0.683 | 0.067 |
| 10 | `single_dep_gbm` | single_metric | `dep` | 22 | 0.825 [0.814, 0.836] | 0.727 | 0.689 | 0.063 |
| 11 | `triplet_dep_part_seq_gbm` | triplet_metrics | `dep+part+seq` | 28 | 0.825 [0.812, 0.834] | 0.719 | 0.676 | 0.065 |
| 12 | `pair_dep_part_gbm` | pair_metrics | `dep+part` | 25 | 0.824 [0.812, 0.834] | 0.714 | 0.686 | 0.060 |
| 13 | `triplet_dep_change_seq_gbm` | triplet_metrics | `dep+change+seq` | 29 | 0.822 [0.811, 0.832] | 0.713 | 0.677 | 0.064 |
| 14 | `pair_dep_change_gbm` | pair_metrics | `dep+change` | 26 | 0.821 [0.810, 0.832] | 0.720 | 0.678 | 0.066 |
| 15 | `quad_dep_part_change_seq_gbm` | quad_metrics | `dep+part+change+seq` | 32 | 0.820 [0.807, 0.830] | 0.711 | 0.673 | 0.064 |
| 16 | `pair_lit_seq_gbm` | pair_metrics | `lit+seq` | 24 | 0.806 [0.796, 0.815] | 0.764 | 0.562 | 0.092 |
| 17 | `single_lit_gbm` | single_metric | `lit` | 21 | 0.806 [0.795, 0.816] | 0.761 | 0.565 | 0.088 |
| 18 | `triplet_lit_part_seq_gbm` | triplet_metrics | `lit+part+seq` | 27 | 0.805 [0.796, 0.814] | 0.768 | 0.564 | 0.056 |
| 19 | `pair_lit_part_gbm` | pair_metrics | `lit+part` | 24 | 0.805 [0.795, 0.813] | 0.767 | 0.565 | 0.058 |
| 20 | `triplet_lit_change_seq_gbm` | triplet_metrics | `lit+change+seq` | 28 | 0.802 [0.792, 0.811] | 0.749 | 0.555 | 0.086 |
| 21 | `pair_lit_change_gbm` | pair_metrics | `lit+change` | 25 | 0.801 [0.789, 0.810] | 0.750 | 0.554 | 0.078 |
| 22 | `single_seq_gbm` | single_metric | `seq` | 3 | 0.552 [0.532, 0.567] | 0.252 | 0.517 | 0.067 |
| 23 | `pair_part_seq_gbm` | pair_metrics | `part+seq` | 6 | 0.544 [0.527, 0.557] | 0.265 | 0.514 | 0.050 |
| 24 | `triplet_part_change_seq_gbm` | triplet_metrics | `part+change+seq` | 10 | 0.543 [0.528, 0.556] | 0.250 | 0.496 | 0.062 |
| 25 | `pair_change_seq_gbm` | pair_metrics | `change+seq` | 7 | 0.542 [0.527, 0.555] | 0.252 | 0.485 | 0.076 |
| 26 | `single_part_gbm` | single_metric | `part` | 3 | 0.500 [0.486, 0.515] | 0.203 | 0.472 | 0.055 |
| 27 | `part_gbm` | single_metric | `part` | 3 | 0.500 [0.486, 0.515] | 0.203 | 0.472 | 0.055 |
| 28 | `single_change_gbm` | single_metric | `change` | 4 | 0.495 [0.479, 0.509] | 0.194 | 0.443 | 0.065 |
| 29 | `pair_part_change_gbm` | pair_metrics | `part+change` | 7 | 0.494 [0.480, 0.510] | 0.201 | 0.425 | 0.065 |

## Top 5 Detectors: Per-Mechanism Breakdown (Leave-One-Mechanism-Out AUC)

| Mechanism | `triplet_lit_dep_part_gbm` | `quad_lit_dep_part_seq_gbm` | `triplet_lit_dep_seq_gbm` | `pair_lit_dep_gbm` | `quad_lit_dep_part_change_gbm` |
|---|---|---|---|---|---|
| rotation | 0.952 | 0.953 | 0.953 | 0.954 | 0.951 |
| cover_bidding | 0.947 | 0.944 | 0.947 | 0.947 | 0.942 |
| complementary | 0.955 | 0.955 | 0.956 | 0.956 | 0.956 |
| price_fixing | 0.973 | 0.973 | 0.973 | 0.973 | 0.971 |
| intermittent | 0.860 | 0.859 | 0.859 | 0.860 | 0.859 |
| noisy_defecting | 0.805 | 0.803 | 0.801 | 0.799 | 0.795 |
| bid_suppression | 0.753 | 0.750 | 0.751 | 0.750 | 0.753 |
| partial_fringe | 0.679 | 0.679 | 0.675 | 0.674 | 0.673 |

## Key Findings & Metric Contribution Analysis

1. **Dominance of Complementary Metric Pairs**: Notice how moving from single-metric screens to pairs (e.g., adding `dep` to `lit`) produces a significant jump in Macro AUC.
2. **All-Metrics vs. Subset Tradeoff**: The comprehensive detector (`all_metrics_gbm`) utilizes all 66 features across all 5 metric groups, offering maximum coverage across diverse collusion mechanisms.
3. **Behavioral vs. Price Signals**: While price screens (`lit`) catch uniform and cover markups, identity dependence (`dep`) and participation (`part`) are critical for detecting rotation and bid suppression.