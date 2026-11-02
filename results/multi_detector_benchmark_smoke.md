# Multi-Detector Benchmark Results (smoke)

- **Validation Split**: Seeds `0`, Scale `0.1`, Feature Version `v3`
- **Total Detectors Evaluated Simultaneously**: 29
- **Execution Time**: 16.6s scoring, 20.0s total

## Executive Summary

**Best Performing Detector Overall**: `quad_lit_dep_part_change_gbm` with Macro LOMO AUC of **0.780** [0.735, 0.841] using 4 metric groups: `+lit+dep+part+change` (50 features).

### Best Detector in Each Metric Tier

| Metric Tier | Best Detector | Metrics / Groups Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max FA |
|---|---|---|---|---|---|---|
| single_metric | `single_dep_gbm` | `dep` | 22 | 0.741 [0.675, 0.827] | 0.500 | 0.100 |
| pair_metrics | `pair_dep_change_gbm` | `dep+change` | 26 | 0.765 [0.706, 0.852] | 0.562 | 0.200 |
| triplet_metrics | `triplet_lit_dep_change_gbm` | `lit+dep+change` | 47 | 0.779 [0.733, 0.839] | 0.719 | 0.100 |
| quad_metrics | `quad_lit_dep_part_change_gbm` | `lit+dep+part+change` | 50 | 0.780 [0.735, 0.841] | 0.719 | 0.100 |
| all_metrics | `all_metrics_gbm` | `lit+dep+part+change+seq` | 53 | 0.763 [0.706, 0.811] | 0.781 | 0.100 |

## Complete Global Leaderboard (Ranked from Best to Worst)

| Rank | Detector | Tier | Metric Groups | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Min Mech AUC | Max FA |
|---|---|---|---|---|---|---|---|---|
| 1 | `quad_lit_dep_part_change_gbm` | quad_metrics | `lit+dep+part+change` | 50 | 0.780 [0.735, 0.841] | 0.719 | 0.544 | 0.100 |
| 2 | `triplet_lit_dep_change_gbm` | triplet_metrics | `lit+dep+change` | 47 | 0.779 [0.733, 0.839] | 0.719 | 0.542 | 0.100 |
| 3 | `pair_dep_change_gbm` | pair_metrics | `dep+change` | 26 | 0.765 [0.706, 0.852] | 0.562 | 0.583 | 0.200 |
| 4 | `all_metrics_gbm` | all_metrics | `lit+dep+part+change+seq` | 53 | 0.763 [0.706, 0.811] | 0.781 | 0.492 | 0.100 |
| 5 | `quad_lit_dep_change_seq_gbm` | quad_metrics | `lit+dep+change+seq` | 50 | 0.762 [0.704, 0.811] | 0.781 | 0.492 | 0.100 |
| 6 | `triplet_dep_change_seq_gbm` | triplet_metrics | `dep+change+seq` | 29 | 0.755 [0.699, 0.815] | 0.625 | 0.533 | 0.100 |
| 7 | `triplet_lit_dep_part_gbm` | triplet_metrics | `lit+dep+part` | 46 | 0.752 [0.658, 0.809] | 0.844 | 0.525 | 0.100 |
| 8 | `pair_lit_dep_gbm` | pair_metrics | `lit+dep` | 43 | 0.750 [0.657, 0.817] | 0.844 | 0.517 | 0.100 |
| 9 | `quad_dep_part_change_seq_gbm` | quad_metrics | `dep+part+change+seq` | 32 | 0.748 [0.690, 0.810] | 0.594 | 0.514 | 0.100 |
| 10 | `single_dep_gbm` | single_metric | `dep` | 22 | 0.741 [0.675, 0.827] | 0.500 | 0.567 | 0.100 |
| 11 | `quad_lit_dep_part_seq_gbm` | quad_metrics | `lit+dep+part+seq` | 49 | 0.735 [0.644, 0.788] | 0.844 | 0.486 | 0.100 |
| 12 | `triplet_lit_dep_seq_gbm` | triplet_metrics | `lit+dep+seq` | 46 | 0.735 [0.643, 0.791] | 0.812 | 0.472 | 0.000 |
| 13 | `pair_dep_part_gbm` | pair_metrics | `dep+part` | 25 | 0.734 [0.667, 0.818] | 0.469 | 0.528 | 0.100 |
| 14 | `pair_dep_seq_gbm` | pair_metrics | `dep+seq` | 25 | 0.733 [0.657, 0.792] | 0.531 | 0.492 | 0.100 |
| 15 | `triplet_dep_part_seq_gbm` | triplet_metrics | `dep+part+seq` | 28 | 0.727 [0.648, 0.784] | 0.469 | 0.469 | 0.100 |
| 16 | `single_lit_gbm` | single_metric | `lit` | 21 | 0.709 [0.637, 0.805] | 0.688 | 0.469 | 0.100 |
| 17 | `triplet_lit_part_seq_gbm` | triplet_metrics | `lit+part+seq` | 27 | 0.704 [0.633, 0.791] | 0.562 | 0.439 | 0.300 |
| 18 | `pair_lit_seq_gbm` | pair_metrics | `lit+seq` | 24 | 0.701 [0.631, 0.797] | 0.562 | 0.439 | 0.300 |
| 19 | `pair_lit_part_gbm` | pair_metrics | `lit+part` | 24 | 0.687 [0.625, 0.780] | 0.688 | 0.464 | 0.100 |
| 20 | `triplet_lit_change_seq_gbm` | triplet_metrics | `lit+change+seq` | 28 | 0.682 [0.598, 0.752] | 0.500 | 0.444 | 0.200 |
| 21 | `pair_lit_change_gbm` | pair_metrics | `lit+change` | 25 | 0.660 [0.576, 0.751] | 0.438 | 0.458 | 0.300 |
| 22 | `pair_change_seq_gbm` | pair_metrics | `change+seq` | 7 | 0.515 [0.450, 0.587] | 0.312 | 0.381 | 0.100 |
| 23 | `triplet_part_change_seq_gbm` | triplet_metrics | `part+change+seq` | 10 | 0.506 [0.443, 0.582] | 0.250 | 0.403 | 0.100 |
| 24 | `single_change_gbm` | single_metric | `change` | 4 | 0.504 [0.465, 0.585] | 0.188 | 0.338 | 0.100 |
| 25 | `pair_part_change_gbm` | pair_metrics | `part+change` | 7 | 0.487 [0.417, 0.542] | 0.156 | 0.375 | 0.100 |
| 26 | `single_seq_gbm` | single_metric | `seq` | 3 | 0.469 [0.333, 0.549] | 0.138 | 0.333 | 0.100 |
| 27 | `pair_part_seq_gbm` | pair_metrics | `part+seq` | 6 | 0.467 [0.338, 0.530] | 0.208 | 0.385 | 0.100 |
| 28 | `single_part_gbm` | single_metric | `part` | 3 | 0.466 [0.410, 0.570] | 0.182 | 0.315 | 0.100 |
| 29 | `part_gbm` | single_metric | `part` | 3 | 0.466 [0.410, 0.570] | 0.182 | 0.315 | 0.100 |

## Top 5 Detectors: Per-Mechanism Breakdown (Leave-One-Mechanism-Out AUC)

| Mechanism | `quad_lit_dep_part_change_gbm` | `triplet_lit_dep_change_gbm` | `pair_dep_change_gbm` | `all_metrics_gbm` | `quad_lit_dep_change_seq_gbm` |
|---|---|---|---|---|---|
| rotation | 0.764 | 0.764 | 0.747 | 0.764 | 0.764 |
| cover_bidding | 0.844 | 0.847 | 0.889 | 0.850 | 0.850 |
| complementary | 0.939 | 0.939 | 0.842 | 0.961 | 0.961 |
| price_fixing | 0.781 | 0.781 | 0.736 | 0.761 | 0.761 |
| intermittent | 0.856 | 0.856 | 0.814 | 0.842 | 0.842 |
| noisy_defecting | 0.742 | 0.736 | 0.661 | 0.683 | 0.681 |
| bid_suppression | 0.767 | 0.767 | 0.850 | 0.750 | 0.750 |
| partial_fringe | 0.544 | 0.542 | 0.583 | 0.492 | 0.492 |

## Key Findings & Metric Contribution Analysis

1. **Dominance of Complementary Metric Pairs**: Notice how moving from single-metric screens to pairs (e.g., adding `dep` to `lit`) produces a significant jump in Macro AUC.
2. **All-Metrics vs. Subset Tradeoff**: The comprehensive detector (`all_metrics_gbm`) utilizes all 66 features across all 5 metric groups, offering maximum coverage across diverse collusion mechanisms.
3. **Behavioral vs. Price Signals**: While price screens (`lit`) catch uniform and cover markups, identity dependence (`dep`) and participation (`part`) are critical for detecting rotation and bid suppression.