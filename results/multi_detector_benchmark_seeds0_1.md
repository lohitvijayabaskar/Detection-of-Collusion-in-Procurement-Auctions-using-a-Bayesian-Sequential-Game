# Multi-Detector Benchmark Results (seeds0_1)

- **Validation Split**: Seeds `0-1`, Scale `0.2`, Feature Version `v3`
- **Total Detectors Evaluated Simultaneously**: 29
- **Execution Time**: 82.1s scoring, 97.9s total

## Executive Summary

**Best Performing Detector Overall**: `pair_lit_dep_gbm` with Macro LOMO AUC of **0.828** [0.770, 0.877] using 2 metric groups: `+lit+dep` (43 features).

### Best Detector in Each Metric Tier

| Metric Tier | Best Detector | Metrics / Groups Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max FA |
|---|---|---|---|---|---|---|
| single_metric | `single_dep_gbm` | `dep` | 22 | 0.788 [0.748, 0.822] | 0.547 | 0.125 |
| pair_metrics | `pair_lit_dep_gbm` | `lit+dep` | 43 | 0.828 [0.770, 0.877] | 0.742 | 0.175 |
| triplet_metrics | `triplet_lit_dep_seq_gbm` | `lit+dep+seq` | 46 | 0.827 [0.779, 0.870] | 0.750 | 0.125 |
| quad_metrics | `quad_lit_dep_part_seq_gbm` | `lit+dep+part+seq` | 49 | 0.823 [0.773, 0.868] | 0.727 | 0.125 |
| all_metrics | `all_metrics_gbm` | `lit+dep+part+change+seq` | 53 | 0.814 [0.770, 0.852] | 0.703 | 0.150 |

## Complete Global Leaderboard (Ranked from Best to Worst)

| Rank | Detector | Tier | Metric Groups | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Min Mech AUC | Max FA |
|---|---|---|---|---|---|---|---|---|
| 1 | `pair_lit_dep_gbm` | pair_metrics | `lit+dep` | 43 | 0.828 [0.770, 0.877] | 0.742 | 0.662 | 0.175 |
| 2 | `triplet_lit_dep_seq_gbm` | triplet_metrics | `lit+dep+seq` | 46 | 0.827 [0.779, 0.870] | 0.750 | 0.658 | 0.125 |
| 3 | `triplet_lit_dep_part_gbm` | triplet_metrics | `lit+dep+part` | 46 | 0.823 [0.764, 0.875] | 0.727 | 0.652 | 0.150 |
| 4 | `quad_lit_dep_part_seq_gbm` | quad_metrics | `lit+dep+part+seq` | 49 | 0.823 [0.773, 0.868] | 0.727 | 0.637 | 0.125 |
| 5 | `quad_lit_dep_change_seq_gbm` | quad_metrics | `lit+dep+change+seq` | 50 | 0.819 [0.777, 0.854] | 0.695 | 0.621 | 0.175 |
| 6 | `triplet_lit_dep_change_gbm` | triplet_metrics | `lit+dep+change` | 47 | 0.815 [0.770, 0.851] | 0.734 | 0.629 | 0.100 |
| 7 | `all_metrics_gbm` | all_metrics | `lit+dep+part+change+seq` | 53 | 0.814 [0.770, 0.852] | 0.703 | 0.618 | 0.150 |
| 8 | `quad_lit_dep_part_change_gbm` | quad_metrics | `lit+dep+part+change` | 50 | 0.811 [0.768, 0.847] | 0.648 | 0.626 | 0.150 |
| 9 | `single_dep_gbm` | single_metric | `dep` | 22 | 0.788 [0.748, 0.822] | 0.547 | 0.666 | 0.125 |
| 10 | `pair_dep_seq_gbm` | pair_metrics | `dep+seq` | 25 | 0.786 [0.745, 0.825] | 0.523 | 0.657 | 0.175 |
| 11 | `pair_dep_part_gbm` | pair_metrics | `dep+part` | 25 | 0.782 [0.738, 0.820] | 0.500 | 0.677 | 0.150 |
| 12 | `single_lit_gbm` | single_metric | `lit` | 21 | 0.779 [0.699, 0.825] | 0.672 | 0.654 | 0.075 |
| 13 | `pair_dep_change_gbm` | pair_metrics | `dep+change` | 26 | 0.772 [0.735, 0.819] | 0.555 | 0.638 | 0.175 |
| 14 | `pair_lit_seq_gbm` | pair_metrics | `lit+seq` | 24 | 0.771 [0.724, 0.803] | 0.633 | 0.647 | 0.075 |
| 15 | `triplet_dep_part_seq_gbm` | triplet_metrics | `dep+part+seq` | 28 | 0.769 [0.715, 0.815] | 0.477 | 0.644 | 0.150 |
| 16 | `triplet_dep_change_seq_gbm` | triplet_metrics | `dep+change+seq` | 29 | 0.768 [0.725, 0.814] | 0.500 | 0.622 | 0.225 |
| 17 | `pair_lit_part_gbm` | pair_metrics | `lit+part` | 24 | 0.767 [0.681, 0.823] | 0.633 | 0.608 | 0.050 |
| 18 | `triplet_lit_change_seq_gbm` | triplet_metrics | `lit+change+seq` | 28 | 0.760 [0.705, 0.796] | 0.633 | 0.628 | 0.050 |
| 19 | `triplet_lit_part_seq_gbm` | triplet_metrics | `lit+part+seq` | 27 | 0.759 [0.707, 0.798] | 0.609 | 0.629 | 0.100 |
| 20 | `quad_dep_part_change_seq_gbm` | quad_metrics | `dep+part+change+seq` | 32 | 0.759 [0.713, 0.805] | 0.508 | 0.614 | 0.175 |
| 21 | `pair_lit_change_gbm` | pair_metrics | `lit+change` | 25 | 0.758 [0.685, 0.799] | 0.688 | 0.631 | 0.100 |
| 22 | `pair_part_seq_gbm` | pair_metrics | `part+seq` | 6 | 0.584 [0.520, 0.655] | 0.273 | 0.504 | 0.075 |
| 23 | `single_part_gbm` | single_metric | `part` | 3 | 0.562 [0.480, 0.632] | 0.297 | 0.396 | 0.025 |
| 24 | `part_gbm` | single_metric | `part` | 3 | 0.562 [0.480, 0.632] | 0.297 | 0.396 | 0.025 |
| 25 | `single_seq_gbm` | single_metric | `seq` | 3 | 0.557 [0.487, 0.623] | 0.255 | 0.475 | 0.025 |
| 26 | `triplet_part_change_seq_gbm` | triplet_metrics | `part+change+seq` | 10 | 0.535 [0.476, 0.584] | 0.219 | 0.454 | 0.050 |
| 27 | `pair_change_seq_gbm` | pair_metrics | `change+seq` | 7 | 0.519 [0.448, 0.578] | 0.195 | 0.453 | 0.100 |
| 28 | `pair_part_change_gbm` | pair_metrics | `part+change` | 7 | 0.501 [0.431, 0.562] | 0.180 | 0.372 | 0.025 |
| 29 | `single_change_gbm` | single_metric | `change` | 4 | 0.479 [0.421, 0.522] | 0.141 | 0.408 | 0.075 |

## Top 5 Detectors: Per-Mechanism Breakdown (Leave-One-Mechanism-Out AUC)

| Mechanism | `pair_lit_dep_gbm` | `triplet_lit_dep_seq_gbm` | `triplet_lit_dep_part_gbm` | `quad_lit_dep_part_seq_gbm` | `quad_lit_dep_change_seq_gbm` |
|---|---|---|---|---|---|
| rotation | 0.894 | 0.888 | 0.894 | 0.886 | 0.896 |
| cover_bidding | 0.917 | 0.919 | 0.912 | 0.916 | 0.907 |
| complementary | 0.848 | 0.854 | 0.859 | 0.869 | 0.861 |
| price_fixing | 0.943 | 0.947 | 0.939 | 0.942 | 0.917 |
| intermittent | 0.843 | 0.814 | 0.832 | 0.819 | 0.834 |
| noisy_defecting | 0.736 | 0.745 | 0.741 | 0.754 | 0.746 |
| bid_suppression | 0.776 | 0.788 | 0.758 | 0.763 | 0.768 |
| partial_fringe | 0.662 | 0.658 | 0.652 | 0.637 | 0.621 |

## Key Findings & Metric Contribution Analysis

1. **Dominance of Complementary Metric Pairs**: Notice how moving from single-metric screens to pairs (e.g., adding `dep` to `lit`) produces a significant jump in Macro AUC.
2. **All-Metrics vs. Subset Tradeoff**: The comprehensive detector (`all_metrics_gbm`) utilizes all 66 features across all 5 metric groups, offering maximum coverage across diverse collusion mechanisms.
3. **Behavioral vs. Price Signals**: While price screens (`lit`) catch uniform and cover markups, identity dependence (`dep`) and participation (`part`) are critical for detecting rotation and bid suppression.