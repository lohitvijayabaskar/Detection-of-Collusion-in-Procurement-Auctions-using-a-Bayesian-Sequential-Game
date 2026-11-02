# Multi-Detector Benchmark Results (tiers_eval)

- **Validation Split**: Seeds `0-1`, Scale `0.2`, Feature Version `v3`
- **Total Detectors Evaluated Simultaneously**: 17
- **Execution Time**: 27.0s scoring, 54.0s total

## Executive Summary

**Best Performing Detector Overall**: `pair_lit_dep_gbm` with Macro LOMO AUC of **0.828** [0.772, 0.878] using 2 metric groups: `+lit+dep` (43 features).

### Best Detector in Each Metric Tier

| Metric Tier | Best Detector | Metrics / Groups Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max FA |
|---|---|---|---|---|---|---|
| single_metric | `single_dep_gbm` | `dep` | 22 | 0.788 [0.739, 0.826] | 0.547 | 0.125 |
| pair_metrics | `pair_lit_dep_gbm` | `lit+dep` | 43 | 0.828 [0.772, 0.878] | 0.742 | 0.175 |
| all_metrics | `all_metrics_gbm` | `lit+dep+part+change+seq` | 53 | 0.814 [0.771, 0.853] | 0.703 | 0.150 |

## Complete Global Leaderboard (Ranked from Best to Worst)

| Rank | Detector | Tier | Metric Groups | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Min Mech AUC | Max FA |
|---|---|---|---|---|---|---|---|---|
| 1 | `pair_lit_dep_gbm` | pair_metrics | `lit+dep` | 43 | 0.828 [0.772, 0.878] | 0.742 | 0.662 | 0.175 |
| 2 | `all_metrics_gbm` | all_metrics | `lit+dep+part+change+seq` | 53 | 0.814 [0.771, 0.853] | 0.703 | 0.618 | 0.150 |
| 3 | `single_dep_gbm` | single_metric | `dep` | 22 | 0.788 [0.739, 0.826] | 0.547 | 0.666 | 0.125 |
| 4 | `pair_dep_seq_gbm` | pair_metrics | `dep+seq` | 25 | 0.786 [0.744, 0.823] | 0.523 | 0.657 | 0.175 |
| 5 | `pair_dep_part_gbm` | pair_metrics | `dep+part` | 25 | 0.782 [0.733, 0.823] | 0.500 | 0.677 | 0.150 |
| 6 | `single_lit_gbm` | single_metric | `lit` | 21 | 0.779 [0.706, 0.830] | 0.672 | 0.654 | 0.075 |
| 7 | `pair_dep_change_gbm` | pair_metrics | `dep+change` | 26 | 0.772 [0.731, 0.814] | 0.555 | 0.638 | 0.175 |
| 8 | `pair_lit_seq_gbm` | pair_metrics | `lit+seq` | 24 | 0.771 [0.727, 0.806] | 0.633 | 0.647 | 0.075 |
| 9 | `pair_lit_part_gbm` | pair_metrics | `lit+part` | 24 | 0.767 [0.700, 0.826] | 0.633 | 0.608 | 0.050 |
| 10 | `pair_lit_change_gbm` | pair_metrics | `lit+change` | 25 | 0.758 [0.702, 0.800] | 0.688 | 0.631 | 0.100 |
| 11 | `pair_part_seq_gbm` | pair_metrics | `part+seq` | 6 | 0.584 [0.530, 0.658] | 0.273 | 0.504 | 0.075 |
| 12 | `single_part_gbm` | single_metric | `part` | 3 | 0.562 [0.481, 0.646] | 0.297 | 0.396 | 0.025 |
| 13 | `part_gbm` | single_metric | `part` | 3 | 0.562 [0.481, 0.646] | 0.297 | 0.396 | 0.025 |
| 14 | `single_seq_gbm` | single_metric | `seq` | 3 | 0.557 [0.500, 0.616] | 0.255 | 0.475 | 0.025 |
| 15 | `pair_change_seq_gbm` | pair_metrics | `change+seq` | 7 | 0.519 [0.456, 0.577] | 0.195 | 0.453 | 0.100 |
| 16 | `pair_part_change_gbm` | pair_metrics | `part+change` | 7 | 0.501 [0.430, 0.576] | 0.180 | 0.372 | 0.025 |
| 17 | `single_change_gbm` | single_metric | `change` | 4 | 0.479 [0.420, 0.543] | 0.141 | 0.408 | 0.075 |

## Top 5 Detectors: Per-Mechanism Breakdown (Leave-One-Mechanism-Out AUC)

| Mechanism | `pair_lit_dep_gbm` | `all_metrics_gbm` | `single_dep_gbm` | `pair_dep_seq_gbm` | `pair_dep_part_gbm` |
|---|---|---|---|---|---|
| rotation | 0.894 | 0.898 | 0.801 | 0.806 | 0.796 |
| cover_bidding | 0.917 | 0.893 | 0.883 | 0.874 | 0.889 |
| complementary | 0.848 | 0.857 | 0.839 | 0.838 | 0.854 |
| price_fixing | 0.943 | 0.905 | 0.914 | 0.914 | 0.904 |
| intermittent | 0.843 | 0.836 | 0.759 | 0.779 | 0.768 |
| noisy_defecting | 0.736 | 0.765 | 0.666 | 0.676 | 0.677 |
| bid_suppression | 0.776 | 0.738 | 0.736 | 0.747 | 0.684 |
| partial_fringe | 0.662 | 0.618 | 0.703 | 0.657 | 0.683 |

## Key Findings & Metric Contribution Analysis

1. **Dominance of Complementary Metric Pairs**: Notice how moving from single-metric screens to pairs (e.g., adding `dep` to `lit`) produces a significant jump in Macro AUC.
2. **All-Metrics vs. Subset Tradeoff**: The comprehensive detector (`all_metrics_gbm`) utilizes all 66 features across all 5 metric groups, offering maximum coverage across diverse collusion mechanisms.
3. **Behavioral vs. Price Signals**: While price screens (`lit`) catch uniform and cover markups, identity dependence (`dep`) and participation (`part`) are critical for detecting rotation and bid suppression.