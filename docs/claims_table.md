# Claims table

All numbers are from LOCKED run 1 (`results/locked_1_full_gbm.json`, design full_gbm, hash
6cc622540eac3929, seeds 0-9, two-level bootstrap 95 % CI, 1000 resamples) unless marked
"validation". Synthetic data only. Which of these become paper claims is the user's decision
(none chosen yet).

| # | Candidate claim | Status | Evidence |
|---|---|---|---|
| 1 | Same seed gives identical features across processes and worker counts | supported | `scripts/check_determinism.py` (E0) |
| 2 | The identity-dependence detector beats the literature-screen GBM on held-out mechanisms | supported | macro LOMO AUC 0.870 [0.862, 0.878] vs 0.803 [0.792, 0.812]; paired difference +0.067 [0.060, 0.074] |
| 3 | It beats every single literature screen (CV, skewness, kurtosis, RD, RD-alt, gap, spread) | supported | paired differences +0.196 to +0.382, all CIs above 0 |
| 4 | Held-out rotation, price fixing, complementary bidding and cover bidding are detected with AUC ≥ 0.85, including unseen sizes and spreads | supported | worst condition: 0.939 / 0.963 / 0.947 / 0.921 |
| 5 | False alarms stay ≤ 10 % on every honest test set (unseen sizes and spreads, high markup, capacity effects, noisy estimates) | supported for the all-mechanism model | max 0.075 [0.050, 0.105] (sizes); the LOMO models reach 0.116 on sizes |
| 6 | Precision@10 % exceeds the base rate (0.167) for every held-out mechanism | supported | lowest 0.378 [0.335, 0.418] (partial_fringe) |
| 7 | All eight held-out mechanisms reach AUC ≥ 0.85 (criterion C1) | NOT supported | partial_fringe 0.672, bid_suppression 0.744, noisy_defecting 0.823, intermittent 0.849 at unseen spreads |
| 8 | Uniform-markup collusion is not identifiable from bids alone | consistent (negative control) | AUC 0.478 [0.450, 0.503], alarm 0.047 |
| 9 | Within-market change scans and the sequential layer (CUSUM, HMM) add detection power | NOT supported (validation) | change_gbm 0.494, seq_gbm 0.549; dropping them changes full_gbm by ≤ 0.002 (V1, I1) |
| 10 | The within-firm time-shuffle null improves the dependence features | supported (validation) | dep_gbm 0.771 → 0.826; full_gbm 0.847 → 0.862 (I1) |
| 11 | A knowledge-directed tail score, participation features or monotone constraints help held-out mechanisms | NOT supported (validation) | I2 and I4: macro changes ≤ +0.005 within CI; mixed per mechanism |

## Limitations (state in any write-up)

- Synthetic markets only; no labelled per-bid real dataset was available.
- Small cartels in a large competitive fringe (2-3 firms), bid suppression (when held out) and noisy/defecting cartels stay below AUC 0.85.
- Detection of bid suppression under LOMO relies on signals the other mechanisms share; its own signal (missing bids) is not learned when it is held out.
- The detector is a market-level score; it does not identify cartel members.
- Uniform-markup collusion is out of scope (identification limit without cost data).
- Cartels are imposed, not equilibrium behaviour; mechanism parameter ranges are design choices.
