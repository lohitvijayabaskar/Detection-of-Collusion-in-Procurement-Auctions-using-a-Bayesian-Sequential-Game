# Real-data evaluation: proposed frozen design (not yet run)

Status: PROPOSED, waiting for the user's approval. Nothing below has been run on the real
data. Code: `bidrig/realeval.py`, runner `scripts/run_real_lodo.py`. Design hash at
proposal: `1cec4feb56110f6c`, covering `realeval.py`, `realdata.py`, `features.py`,
`metrics.py`, the GBM parameters and the dataset list. The runner refuses a second run with
the same hash.

## Data and unit

- Datasets: Japan (1,080 tenders, 123 collusive), Italy (278, 143), Brazil (101, 33), from
  García Rodríguez et al. (2022) (log D1, D2). Switzerland and the USA are excluded: no
  firm IDs or no estimate.
- Unit: one tender. Label = the paper's tender label `Collusive_competitor` (constant within
  each tender). The per-bid membership label is not used for scoring or training.
- Order: Japan by date; Italy and Brazil by tender id (no or unreliable dates). Order only
  affects the block bootstrap.

## Features (no labels used; a test checks this)

1. Screens, from the tender's own bids (NaN when fewer than three bids): CV, skewness,
   kurtosis, relative distance, RD-alt, normalised gap, spread, and log number of bids.
   These are the same functions as the simulator benchmark.
2. Pair features. For every pair of the tender's bidders, statistics are computed on all
   other tenders of the same dataset (exact leave-tender-out; checked against brute force):
   - residual correlation over co-bids (Fisher z × √(n−3));
   - share of co-bids whose residuals are within 3 % of each other;
   - co-bidding phi × √(T−1).
   A pair needs ≥ 4 co-bids. Per tender: max and top-3 mean of each, plus coverage (the share
   of the tender's pairs with ≥ 4 co-bids). Residual = log(bid / pre-tender estimate) minus
   the tender median.
3. Scaling: each feature is replaced by its within-dataset percentile. This is label-free
   and uses the held-out dataset's unlabelled features (transductive), so currencies,
   project sizes and market density are comparable across countries.

## Designs (all frozen now)

| design | features |
|---|---|
| real_lit_gbm (baseline) | the 8 screens |
| real_pair_gbm | 7 pair features + log number of bids |
| real_full_gbm (proposed) | screens + pair features |
| single:scr_<screen> (7 baselines) | one screen; sign chosen on the training datasets |

The GBM uses the same parameters as the simulator benchmark (`models.GBM_PARAMS`),
random_state 0, no early stopping, so it is deterministic. Seeds therefore have no role
here; the uncertainty is sampling uncertainty over tenders.

## Protocol

- Leave-one-dataset-out: train on two datasets, test on the third (three folds). Each
  dataset is one country's cartel case, so this is the leave-one-cartel-out the rules ask
  for at the granularity the data allows.
- Alarm threshold: 5 % false alarms on 5-fold out-of-fold scores of the training datasets'
  competitive tenders. The false-alarm rate is then measured on the held-out dataset.
- Metrics per held-out dataset: AUC, precision@10 % (base rate = the dataset's collusive
  share), TPR and false-alarm rate at the threshold.
- CIs: moving-block bootstrap over tenders in time order (block 20, 1000 resamples, seed
  2026), with paired AUC differences real_full_gbm − each other design.
- One run only; logged in `results/real_runs.jsonl` and `docs/experiment_log.md`.

## Pre-registered reading of the results

- Main question: does real_full_gbm beat real_lit_gbm on each held-out dataset (paired CI
  above 0)? Reported per dataset, never pooled into one number.
- Also reported: whether the false-alarm rate on each held-out dataset is ≤ 10 %, and whether
  precision@10 % has its CI above the base rate.
- No new criteria thresholds are introduced for real data. AUC ≥ 0.85 is reported against,
  but it is not the simulator's C1.

## Known weaknesses (stated before the run)

- Three folds, each one cartel case. A result on one country is one case study, not a
  distribution.
- Italy is an average-bid auction, so the gap and relative-distance screens (lowest two
  bids) have a different meaning there. Italy is reported separately, with this caveat.
- The tender label is the paper's thresholded label; tenders where a minority of bidders
  were members may be labelled competitive (Japan: 216 tenders mix members and
  non-members).
- Pair features mostly see firms that bid often. In Brazil most firms bid once, so coverage
  will be low.
- Percentile scaling uses the held-out dataset's unlabelled features. That is a
  transductive choice, and it is disclosed.
- Brazil has 33 positives, so its CIs will be wide.
