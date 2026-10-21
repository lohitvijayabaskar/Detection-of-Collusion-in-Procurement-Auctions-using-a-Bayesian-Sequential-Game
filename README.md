# bidrig: a bid-rigging detection benchmark

A simulator of repeated first-price procurement tenders with persistent bidders, a library of
imposed collusion mechanisms, market-level detectors, and a locked evaluation protocol
(leave-one-mechanism-out, held-out market sizes and cost spreads, 10 seeds, bootstrap CIs).

Status: one locked test run (2026-10-21). Three of the four pre-registered criteria are met;
C1 (every held-out mechanism at AUC ≥ 0.85) is not. By the project's own criteria this is
not publication-ready. See `docs/experiment_log.md` for every run and
`docs/claims_table.md` for what is supported.

## Locked result (run 1, design full_gbm, seeds 0-9, 95 % bootstrap CIs)

| criterion | result | met |
|---|---|---|
| C1 LOMO AUC ≥ 0.85 per mechanism, incl. unseen sizes/spreads | rotation, price fixing, complementary, cover ≥ 0.92 in every condition; intermittent 0.849 (unseen spreads); noisy/defecting 0.823; bid suppression 0.744; partial cartel with fringe 0.672 | no |
| C2 FA ≤ 10 % on every honest set | max 0.075 [0.050, 0.105] (unseen sizes) | yes |
| C3 beats literature baselines, CI excluding 0 | macro AUC 0.870 vs lit GBM 0.803, difference +0.067 [0.060, 0.074]; vs single screens +0.196 to +0.382 | yes |
| C4 precision@10 % above base rate (0.167), CI excluding it | lowest 0.378 [0.335, 0.418] | yes |

Uniform-markup negative control: AUC 0.478 [0.450, 0.503], as expected for an
identification limit.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
cd scripts
python check_determinism.py
python build_features.py --seeds 0-9
python run_validation.py --tag first --designs full_gbm,knowledge_gbm,dep_gbm
```

`run_locked_test.py` runs the test split once per frozen design and refuses repeats.

## Benchmark

- Markets: 6-30 firms, 20-60 tenders, randomised cost spread, estimate noise, participation,
  specialisation, capacity backlog, honest uniform markups of 0-25 %.
- Mechanisms (in scope): rotation, cover bidding, complementary bidding, price fixing (three
  levels), partial cartel with a fringe, noisy/defecting cartel, intermittent cartel, bid
  suppression. Negative control (out of scope): uniform-markup collusion.
- Detectors: literature screens (CV, skewness, kurtosis, relative distance, RD-alt, gap,
  spread) alone and in a GBM; identity-dependence features standardised against two
  within-market nulls (within-tender and within-firm shuffles); participation features;
  within-market change scans; a sequential layer (CUSUM, HMM filter); GBM combinations,
  ablations, a knowledge-directed tail score and monotone variants.

## Limitations

- Real data: one tender-level leave-one-dataset-out run on Japan, Italy and Brazil
  (García Rodríguez et al. 2022). Pair features beat the screens on Italy only (+0.144
  [0.078, 0.234] AUC). On Japan and Brazil a single CV screen does as well or better, and
  the alarm threshold did not transfer to Japan (26.5 % false alarms). See
  `results/real_lodo_1.md`.
- Small cartels inside a competitive fringe, bid suppression (when held out) and noisy or
  defecting cartels are not detected at AUC ≥ 0.85.
- Change scans (single-change and epidemic) and the HMM/CUSUM layer added no measurable
  power on validation.
- Market-level scores only; cartel members are not identified.
