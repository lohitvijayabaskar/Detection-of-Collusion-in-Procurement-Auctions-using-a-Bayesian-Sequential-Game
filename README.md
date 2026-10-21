# bidrig: a bid-rigging detection benchmark

A simulator of repeated first-price procurement tenders with persistent bidders, a library of
imposed collusion mechanisms, market-level detectors, and a locked evaluation protocol
(leave-one-mechanism-out, held-out market sizes and cost spreads, 10 seeds, bootstrap CIs).

Status: in development. See `docs/experiment_log.md` for every run and
`docs/claims_table.md` for what is supported.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
cd scripts
python build_features.py --seeds 0-9
python run_validation.py --tag first --designs full_gbm,knowledge_gbm,dep_gbm
```

## Benchmark

- Markets: 6-30 firms, 20-60 tenders, randomised cost spread, estimate noise, participation,
  specialisation, capacity backlog, honest uniform markups of 0-25 %.
- Mechanisms (in scope): rotation, cover bidding, complementary bidding, price fixing (three
  levels), partial cartel with a fringe, noisy/defecting cartel, intermittent cartel, bid
  suppression. Negative control (out of scope): uniform-markup collusion.
- Detectors: literature screens (CV, skewness, kurtosis, relative distance, RD-alt, gap,
  spread) alone and in a GBM; identity-dependence features against a within-tender permutation
  null; within-market change scans; a sequential layer (CUSUM, HMM filter); GBM combinations
  and ablations.

## Limitations

Synthetic data only; no labelled per-bid real dataset has been supplied.
