# Experiment log

Each entry: date, hypothesis, change, command, outcome, failures. Locked-test runs are marked
**LOCKED** and also recorded in `results/locked_runs.jsonl`.

## 2026-10-21 E0: setup and pre-registered protocol

Lessons carried over from the old repo (`docs/experiment_log.md` there):
- A price-level screen flagged honest markups as collusion. Here every identity feature uses
  residuals relative to the tender median, which removes uniform markups and the estimate.
- Calibration that did not match the simulator's start state gave a 70 % false-alarm rate.
  Here thresholds are set on the `calib` split, drawn from the same generator as `tuning`.
- An AUC rank bug once printed all zeros. Here every metric has separated, reversed and tied
  tests (`tests/test_metrics.py`).
- Background scripts with syntax errors were launched unchecked. Scripts are syntax-checked
  with `py_compile` and logs are checked after launch.

Protocol fixed before any validation numbers were seen:
- Splits: `tuning` (400 honest + 50 per mechanism), `calib` (300 honest, same spec), and
  `validation` (honest and each mechanism at in-distribution, sizes {8, 9, 23, 24} and
  spreads [0.04, 0.06] ∪ [0.18, 0.22]; 100 honest / 25 collusive per set). `test`, which is
  locked, has in-distribution sets, unseen sizes {6, 7, 25..30}, unseen spreads
  [0.02, 0.04] ∪ [0.22, 0.30], three honest stress sets (markup 20-25 %, capacity 0.15-0.30,
  noisy estimates and costs) and the uniform-markup control. It has 100 honest / 40 collusive
  per set. Tuning range: 10-22 firms, spread 0.06-0.18, 20-60 tenders.
- Seeds 0-9. CIs use a two-level bootstrap (seeds, then markets within sets), paired across
  designs.
- C1: for each held-out mechanism, LOMO AUC ≥ 0.85, both pooled against all honest test sets
  and per condition (in / size / spread) against honest markets of the same condition.
- C2: the alarm rate at the threshold set for 5 % false alarms on `calib`, using the model
  trained on all mechanisms, is ≤ 0.10 on every honest test set.
- C3: the paired difference in macro LOMO AUC against every baseline (lit GBM and seven
  single screens) has its 95 % CI lower bound above 0.
- C4: for every mechanism, the lower CI bound of precision@10 % (held-out mechanism plus all
  honest markets) is above the base rate (1/6 on test).

Checks: 35 tests pass; `check_determinism.py` gives identical hashes for 1 vs 4 workers and
across processes, and different hashes for seed+1.

Smoke run (not evidence; 2 seeds, scale 0.15, 30 bootstraps):
`run_validation.py --tag smoke --designs full_gbm,dep_gbm --seeds 0-1 --scale 0.15`.
Macro LOMO AUC: full_gbm 0.748, dep_gbm 0.693, lit_gbm 0.739. Dependence features alone are
weaker than the literature GBM at this scale.
