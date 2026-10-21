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

## 2026-10-21 V1: first validation run (feature v1, tender null only)

- Command: `run_validation.py --tag v1 --designs full_gbm,knowledge_gbm,dep_gbm,change_gbm,seq_gbm,full_minus_dep,full_minus_change,full_minus_seq` (seeds 0-9, 300 bootstraps). Output `results/validation_v1.{json,md}`.
- Macro LOMO AUC [95 % CI]: full_gbm 0.847 [0.840, 0.855]; lit_gbm 0.806 [0.795, 0.816]; dep_gbm 0.771; knowledge_gbm 0.770; change_gbm 0.494; seq_gbm 0.549; best single screen (rdalt) 0.692.
- full_gbm per mechanism (pooled): rotation 0.946, price_fixing 0.945, cover 0.932, complementary 0.930, intermittent 0.833, noisy_defecting 0.784, bid_suppression 0.742, partial_fringe 0.661.
- FA at the calib 5 % threshold (all-mechanism model): in 0.053, size 0.096, spread 0.061.
- Validation criteria for full_gbm: C1 fail (4 mechanisms < 0.85), C2 pass, C3 pass, C4 pass.
- Diagnosis (univariate AUCs, tuning seed 0): many dependence z-scores are below 0.5 because the tender null also destroys honest persistent-efficiency structure, so honest markets get large z-scores too. The change and seq groups carry almost no signal alone.
- Process note: the bootstrap took about 50 minutes because macro and difference statistics recomputed every AUC; a per-resample memo was added to `evaluate` (no change to values).

## 2026-10-21 I1 (workstream: dependence features): time null added to the tender null (feature v2)

- Hypothesis: standardising identity statistics against a null that shuffles each firm's residuals across its tenders keeps honest persistent-efficiency structure in the null, so collusive coordination stands out more. Falsifier: macro LOMO AUC of dep_gbm not above v1's CI.
- Change: `permute_within_firms`; every identity statistic is reported against both nulls (`dep_<k>` tender, `dep_<k>_t` time; firm dispersion raw under the time null); new `close_pair` statistic (top-3 pair share of co-bids within 3 %). The time null alone (first try, tuning seed 0 univariate) removed the win/loss signal (cover 0.82 → 0.52), hence both nulls.
- Command: `build_features.py --seeds 0-9`, then `run_validation.py --tag v2 ...` (same designs).
- Outcome: dep_gbm 0.771 → 0.826 [0.814, 0.836]; full_gbm 0.847 → 0.862 [0.853, 0.871]; knowledge_gbm 0.770 → 0.822. full_gbm per mechanism: price_fixing 0.972, complementary 0.957, rotation 0.951, cover 0.945, intermittent 0.856, noisy_defecting 0.794, bid_suppression 0.751, partial_fringe 0.668. FA in/size/spread 0.051/0.073/0.053. Validation criteria: C1 fail, C2-C4 pass. KEPT.
- Ablation: full_minus_change 0.864, full_minus_seq 0.862, full_minus_dep 0.801. The change and seq groups add nothing measurable; dep adds about 0.06.
