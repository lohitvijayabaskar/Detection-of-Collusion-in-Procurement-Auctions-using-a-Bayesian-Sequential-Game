# Proposed frozen design: `stat_auc`

Status: PROPOSED on 2026-11-01. Nothing has been run on the test split. A locked run needs the
user's approval.

## Design

- Name: `stat_auc` (`bidrig/detectors/stat_auc.py`, `CHOSEN = ("lit_dep", 0.1)`).
- Model: median imputation then standardisation (both fitted on the tuning split), L2 logistic
  regression (lbfgs, max_iter 2000, C = 0.1) on the 43 columns of the `lit` and `dep` groups.
- Selection: macro leave-one-scheme-out AUC on the VALIDATION split, seeds 0-9, 16 grid
  configurations (subsets dep, lit_dep, lit_dep_seq, all; C in 0.01, 0.1, 1, 10). Record:
  `results/stat_auc_selection_stat_auc_select_v1.{json,md}`.
- Feature version 3. Training: `tuning` split. Alarm threshold: 5 % false alarms on `calib`.
  Seeds 0-9; 1000 bootstrap resamples for the locked run.

## Validation evidence (`results/variant_stat_auc_v1.md`, optimistic: validation-tuned)

Macro AUC 0.864 [0.856, 0.871] vs 0.862 [0.853, 0.871] for `full_gbm`; paired difference +0.002
[-0.005, 0.009].

## Hash

`design_hash("stat_auc")` from `scripts/run_locked_test.py`: `6a64930fd6bed238`. It covers `bidrig/*.py` and
`bidrig/detectors/*.py`, so any edit to those files changes it.

## Known gaps (not fixed; need approval)

- Adding or editing any detector module changes every design's hash, so earlier frozen hashes
  (for example `previous`, locked as 90b0cbe63f309f0e) no longer match the current source and the
  hash alone would not refuse a repeat run. A name-based guard in `run_locked_test.py` would close
  this.

## Approval needed

One locked run: `cd scripts && python run_locked_test.py --design stat_auc --note "<reason>"`
(about 15 minutes). The unbiased test of the "validation-tuned" caveat.
