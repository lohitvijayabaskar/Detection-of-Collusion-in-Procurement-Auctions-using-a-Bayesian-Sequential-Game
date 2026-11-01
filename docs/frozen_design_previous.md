# Proposed frozen design: `previous`

Status: FROZEN on 2026-11-01 with the user's approval of the hash fix and the freeze. The
freeze hash is `90b0cbe63f309f0e`, computed by `scripts/run_locked_test.py` (`design_hash("previous")`).
No locked run has been made. The locked run itself still needs the user's separate go-ahead.

## Design

- Name: `previous` (registered in `bidrig/detectors/previous.py`).
- Model: gradient boosting with the shared settings `models.GBM_PARAMS` (200 iterations,
  learning rate 0.05, 15 leaves, min 20 samples per leaf, L2 1.0, no early stopping,
  random_state 0).
- Features: the 40 columns of `full_gbm` (groups lit, dep, change, seq) after removing every
  column whose name ends in `_t` (the within-firm null).
- Feature version: 3 (`bidrig/datasets.FEATURE_VERSION`).
- Training data: the `tuning` split only. Alarm threshold: 5 % false alarms on the `calib` split.
  Seeds 0-9. Bootstrap: 1000 resamples for the locked run.

## Why it qualifies for a locked run

Validation (`results/variant_previous_v1.md`, seeds 0-9, 300 resamples): macro LOMO AUC 0.848
[0.839, 0.855] against 0.862 [0.853, 0.871] for `full_gbm`; paired difference -0.014
[-0.019, -0.008]. The falsifying test (gap inside the CI) did not trigger. No design choice was
made on the test split.

## Hash

- Frozen hash: `90b0cbe63f309f0e` (from `design_hash("previous")`).
- `design_hash` now covers the design name, `GBM_PARAMS`, feature version and every file in
  `bidrig/*.py` and `bidrig/detectors/*.py`. Any edit to those files changes the hash, so the
  one-run rule can tell that the design changed. A test checks this
  (`tests/test_locked_hash.py`).
- The earlier proposal hash `5f10b7eb8738d81f` used a different recipe and is superseded.

## Known gap in the one-run guard (not fixed; needs approval)

The guard compares hashes only. The `full_gbm` hash recorded in locked run 1 is
`6cc622540eac3929`; today `design_hash("full_gbm")` gives a different value because source files
changed after that run (for example the registry lookup in `models.make_design`). A repeat locked
run of `full_gbm` would therefore not be refused. A name-based check (refuse a design name that
already has a locked run) would close this. It is outside the approved edit, so it is not made.

## Approval still needed

One locked run of `previous` with this design: `cd scripts && python run_locked_test.py --design
previous --note "<reason>"`. It takes about ten minutes and the script refuses a second run
with this hash.
