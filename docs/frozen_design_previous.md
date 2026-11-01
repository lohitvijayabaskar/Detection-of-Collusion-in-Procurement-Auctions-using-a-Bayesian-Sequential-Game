# Proposed frozen design: `previous`

Status: PROPOSED. No locked run has been made for this design. A locked run needs the user's
approval (see "Approval needed").

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

## Hashes

| recipe | value |
|---|---|
| proposed: name, feature list, GBM_PARAMS, feature version, `bidrig/*.py` and `bidrig/detectors/*.py` | `5f10b7eb8738d81f` |
| what `scripts/run_locked_test.py` computes today (omits `bidrig/detectors/`) | differs; an edit under `bidrig/detectors/` would not change it |

The hash in `run_locked_test.py` covers `bidrig/*.py` only. For this design the one-run rule
would not detect an edit to `previous.py`. The proposed recipe closes this gap.

## Approval needed

1. Approve extending `design_hash` in `scripts/run_locked_test.py` to include
   `bidrig/detectors/*.py` (a change to the locked-run script).
2. Approve the single locked run of `previous` with the design above.

Until both are given, no locked run is made.
