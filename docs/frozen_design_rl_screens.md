# Proposed frozen design: `rl_screens`

Status: PROPOSED on 2026-11-01. Nothing has been run on the test split. A locked run needs the
user's approval.

## Design

- Name: `rl_screens` (`bidrig/detectors/rl_screens.py`).
- Arms: lit; dep; lit+dep; lit+dep+change; lit+dep+seq; all groups.
- Bandit: successive halving, 2 rounds; reward is the out-of-fold AUC of the standard GBM
  (`bidrig/detectors/config.py`) on the arm's features, 3-fold stratified CV on the training data
  passed to `fit` (CV seeds 0 and 1). Round 1 evaluates all six arms and keeps the best three;
  round 2 evaluates the survivors on a different split; the chosen arm has the highest mean
  reward (ties: first in arm order) and is refitted on all training data.
- The bandit re-runs inside every leave-one-scheme-out fold, on tuning data that excludes the
  held-out scheme. Validation and test data never enter the choice (`fit(X, y)` takes only the
  training split; a test checks the signature).
- Feature version 3. Training: `tuning` split. Alarm threshold: 5 % false alarms on `calib`. Seeds
  0-9; 1000 bootstrap resamples for the locked run. Scope: one global subset, not per market, and
  not PPO.

## Validation evidence (`results/variant_rl_screens_v1.md`)

Macro AUC 0.863 [0.855, 0.871] vs 0.862 [0.853, 0.871] for `full_gbm`; paired difference +0.001
[-0.001, 0.003]. Arm choices per seed: `results/rl_screens_arms_rl_screens_arms_v1.md`.

## Hash

`design_hash("rl_screens")` from `scripts/run_locked_test.py`: `002a3d7267394d48`. It hashes every file under
`bidrig/` (subpackages included), so any code change before the run changes it and the freeze must
then be re-proposed.

## Known gap (not fixed; needs approval)

A design name that already has a locked run is not refused by name; the guard is hash-based only.

## Approval needed

One locked run: `cd scripts && python run_locked_test.py --design rl_screens --note "<reason>"`
(about 50 minutes).
