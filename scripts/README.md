# Scripts

Run from this folder (`cd scripts`); each script adds the repo root to the import path.

| script | what it does | writes |
|---|---|---|
| `build_features.py` | Generate markets for a split and cache their features (`cache/`, gitignored) | `cache/*.npz` |
| `check_determinism.py` | Check the same seed gives identical features across processes and worker counts | stdout |
| `run_validation.py` | Score named designs and the baselines on the VALIDATION split | `results/validation_<tag>.{json,md}` |
| `run_variant_validation.py` | Score one detector variant against a reference (default `full_gbm`) on validation, with paired CIs | `results/variant_<tag>.{json,md}` |
| `select_stat_auc.py` | Choose the `stat_auc` configuration on validation (16 configurations) | `results/stat_auc_selection_<tag>.{json,md}` |
| `rl_screens_arms.py` | Diagnostic: which screen subset the bandit picks per seed (tuning split only) | `results/rl_screens_arms_<tag>.{json,md}` |
| `run_locked_test.py` | LOCKED test run, once per frozen design; refuses a repeated design hash | `results/locked_<n>_<design>.{json,md}`, `results/locked_runs.jsonl` |
| `run_real_lodo.py` | The single approved real-data leave-one-dataset-out run (needs `BIDRIG_REAL_DATA`) | `results/real_lodo_<n>.{json,md}`, `results/real_runs.jsonl` |
| `make_figures.py` | Regenerate the paper figures from logged result files (no experiment re-run) | `docs/figures/` |
| `_common.py` | Shared helpers (repo root, results folder, seed parsing) | |

Only `run_locked_test.py` and `run_real_lodo.py` touch the test split or the real data, and
only after a frozen design is approved. Never run them while choosing a design.
