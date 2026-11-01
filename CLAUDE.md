# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

A benchmark and detector for bid rigging in repeated first-price procurement tenders, built
from scratch (an older reinforcement-learning project was only read for lessons). Cartels are
imposed from a library of mechanisms, not learned. Detectors score whole markets from public
data: bids, bidder identities, per-tender public estimates. Results, failures included, go in
`docs/experiment_log.md`. Claims go in `docs/claims_table.md` only with the numbers that
support them.

## Commands (run from repo root unless noted; scripts are run from `scripts/`)

```bash
pip install -r requirements.txt
python -m pytest -q                                   # all tests (~2.5 min, includes the golden test)
black . && isort .                                    # formatting (black target py310)
cd scripts && python check_determinism.py             # same seed -> identical features across processes
cd scripts && python build_features.py --seeds 0-9    # cache tuning/calib/validation features (~7 min)
cd scripts && python run_validation.py --tag <tag> --designs full_gbm,dep_gbm --seeds 0-9
cd scripts && python run_variant_validation.py --variant <name> --tag <tag> --seeds 0-9   # variant vs full_gbm
cd scripts && python run_locked_test.py --design <name> --note "<why frozen>"   # LOCKED, once per design
cd scripts && python make_figures.py                  # regenerate docs/figures from logged results
```

Real data (Japan, Italy, Brazil) lives outside the repo: set `BIDRIG_REAL_DATA` to the folder
with the extracted CSV files (default `data/real`). Long jobs: start in the background, write the
log to `results/logs/`, and check the log is non-empty before relying on it. `scripts/README.md`
lists every script.

## Architecture

```
bidrig/
  simulator.py      markets: MarketSpec, sample_config, simulate_market
  collusion.py      the eight in-scope mechanisms and the uniform-markup control
  features/         one module per feature group (literature, dependence, participation,
                    change, sequential) + common, groups (column order), extract (market_features)
  datasets.py       splits (tuning, calib, validation, test), per-market rng, feature cache
  detectors/        one registered module per detector family (see below)
  protocol.py       leave-one-mechanism-out folds, 5 % threshold, metrics, bootstrap, criteria
  metrics.py        AUC, threshold, precision@K, bootstrap CIs
  report.py         plain-text tables
  realdata.py       loader for the real bid data (windowing helpers)
  realeval.py       tender-level leave-one-dataset-out evaluation on the real data
  models.py         compatibility layer only (re-exports from detectors; no logic)
```

- Simulator: the estimate is project size times noise; it is drawn before any mechanism
  parameter, so it is identical for honest and collusive markets of the same seed.
- Features: `bidrig.features.FEATURE_GROUPS` and `ALL_FEATURES` define column order; bump
  `FEATURE_VERSION` in `datasets.py` when features or the simulator change.
- Detectors: every detector has `fit(X, y)` and `score(X)` and is built by name with
  `bidrig.detectors.make(name)`. Modules depend only on `detectors/base.py`, `config.py`,
  `priors.py` and `bidrig.features`; no detector module imports another. Families:
  `gbm_groups` (`full_gbm` and ablations), `single_screen`, `tail_fused`, `previous`,
  `stat_auc`, `rl_screens`.

### Adding a detector variant

1. New file `bidrig/detectors/<name>.py` with a class and `@register("<name>")`.
2. Import it in `bidrig/detectors/__init__.py`.
3. Unit tests in `tests/test_<name>.py`; add the name to `DETECTORS` in `tests/test_golden.py` and
   regenerate the reference only with a logged reason.
4. Pre-register the hypothesis in `docs/experiment_log.md`, smoke-test, validate with
   `run_variant_validation.py`, log, then propose the frozen design before any locked run.

## Rules

- Never touch the test split while choosing designs; validation only.
- No global `random`; pass an `rng`. Sort sets before iterating.
- A locked run is made once per frozen design, only after the user approves it.
  `scripts/run_locked_test.py` hashes every file under `bidrig/`; code moves change every hash.
  Known gap: a design name that already has a locked run is not refused by name.
- `tests/test_golden.py` guards behaviour: a refactor must not change feature values or
  detector scores. `tests/test_no_identifying_paths.py` keeps user-specific paths out of code.
- Never delete files in `results/` or `docs/`. Ask the user before choosing paper claims, any run
  over about 2 hours, downloading data, or dropping a workstream.
- Commit only with tests passing, black/isort clean, on the work branch.
