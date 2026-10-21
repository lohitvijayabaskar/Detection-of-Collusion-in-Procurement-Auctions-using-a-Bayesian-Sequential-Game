# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

A benchmark and detector for bid rigging in repeated first-price procurement tenders, built
from scratch (the older RL repo `../Jayati ma'am project` was only read for lessons). Cartels
are imposed from a library of mechanisms, not learned. The detector scores whole markets from
public data: bids, bidder identities, per-tender public estimates. Results, failures included,
go in `docs/experiment_log.md`. Claims go in `docs/claims_table.md` only with the numbers that
support them.

## Commands (run from repo root unless noted; scripts are run from `scripts/`)

```bash
pip install -r requirements.txt
python -m pytest -q                                   # all tests (~20 s)
black . && isort .                                    # formatting (black target py310)
cd scripts && python check_determinism.py             # same seed -> identical features across processes
cd scripts && python build_features.py --seeds 0-9    # cache tuning/calib/validation features (~7 min)
cd scripts && python run_validation.py --tag <tag> --designs full_gbm,dep_gbm --seeds 0-9
cd scripts && python run_locked_test.py --design <name> --note "<why frozen>"   # LOCKED, once per design
```

Long jobs: start in the background, write the log to `results/logs/`, and check the log is
non-empty before relying on it.

## Architecture

- `bidrig/simulator.py`: `MarketSpec` (ranges as unions of intervals), `sample_config`,
  `simulate_market(cfg, mechanism, rng)`. Persistent firms: efficiency, type specialisation,
  capacity backlog. The estimate is project size times noise; it is drawn before any
  mechanism parameter, so it is identical for honest and collusive markets of the same seed.
- `bidrig/collusion.py`: `Mechanism` subclasses. `IN_SCOPE` lists the 8 in-scope mechanisms;
  `uniform_markup` is the out-of-scope negative control (`in_scope=False`).
- `bidrig/features.py`: groups `lit` (literature screens), `dep` (identity dependence vs a
  within-tender permutation null), `change` (Brownian-bridge scans), `seq` (CUSUM, HMM filter).
  `FEATURE_GROUPS` and `ALL_FEATURES` define column order.
- `bidrig/datasets.py`: split definitions (`tuning`, `calib`, `validation`, `test`), per-market
  rng `default_rng([seed, crc(split), crc(set), i, stream])`, feature cache in `cache/`
  (gitignored; bump `FEATURE_VERSION` when features or simulator change).
- `bidrig/models.py`: `DESIGNS` (GBM on feature groups), `BASELINES` (lit GBM + single screens).
- `bidrig/protocol.py`: LOMO folds, 5 % threshold from `calib`, metrics, two-level bootstrap,
  `criteria`.
- `scripts/run_locked_test.py`: refuses a second run of the same design hash (design name,
  groups, GBM params, feature version and all `bidrig/*.py` source). Every run is appended to
  `results/locked_runs.jsonl`.

## Rules

- Never touch the test split while choosing designs; validation only.
- No global `random`; pass an `rng`. Sort sets before iterating.
- Never delete files in `results/`. Ask the user before choosing paper claims, any run over
  about 2 hours, downloading data, or dropping a workstream.
- Commit only with tests passing, black/isort clean, on the work branch.
