# Claims table

Only claims backed by a logged run. "Evidence" names the result file. Locked-test numbers
come from `results/locked_*.json`; validation numbers are marked as such and are not
headline results.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Same seed gives identical features across processes and worker counts | supported | `scripts/check_determinism.py` (E0) |
