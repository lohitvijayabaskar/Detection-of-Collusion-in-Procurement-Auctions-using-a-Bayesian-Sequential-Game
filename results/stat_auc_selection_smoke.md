# stat_auc selection (validation, seeds 0-1)

| subset | C | macro AUC |
|---|---|---|
| dep | 0.01 | 0.8206 |
| dep | 0.1 | 0.8191 |
| dep | 1 | 0.8091 |
| dep | 10 | 0.8068 |
| lit_dep | 0.01 | 0.8568 |
| lit_dep | 0.1 | 0.8517 |
| lit_dep | 1 | 0.8279 |
| lit_dep | 10 | 0.7924 |
| lit_dep_seq | 0.01 | 0.8562 |
| lit_dep_seq | 0.1 | 0.8485 |
| lit_dep_seq | 1 | 0.8244 |
| lit_dep_seq | 10 | 0.8001 |
| all | 0.01 | 0.8612  <- chosen |
| all | 0.1 | 0.8469 |
| all | 1 | 0.8245 |
| all | 10 | 0.8068 |

CHOSEN = ("all", 0.01)  # freeze this in bidrig/detectors/stat_auc.py