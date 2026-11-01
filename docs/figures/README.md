# Figures

Regenerate with `cd scripts && python make_figures.py`. The script reads only logged result
files; nothing is re-run. Each figure has its data table in `data/<figure>.csv`.

| figure | what it shows | source |
|---|---|---|
| `fig1_locked_auc_by_mechanism.png` | LOMO AUC per held-out mechanism × condition (in / unseen sizes / unseen spreads), 0.85 target; four mechanisms miss it | `results/locked_1_full_gbm.json` |
| `fig2_locked_vs_baselines.png` | macro LOMO AUC: full_gbm vs screens GBM vs seven single screens | `results/locked_1_full_gbm.json` |
| `fig3_locked_false_alarms.png` | false-alarm rate on six honest test sets, deployable vs worst leave-one-out model | `results/locked_1_full_gbm.json` |
| `fig4_locked_precision10.png` | precision@10 % per mechanism vs the 0.167 base rate | `results/locked_1_full_gbm.json` |
| `fig5_validation_history.png` | validation macro and worst-mechanism AUC across iterations (V1, I1, I2, I4, W2-3) | `results/validation_*.json` |
| `fig6_ablation.png` | validation ablation of feature groups (feature v2) | `results/validation_v2.json` |
| `fig7_real_data_lodo.png` | real data, tender-level leave-one-country-out: AUC and false alarms | `results/real_lodo_1.json` |
| `fig8_old_env_transfer.png` | frozen detector on the old RL environment: AUC (E1) and alarm rates at the transferred vs refitted threshold (E2) | `results/old_env/new_detector_on_old_env/`, `results/old_env/new_detector_recalibrated/` (copies of the results of the old-environment checks) |

Colours: dataviz reference categorical slots 1-3 (validated all-pairs, light mode) and a
neutral gray. Series are also told apart by marker shape and legend or direct labels.
