# Validation smoke (seeds 0-1, feature v1)

| design | macro LOMO AUC | macro prec@10% | min mech AUC |
|---|---|---|---|
| full_gbm | 0.748 [0.699, 0.796] | 0.521 [0.417, 0.689] | 0.522 |
| dep_gbm | 0.693 [0.626, 0.739] | 0.437 [0.330, 0.542] | 0.524 |
| lit_gbm | 0.739 [0.701, 0.767] | 0.625 [0.549, 0.757] | 0.545 |
| single:lit_cv_mean | 0.652 [0.575, 0.694] | 0.479 [0.354, 0.583] | 0.500 |
| single:lit_skew_mean | 0.568 [0.489, 0.625] | 0.531 [0.432, 0.625] | 0.418 |
| single:lit_kurt_mean | 0.524 [0.453, 0.598] | 0.312 [0.244, 0.460] | 0.356 |
| single:lit_rd_mean | 0.631 [0.566, 0.702] | 0.521 [0.372, 0.638] | 0.502 |
| single:lit_rdalt_mean | 0.650 [0.557, 0.724] | 0.583 [0.476, 0.708] | 0.532 |
| single:lit_gap_mean | 0.553 [0.460, 0.613] | 0.281 [0.185, 0.461] | 0.397 |
| single:lit_spread_mean | 0.646 [0.578, 0.694] | 0.479 [0.390, 0.586] | 0.469 |

| difference (macro AUC) | value [95% CI] |
|---|---|
| dep_gbm-lit_gbm | -0.046 [-0.127, 0.005] |
| dep_gbm-single:lit_cv_mean | 0.041 [-0.033, 0.122] |
| dep_gbm-single:lit_gap_mean | 0.140 [0.023, 0.247] |
| dep_gbm-single:lit_kurt_mean | 0.169 [0.035, 0.277] |
| dep_gbm-single:lit_rd_mean | 0.062 [-0.004, 0.123] |
| dep_gbm-single:lit_rdalt_mean | 0.043 [-0.027, 0.087] |
| dep_gbm-single:lit_skew_mean | 0.125 [0.063, 0.185] |
| dep_gbm-single:lit_spread_mean | 0.047 [-0.041, 0.119] |
| full_gbm-lit_gbm | 0.009 [-0.040, 0.052] |
| full_gbm-single:lit_cv_mean | 0.096 [0.038, 0.187] |
| full_gbm-single:lit_gap_mean | 0.195 [0.114, 0.282] |
| full_gbm-single:lit_kurt_mean | 0.224 [0.113, 0.328] |
| full_gbm-single:lit_rd_mean | 0.117 [0.018, 0.173] |
| full_gbm-single:lit_rdalt_mean | 0.097 [0.013, 0.174] |
| full_gbm-single:lit_skew_mean | 0.180 [0.083, 0.236] |
| full_gbm-single:lit_spread_mean | 0.102 [0.033, 0.173] |
| lit_gbm-single:lit_cv_mean | 0.087 [0.052, 0.156] |
| lit_gbm-single:lit_gap_mean | 0.186 [0.127, 0.269] |
| lit_gbm-single:lit_kurt_mean | 0.215 [0.142, 0.302] |
| lit_gbm-single:lit_rd_mean | 0.108 [0.034, 0.184] |
| lit_gbm-single:lit_rdalt_mean | 0.089 [0.020, 0.175] |
| lit_gbm-single:lit_skew_mean | 0.171 [0.108, 0.251] |
| lit_gbm-single:lit_spread_mean | 0.093 [0.053, 0.150] |
| single:lit_cv_mean-lit_gbm | -0.087 [-0.156, -0.052] |
| single:lit_cv_mean-single:lit_gap_mean | 0.099 [0.032, 0.172] |
| single:lit_cv_mean-single:lit_kurt_mean | 0.128 [0.020, 0.223] |
| single:lit_cv_mean-single:lit_rd_mean | 0.021 [-0.086, 0.091] |
| single:lit_cv_mean-single:lit_rdalt_mean | 0.002 [-0.086, 0.075] |
| single:lit_cv_mean-single:lit_skew_mean | 0.084 [0.001, 0.147] |
| single:lit_cv_mean-single:lit_spread_mean | 0.006 [-0.027, 0.034] |
| single:lit_gap_mean-lit_gbm | -0.186 [-0.269, -0.127] |
| single:lit_gap_mean-single:lit_cv_mean | -0.099 [-0.172, -0.032] |
| single:lit_gap_mean-single:lit_kurt_mean | 0.029 [-0.029, 0.113] |
| single:lit_gap_mean-single:lit_rd_mean | -0.078 [-0.224, 0.040] |
| single:lit_gap_mean-single:lit_rdalt_mean | -0.098 [-0.234, 0.021] |
| single:lit_gap_mean-single:lit_skew_mean | -0.015 [-0.146, 0.107] |
| single:lit_gap_mean-single:lit_spread_mean | -0.093 [-0.176, -0.029] |
| single:lit_kurt_mean-lit_gbm | -0.215 [-0.302, -0.142] |
| single:lit_kurt_mean-single:lit_cv_mean | -0.128 [-0.223, -0.020] |
| single:lit_kurt_mean-single:lit_gap_mean | -0.029 [-0.113, 0.029] |
| single:lit_kurt_mean-single:lit_rd_mean | -0.107 [-0.239, 0.024] |
| single:lit_kurt_mean-single:lit_rdalt_mean | -0.127 [-0.263, 0.039] |
| single:lit_kurt_mean-single:lit_skew_mean | -0.044 [-0.152, 0.106] |
| single:lit_kurt_mean-single:lit_spread_mean | -0.122 [-0.220, -0.027] |
| single:lit_rd_mean-lit_gbm | -0.108 [-0.184, -0.034] |
| single:lit_rd_mean-single:lit_cv_mean | -0.021 [-0.091, 0.086] |
| single:lit_rd_mean-single:lit_gap_mean | 0.078 [-0.040, 0.224] |
| single:lit_rd_mean-single:lit_kurt_mean | 0.107 [-0.024, 0.239] |
| single:lit_rd_mean-single:lit_rdalt_mean | -0.019 [-0.067, 0.029] |
| single:lit_rd_mean-single:lit_skew_mean | 0.063 [0.013, 0.118] |
| single:lit_rd_mean-single:lit_spread_mean | -0.015 [-0.080, 0.079] |
| single:lit_rdalt_mean-lit_gbm | -0.089 [-0.175, -0.020] |
| single:lit_rdalt_mean-single:lit_cv_mean | -0.002 [-0.075, 0.086] |
| single:lit_rdalt_mean-single:lit_gap_mean | 0.098 [-0.021, 0.234] |
| single:lit_rdalt_mean-single:lit_kurt_mean | 0.127 [-0.039, 0.263] |
| single:lit_rdalt_mean-single:lit_rd_mean | 0.019 [-0.029, 0.067] |
| single:lit_rdalt_mean-single:lit_skew_mean | 0.082 [0.049, 0.124] |
| single:lit_rdalt_mean-single:lit_spread_mean | 0.004 [-0.090, 0.104] |
| single:lit_skew_mean-lit_gbm | -0.171 [-0.251, -0.108] |
| single:lit_skew_mean-single:lit_cv_mean | -0.084 [-0.147, -0.001] |
| single:lit_skew_mean-single:lit_gap_mean | 0.015 [-0.107, 0.146] |
| single:lit_skew_mean-single:lit_kurt_mean | 0.044 [-0.106, 0.152] |
| single:lit_skew_mean-single:lit_rd_mean | -0.063 [-0.118, -0.013] |
| single:lit_skew_mean-single:lit_rdalt_mean | -0.082 [-0.124, -0.049] |
| single:lit_skew_mean-single:lit_spread_mean | -0.078 [-0.164, 0.005] |
| single:lit_spread_mean-lit_gbm | -0.093 [-0.150, -0.053] |
| single:lit_spread_mean-single:lit_cv_mean | -0.006 [-0.034, 0.027] |
| single:lit_spread_mean-single:lit_gap_mean | 0.093 [0.029, 0.176] |
| single:lit_spread_mean-single:lit_kurt_mean | 0.122 [0.027, 0.220] |
| single:lit_spread_mean-single:lit_rd_mean | 0.015 [-0.079, 0.080] |
| single:lit_spread_mean-single:lit_rdalt_mean | -0.004 [-0.104, 0.090] |
| single:lit_spread_mean-single:lit_skew_mean | 0.078 [-0.005, 0.164] |

## full_gbm

criteria (validation): {'C1_lomo_auc>=0.85': False, 'C2_fa<=0.10_every_honest_set': False, 'C3_beats_baselines_ci': False, 'C4_prec10_above_base_ci': False}

| mechanism | LOMO AUC (pooled) | AUC @in | AUC @size | AUC @spread | prec@10% | base | TPR@5%FA |
|---|---|---|---|---|---|---|---|
| bid_suppression | 0.735 [0.629, 0.828] | 0.767 [0.626, 0.952] | 0.667 [0.487, 0.805] | 0.733 [0.463, 0.928] | 0.417 [0.204, 0.750] | 0.211 | 0.208 [0.030, 0.470] |
| complementary | 0.735 [0.551, 0.905] | 0.533 [0.173, 0.849] | 0.783 [0.583, 0.925] | 0.850 [0.608, 1.000] | 0.500 [0.250, 0.773] | 0.211 | 0.375 [0.292, 0.542] |
| cover_bidding | 0.835 [0.778, 0.908] | 0.900 [0.803, 1.000] | 0.717 [0.575, 0.923] | 0.933 [0.812, 1.000] | 0.750 [0.394, 0.917] | 0.211 | 0.542 [0.405, 0.678] |
| intermittent | 0.769 [0.651, 0.828] | 0.842 [0.715, 0.958] | 0.692 [0.410, 0.890] | 0.775 [0.616, 0.913] | 0.500 [0.227, 0.833] | 0.211 | 0.333 [0.042, 0.595] |
| noisy_defecting | 0.716 [0.620, 0.810] | 0.733 [0.497, 0.949] | 0.600 [0.375, 0.852] | 0.808 [0.613, 0.930] | 0.583 [0.250, 0.856] | 0.211 | 0.375 [0.197, 0.542] |
| partial_fringe | 0.522 [0.348, 0.682] | 0.458 [0.243, 0.684] | 0.433 [0.200, 0.686] | 0.667 [0.418, 0.882] | 0.333 [0.083, 0.606] | 0.211 | 0.208 [0.000, 0.553] |
| price_fixing | 0.854 [0.778, 0.926] | 0.925 [0.859, 1.000] | 0.858 [0.736, 0.961] | 0.808 [0.622, 0.926] | 0.500 [0.250, 0.833] | 0.211 | 0.542 [0.269, 0.803] |
| rotation | 0.815 [0.624, 0.932] | 0.842 [0.425, 1.000] | 0.725 [0.482, 0.886] | 0.867 [0.737, 0.953] | 0.583 [0.310, 0.940] | 0.211 | 0.417 [0.042, 0.845] |
| **macro** | 0.748 [0.699, 0.796] |  |  |  | 0.521 [0.417, 0.689] | | |

| honest set | FA (all-mech model) | worst LOMO FA |
|---|---|---|
| honest@in | 0.033 [0.000, 0.094] | 0.067 [0.000, 0.167] |
| honest@size | 0.233 [0.115, 0.418] | 0.300 [0.158, 0.594] |
| honest@spread | 0.033 [0.000, 0.100] | 0.167 [0.091, 0.309] |

## dep_gbm

criteria (validation): {'C1_lomo_auc>=0.85': False, 'C2_fa<=0.10_every_honest_set': True, 'C3_beats_baselines_ci': False, 'C4_prec10_above_base_ci': False}

| mechanism | LOMO AUC (pooled) | AUC @in | AUC @size | AUC @spread | prec@10% | base | TPR@5%FA |
|---|---|---|---|---|---|---|---|
| bid_suppression | 0.758 [0.599, 0.885] | 0.733 [0.409, 0.970] | 0.808 [0.672, 0.943] | 0.733 [0.458, 0.951] | 0.500 [0.181, 0.690] | 0.211 | 0.292 [0.060, 0.542] |
| complementary | 0.742 [0.643, 0.819] | 0.767 [0.572, 0.880] | 0.708 [0.511, 0.907] | 0.792 [0.601, 0.927] | 0.500 [0.144, 0.667] | 0.211 | 0.250 [0.132, 0.375] |
| cover_bidding | 0.806 [0.702, 0.881] | 0.858 [0.640, 1.000] | 0.825 [0.735, 0.986] | 0.742 [0.405, 0.976] | 0.583 [0.333, 0.773] | 0.211 | 0.292 [0.155, 0.470] |
| intermittent | 0.633 [0.546, 0.715] | 0.508 [0.250, 0.692] | 0.542 [0.367, 0.716] | 0.800 [0.598, 0.905] | 0.333 [0.167, 0.583] | 0.211 | 0.167 [0.072, 0.303] |
| noisy_defecting | 0.610 [0.500, 0.723] | 0.500 [0.281, 0.750] | 0.667 [0.422, 0.811] | 0.692 [0.489, 0.839] | 0.333 [0.083, 0.440] | 0.211 | 0.042 [0.000, 0.136] |
| partial_fringe | 0.524 [0.386, 0.677] | 0.717 [0.477, 0.927] | 0.342 [0.173, 0.582] | 0.442 [0.210, 0.694] | 0.250 [0.083, 0.440] | 0.211 | 0.083 [0.000, 0.261] |
| price_fixing | 0.709 [0.613, 0.792] | 0.750 [0.530, 0.930] | 0.825 [0.580, 0.971] | 0.558 [0.337, 0.730] | 0.417 [0.227, 0.712] | 0.211 | 0.208 [0.000, 0.386] |
| rotation | 0.761 [0.648, 0.841] | 0.742 [0.409, 0.964] | 0.892 [0.723, 1.000] | 0.642 [0.476, 0.849] | 0.583 [0.310, 0.667] | 0.211 | 0.292 [0.144, 0.428] |
| **macro** | 0.693 [0.626, 0.739] |  |  |  | 0.437 [0.330, 0.542] | | |

| honest set | FA (all-mech model) | worst LOMO FA |
|---|---|---|
| honest@in | 0.067 [0.000, 0.167] | 0.067 [0.000, 0.209] |
| honest@size | 0.100 [0.000, 0.276] | 0.167 [0.000, 0.442] |
| honest@spread | 0.033 [0.000, 0.100] | 0.100 [0.024, 0.276] |