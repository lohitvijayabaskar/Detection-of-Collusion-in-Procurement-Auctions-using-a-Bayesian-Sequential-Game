# Literature comparison plan (draft, 2026-11-02)

Goal: compare our detectors with published detectors that use different detection methods, on
one set of metrics. Status "checked" says what was actually read. Nothing here is a result.

## Candidate methods (one per detection method)

| # | Detection method | Paper | Checked | Reported metric | Re-implementable on our data? |
|---|---|---|---|---|---|
| 1 | Variance / coefficient-of-variation screen | Abrantes-Metz, Froeb, Geweke, Taylor (2006, IJIO) | search summary, FTC working paper listed | none (screen) | Yes (our `lit_cv_mean`) |
| 2 | Descriptive-statistic screens (relative distance, skewness, kurtosis) | Imhof (2017 working paper; 2019) | search summary; formulas not seen | none (screen) | Yes (our `lit_rd_mean`, `lit_skew_mean`, `lit_kurt_mean`) |
| 3 | Rank / participation regression | Porter & Zona (1993, JPE) | earlier session, not re-read | test results | Partly: needs costs, which we hide |
| 4 | Independence / exchangeability tests of bids | Bajari & Ye (2003, REStat) | bibliographic record only | test results | Approximately, with covariates |
| 5 | Structural: latent-cost stochastic dominance | Aryal & Gabrielli (2013, IJIO) | abstract | test rejection | Heavy (structural estimation) |
| 6 | Randomisation inference on bidder groups | Conley & Decarolis (2016, AEJ: Micro) | abstract | validated on 8 known groups, no AUC | Yes, with firm identities |
| 7 | Isolated winners / missing close losers | Chassang, Kawai, Nakabayashi, Ortner (2022, Econometrica); Kawai & Nakabayashi (2022, JPE) | abstract | test rejection | Only as a gap screen: needs rebids and secret reserves |
| 8 | Co-bidding network and community detection | Wachs & Kertesz (2019, Sci. Rep.) | full text read | no AUC; compares flagged and ordinary groups on four screens | Yes, across tenders in a market |
| 9 | Screens plus lasso / random forest / ensemble | Huber & Imhof (2019, IJIO) | bibliographic; accuracy not seen | out-of-sample accuracy | Yes (our `lit_gbm` is close) |
| 10 | Graph neural network (graph attention) | Gomes et al. (2024); Imhof, Viklund, Huber (2025, arXiv 2507.12369) | Gomes: earlier session; Imhof: abstract | F1, balanced accuracy; accuracy 80-90 % (91 % cross-market) | Heavy; possible on our graphs |
| 11 | Convolutional network on bid patterns | Huber & Imhof (IJIO, deep-learning flagging paper) | not read (PDF unreadable here); snippet: 19 of 20 firms | firm-level accuracy | Heavy |
| 12 | Unsupervised clustering then classifier | Silveira et al. (2022, Zenodo 7111547) | abstract: 80.25 % on a labelled case | accuracy | Yes (mixture model on market features) |
| 13 | Benford's-law digit test | Bugarin & da Cunha (2015) and later Brazilian / Macedonian studies | search summary only | none (test) | Yes |
| 14 | Isolation forest / local outlier factor / one-class SVM | Brazilian (Paraiba) comparison; Indonesian thesis | search summary only; weak sources | overlap ratio, no labels | Yes |
| 15 | Bayesian sequential screen (our earlier project) | This project, old environment | own code | none | Yes (CUSUM / posterior on bids) |

Same method family, listed but not counted separately: Imhof & Wallimann (2021, coalition
screens with SVM / forest, about 90 % accuracy), Wallimann, Imhof & Huber (2020, subgroup
screens, no numbers on the abstract page), Huber, Imhof & Ishii (2022, JRSS-A, 88-97 % on
Okinawa), Wallimann & Sticher (2023, railway price screens, no numbers on the abstract page).

## Why a common metric is needed

The published papers report accuracy, F1 or balanced accuracy at a chosen cut-off, on different
units (bid, firm, coalition, tender, market) and different data. Only one paper's text gave an
AUC, and from its best run. Numbers cannot be compared across papers as they stand.

## Proposed final metrics (all on our locked benchmark, re-implemented methods)

1. **Primary: macro leave-one-mechanism-out AUC** with 95 % bootstrap CI (same as the main study).
   Methods that need no training (screens, tests, Benford, clustering, anomaly detection) are
   scored directly; trainable ones (9, 10, 11, 12) are trained leave-one-mechanism-out.
2. **Operating-point metrics at the 5 % false-alarm threshold set on the calibration split:**
   true-positive rate, precision@10 %, and balanced accuracy (so published accuracy and F1
   figures can be put alongside).
3. **Worst-mechanism AUC** and false-alarm rate on every honest set (criteria C1 and C2).
4. **Paired differences** to our detector with CIs, so that "beats" needs an interval that
   excludes zero.
5. **Real-data check** (Japan, Italy, Brazil) only for methods that run on tender-level data.

Published numbers go in a separate context table, marked as different protocol and never
mixed with the above.

## Open decisions for the user

- Heavy methods (5, 10, 11): implement, or cite only?
- Methods needing data we do not simulate (3, 7): approximate, or exclude?
- Verify papers 3-5, 9, 11, 13 and 14 at source (full text) before citing; downloading needs approval.
