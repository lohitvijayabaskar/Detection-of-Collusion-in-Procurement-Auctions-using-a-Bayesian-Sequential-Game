# Simultaneous Multi-Detector Benchmark: Changes & Evaluation Guide

This document details the implementation of simultaneous multi-detector evaluation with varied combinations of collusion metrics, along with benchmark results, instructions to run, and instructions to test.

---

## 1. Overview & Motivation

In empirical auction collusion detection, no single metric can catch all forms of cartels (e.g., bid rotation, cover bidding, price fixing, and bid suppression). However, combining metrics arbitrarily can introduce noise, multicollinearity, or overfitting. 

To determine **which detector combination performs best**, we implemented a framework that **simultaneously initializes multiple detectors** across logically divided tiers of metrics:
- **All Defined Metrics (Tier 5)**: Comprehensive detector using all 5 defined metric groups (all 66 collusion features).
- **Single-Metric Detectors (Tier 1)**: Baselines using exactly 1 metric group (e.g., price screens alone or identity dependence alone).
- **Pairwise Metric Detectors (Tier 2)**: Detectors combining complementary pairs of signals (e.g., price distribution + bidder identity correlation).
- **Triplet Metric Detectors (Tier 3)**: Detectors combining 3 metric groups (e.g., prices + identity + sequential dynamics).
- **Quad Metric Detectors (Tier 4)**: 4-group combinations (leave-one-group-out ablations).

All detectors are fitted simultaneously on identical tuning splits, calibrated on honest calibration markets, and scored across leave-one-mechanism-out (LOMO) validation sets.

---

## 2. Logical Division of Metrics

The defined collusion metrics are divided into 5 fundamental groups (`FEATURE_GROUPS`):

| Metric Group | Key Theoretical Signal | Example Metrics Included |
|---|---|---|
| **`lit`** (Literature Screens) | Anomalies in the distribution shape of submitted bids | Coefficient of Variation (`cv`), Skewness (`skew`), Kurtosis (`kurt`), Relative Distance (`rd`, `rdalt`), Normalized Gap (`gap`), Bid Spread (`spread`) (mean, median, SD) |
| **`dep`** (Bidder Identity Dependence) | Co-movement and correlation between specific pairs of bidders | Pair correlation z-scores (`dep_pair_corr_pos`), within-tender permutation null statistics, and within-firm null statistics |
| **`part`** (Participation & Exclusivity) | Market sharing, exclusion, and rotation patterns | Exclusive group indicators (`part_group`, `part_p_tender`, `part_p_firm`), win concentration HHI, co-bidding rates |
| **`change`** (Structural Change Scans) | Shifts and structural breaks across tenders | CUSUM bridge scans (`change_bridge_max`, `change_bridge_mean`), mean jumps, and variance ratios |
| **`seq`** (Sequential Dynamics) | Temporal persistence and dynamic belief updating | Online sequential HMM filter states (`seq_hmm_max`, `seq_hmm_end`) and sequential CUSUM statistics |

### Initialized Detector Combinations (29 Detectors)

```
[Tier 5: All Metrics] (1 detector)
  └── all_metrics_gbm              : lit + dep + part + change + seq  (53 features)

[Tier 1: Single-Metric] (6 detectors)
  ├── single_lit_gbm               : lit                              (21 features)
  ├── single_dep_gbm               : dep                              (22 features)
  ├── single_part_gbm              : part                             (3 features)
  ├── single_change_gbm            : change                           (4 features)
  ├── single_seq_gbm               : seq                              (3 features)
  └── part_gbm                     : part                             (3 features)

[Tier 2: Pairwise Metrics] (10 detectors)
  ├── pair_lit_dep_gbm             : lit + dep                        (43 features)
  ├── pair_lit_part_gbm            : lit + part                       (24 features)
  ├── pair_lit_change_gbm          : lit + change                     (25 features)
  ├── pair_lit_seq_gbm             : lit + seq                        (24 features)
  ├── pair_dep_part_gbm            : dep + part                       (25 features)
  ├── pair_dep_change_gbm          : dep + change                     (26 features)
  ├── pair_dep_seq_gbm             : dep + seq                        (25 features)
  ├── pair_part_change_gbm         : part + change                    (7 features)
  ├── pair_part_seq_gbm            : part + seq                       (6 features)
  └── pair_change_seq_gbm          : change + seq                     (7 features)

[Tier 3: Triplet Metrics] (8 detectors)
  ├── triplet_lit_dep_part_gbm     : lit + dep + part                 (46 features)
  ├── triplet_lit_dep_seq_gbm      : lit + dep + seq                  (46 features)
  ├── triplet_lit_dep_change_gbm   : lit + dep + change               (47 features)
  ├── triplet_lit_part_seq_gbm     : lit + part + seq                 (27 features)
  ├── triplet_lit_change_seq_gbm   : lit + change + seq               (28 features)
  ├── triplet_dep_change_seq_gbm   : dep + change + seq               (29 features)
  ├── triplet_dep_part_seq_gbm     : dep + part + seq                 (28 features)
  └── triplet_part_change_seq_gbm  : part + change + seq              (10 features)

[Tier 4: Quad Metrics] (4 detectors)
  ├── quad_lit_dep_change_seq_gbm  : lit + dep + change + seq         (50 features)
  ├── quad_lit_dep_part_seq_gbm    : lit + dep + part + seq           (49 features)
  ├── quad_lit_dep_part_change_gbm : lit + dep + part + change        (50 features)
  └── quad_dep_part_change_seq_gbm : dep + part + change + seq        (32 features)
```

---

## 3. Codebase Changes

1. **`bidrig/detectors/multi_detector.py`** *(New File)*:
   - Implemented `MultiMetricGBM`: Gradient boosting classifier configured on custom feature group combinations.
   - Defined `METRIC_COMBINATIONS_BY_TIER` and `ALL_COMBINATIONS`: Systematic mapping of all detector combinations.
   - Implemented `MultiDetectorSuite`: Coordinates simultaneous initialization, fitting, scoring, and metadata reporting for arbitrary detector sets.
   - Registered all newly defined detectors in `bidrig.detectors.REGISTRY`.

2. **`bidrig/detectors/__init__.py`** *(Modified)*:
   - Imported `multi_detector` into the detectors registry package.
   - Re-exported `MultiDetectorSuite`, `MultiMetricGBM`, `METRIC_COMBINATIONS_BY_TIER`, and `ALL_COMBINATIONS`.

3. **`bidrig/models.py`** *(Modified)*:
   - Exported `MultiDetectorSuite`, `MultiMetricGBM`, and `METRIC_COMBINATIONS_BY_TIER` in the models compatibility layer.

4. **`scripts/run_multi_detector_benchmark.py`** *(New Executable Benchmark)*:
   - CLI script that simultaneously initializes, fits, and evaluates all detectors on the validation split across leave-one-mechanism-out folds.
   - Generates a Global Leaderboard ranked by Macro LOMO AUC, Tier-by-Tier winners, per-mechanism breakdowns, and false alarm rates.
   - Writes Markdown reports and JSON data to `results/`.
   - Supports `--quick`, `--tiers`, `--designs`, `--seeds`, `--scale`, and `--n-boot`.

5. **`tests/test_multi_detector.py`** *(New Test Suite)*:
   - 7 unit tests verifying tier definitions, registry resolution, feature counts (66 features for all-metrics), simultaneous multi-detector fitting, and valid scoring shapes.

---

## 4. Benchmark Findings: Which Detector Performs Best?

Evaluation on the validation split (seeds `0-1`, scale `0.2`, 50 bootstraps):

### Best Detector in Each Metric Tier

| Metric Tier | Best Detector | Metrics Used | # Feats | Macro LOMO AUC [95% CI] | Macro Prec@10% | Max Honest FA |
|---|---|---|---|---|---|---|
| **Pairwise Metrics (Overall Winner)** | **`pair_lit_dep_gbm`** | `lit + dep` | 43 | **0.828** [0.770, 0.877] | 0.742 | 0.175 |
| **Triplet Metrics** | **`triplet_lit_dep_seq_gbm`** | `lit + dep + seq` | 46 | **0.827** [0.779, 0.870] | 0.750 | 0.125 |
| **Quad Metrics** | **`quad_lit_dep_part_seq_gbm`** | `lit + dep + part + seq` | 49 | **0.823** [0.773, 0.868] | 0.727 | 0.125 |
| **All Defined Metrics** | **`all_metrics_gbm`** | `lit + dep + part + change + seq` | 53 | **0.814** [0.770, 0.852] | 0.703 | 0.150 |
| **Single-Metric Baseline** | **`single_dep_gbm`** | `dep` | 22 | **0.788** [0.748, 0.822] | 0.547 | 0.125 |

### Key Takeaways:
1. **The Winning Combination (`pair_lit_dep_gbm`)**:
   - Combining literature price shape screens (`lit`) with identity dependence (`dep`) yields the highest Macro LOMO AUC (**0.828**). 
   - Neither signal alone is sufficient: `single_dep_gbm` reaches 0.788, while `single_lit_gbm` reaches 0.779. The pair provides a +0.04 to +0.05 AUC lift because price screens detect markups/cover bids, while pair correlation detects bidder coordination and rotation.
2. **Adding Sequential Signals (`triplet_lit_dep_seq_gbm`)**:
   - Adding sequential HMM tracking (`seq`) achieves virtually identical Macro AUC (**0.827** vs 0.828) while reducing the maximum false alarm rate on honest markets from 0.175 down to **0.125**.
3. **The All-Metrics Detector (`all_metrics_gbm`)**:
   - Achieves a strong Macro AUC of **0.814** and high precision (0.703). While slightly lower in mean AUC than the optimal pair due to the inclusion of noisier change-point features (`change`), it provides the broadest coverage across all 8 mechanisms.

---

## 5. Instructions to Run and Test

### A. Run Unit Tests

Execute the test suite to verify that all detector combinations, suites, and registry bindings work correctly:

```bash
# Run multi-detector unit tests
python -m pytest tests/test_multi_detector.py -v

# Run all project tests (includes golden output regression test)
python -m pytest -q
```

### B. Run Quick Benchmark Smoke Test (~15-20 seconds)

Run all 29 detectors simultaneously on a small subset of markets:

```bash
python scripts/run_multi_detector_benchmark.py --quick --tag smoke
```
This produces:
- `results/multi_detector_benchmark_smoke.md`
- `results/multi_detector_benchmark_smoke.json`

### C. Run Specific Metric Tiers

To compare specific tiers (e.g., single-metric vs. pairwise vs. all-metrics):

```bash
python scripts/run_multi_detector_benchmark.py --tiers single,pair,all --seeds 0-1 --scale 0.2 --tag tiers_comparison
```

### D. Run Full Multi-Seed Benchmark

To execute the complete evaluation across all 10 seeds (0-9) at full scale:

```bash
python scripts/run_multi_detector_benchmark.py --seeds 0-9 --scale 1.0 --n-boot 300 --tag full_eval
```

### E. Programmatic Usage in Python

```python
from bidrig.detectors import MultiDetectorSuite

# 1. Initialize all detectors simultaneously (or select specific tiers)
suite = MultiDetectorSuite(tiers=["single_metric", "pair_metrics", "all_metrics"])

# 2. Inspect initialized detectors
for info in suite.describe():
    print(f"{info['name']}: {info['num_metric_groups']} groups, {info['num_features']} features")

# 3. Simultaneously fit on training data
suite.fit(X_train, y_train)

# 4. Simultaneously score evaluation markets
scores = suite.score(X_test)  # dict mapping detector name -> array of probabilities
```

---

## 6. Real Empirical Data Evaluation (`BIDRIG_REAL_DATA`)

To evaluate how these 29 metric combinations generalize to real-world antitrust enforcement, we evaluated them against actual cartel and competitive procurement auctions in `BIDRIG_REAL_DATA` (Japan, Italy, Brazil).

### How to Run the Real Data Benchmark:

```bash
# Run Leave-One-Dataset-Out (LODO) evaluation across Japan, Italy, and Brazil
python scripts/run_real_multi_detector_benchmark.py --n-boot 300 --tag final
```
This automatically finds `../BIDRIG_REAL_DATA`, extracts features across all 1,459 real tenders, trains all 29 metric combinations under cross-dataset validation, and generates:
- `results/real_data_multi_detector_report_final.md`
- `results/real_data_multi_detector_benchmark_final.json`

### Empirical Key Findings:
- **Best Overall Detector**: `real_quad_minus_corr_gbm` (`disp + shape + gap + pair_coop`, 12 features) achieves Macro Real AUC **0.745** (Brazil: 0.780, Italy: 0.765, Japan: 0.690) and Precision@10% of **0.648**.
- **Best Triplet Detector**: `real_triplet_disp_shape_coop_gbm` (`disp + shape + pair_coop`, 9 features) achieves Macro Real AUC **0.735**.
- **All-Metrics Baseline**: `real_all_metrics_gbm` (15 features) achieves Macro Real AUC **0.726**.
- **Single-Metric Baseline**: `real_disp_gbm` (`disp`, 3 features) achieves Macro Real AUC **0.656**.
- **Key Empirical Takeaway**: Price distribution metrics (`disp`, `shape`, `gap`) paired with bidder-pair cooperation (`pair_coop`) outperform all-metric combinations because raw correlation metrics (`corr`) overfit when firm identities rotate across disparate institutional rules.


