# Experiment log

Each entry: date, hypothesis, change, command, outcome, failures. Locked-test runs are marked
**LOCKED** and also recorded in `results/locked_runs.jsonl`.

## 2026-10-21 E0: setup and pre-registered protocol

Lessons carried over from the old repo (`docs/experiment_log.md` there):
- A price-level screen flagged honest markups as collusion. Here every identity feature uses
  residuals relative to the tender median, which removes uniform markups and the estimate.
- Calibration that did not match the simulator's start state gave a 70 % false-alarm rate.
  Here thresholds are set on the `calib` split, drawn from the same generator as `tuning`.
- An AUC rank bug once printed all zeros. Here every metric has separated, reversed and tied
  tests (`tests/test_metrics.py`).
- Background scripts with syntax errors were launched unchecked. Scripts are syntax-checked
  with `py_compile` and logs are checked after launch.

Protocol fixed before any validation numbers were seen:
- Splits: `tuning` (400 honest + 50 per mechanism), `calib` (300 honest, same spec), and
  `validation` (honest and each mechanism at in-distribution, sizes {8, 9, 23, 24} and
  spreads [0.04, 0.06] ∪ [0.18, 0.22]; 100 honest / 25 collusive per set). `test`, which is
  locked, has in-distribution sets, unseen sizes {6, 7, 25..30}, unseen spreads
  [0.02, 0.04] ∪ [0.22, 0.30], three honest stress sets (markup 20-25 %, capacity 0.15-0.30,
  noisy estimates and costs) and the uniform-markup control. It has 100 honest / 40 collusive
  per set. Tuning range: 10-22 firms, spread 0.06-0.18, 20-60 tenders.
- Seeds 0-9. CIs use a two-level bootstrap (seeds, then markets within sets), paired across
  designs.
- C1: for each held-out mechanism, LOMO AUC ≥ 0.85, both pooled against all honest test sets
  and per condition (in / size / spread) against honest markets of the same condition.
- C2: the alarm rate at the threshold set for 5 % false alarms on `calib`, using the model
  trained on all mechanisms, is ≤ 0.10 on every honest test set.
- C3: the paired difference in macro LOMO AUC against every baseline (lit GBM and seven
  single screens) has its 95 % CI lower bound above 0.
- C4: for every mechanism, the lower CI bound of precision@10 % (held-out mechanism plus all
  honest markets) is above the base rate (1/6 on test).

Checks: 35 tests pass; `check_determinism.py` gives identical hashes for 1 vs 4 workers and
across processes, and different hashes for seed+1.

Smoke run (not evidence; 2 seeds, scale 0.15, 30 bootstraps):
`run_validation.py --tag smoke --designs full_gbm,dep_gbm --seeds 0-1 --scale 0.15`.
Macro LOMO AUC: full_gbm 0.748, dep_gbm 0.693, lit_gbm 0.739. Dependence features alone are
weaker than the literature GBM at this scale.

## 2026-10-21 V1: first validation run (feature v1, tender null only)

- Command: `run_validation.py --tag v1 --designs full_gbm,knowledge_gbm,dep_gbm,change_gbm,seq_gbm,full_minus_dep,full_minus_change,full_minus_seq` (seeds 0-9, 300 bootstraps). Output `results/validation_v1.{json,md}`.
- Macro LOMO AUC [95 % CI]: full_gbm 0.847 [0.840, 0.855]; lit_gbm 0.806 [0.795, 0.816]; dep_gbm 0.771; knowledge_gbm 0.770; change_gbm 0.494; seq_gbm 0.549; best single screen (rdalt) 0.692.
- full_gbm per mechanism (pooled): rotation 0.946, price_fixing 0.945, cover 0.932, complementary 0.930, intermittent 0.833, noisy_defecting 0.784, bid_suppression 0.742, partial_fringe 0.661.
- FA at the calib 5 % threshold (all-mechanism model): in 0.053, size 0.096, spread 0.061.
- Validation criteria for full_gbm: C1 fail (4 mechanisms < 0.85), C2 pass, C3 pass, C4 pass.
- Diagnosis (univariate AUCs, tuning seed 0): many dependence z-scores are below 0.5 because the tender null also destroys honest persistent-efficiency structure, so honest markets get large z-scores too. The change and seq groups carry almost no signal alone.
- Process note: the bootstrap took about 50 minutes because macro and difference statistics recomputed every AUC; a per-resample memo was added to `evaluate` (no change to values).

## 2026-10-21 I1 (workstream: dependence features): time null added to the tender null (feature v2)

- Hypothesis: standardising identity statistics against a null that shuffles each firm's residuals across its tenders keeps honest persistent-efficiency structure in the null, so collusive coordination stands out more. Falsifier: macro LOMO AUC of dep_gbm not above v1's CI.
- Change: `permute_within_firms`; every identity statistic is reported against both nulls (`dep_<k>` tender, `dep_<k>_t` time; firm dispersion raw under the time null); new `close_pair` statistic (top-3 pair share of co-bids within 3 %). The time null alone (first try, tuning seed 0 univariate) removed the win/loss signal (cover 0.82 → 0.52), hence both nulls.
- Command: `build_features.py --seeds 0-9`, then `run_validation.py --tag v2 ...` (same designs).
- Outcome: dep_gbm 0.771 → 0.826 [0.814, 0.836]; full_gbm 0.847 → 0.862 [0.853, 0.871]; knowledge_gbm 0.770 → 0.822. full_gbm per mechanism: price_fixing 0.972, complementary 0.957, rotation 0.951, cover 0.945, intermittent 0.856, noisy_defecting 0.794, bid_suppression 0.751, partial_fringe 0.668. FA in/size/spread 0.051/0.073/0.053. Validation criteria: C1 fail, C2-C4 pass. KEPT.
- Ablation: full_minus_change 0.864, full_minus_seq 0.862, full_minus_dep 0.801. The change and seq groups add nothing measurable; dep adds about 0.06.

## 2026-10-21 I2 (workstream: detector): participation features + knowledge tail score fused with the GBM (feature v3)

- Hypothesis: under LOMO a held-out mechanism's unique signal (e.g. bid suppression's missing bids) cannot be learned from the other mechanisms. A tail score with label-free, knowledge-fixed directions (`TAIL_DIRECTIONS`), fused with the GBM through honest p-values (`Fused`, score = -log min(p_gbm, p_tail)), should raise the worst mechanisms. Falsifier: no gain in min-mechanism or macro LOMO AUC beyond the CI.
- Change: `part` group (`part_rate`, `part_low`, `part_group` greedy mutual-exclusion group); `TailScore`, `Fused`; designs full_part_gbm, lit_dep_part_gbm, tail_only, full_fused, lit_dep_part_fused.
- Command: `build_features.py --seeds 0-9` then `run_validation.py --tag v3 --designs full_gbm,full_part_gbm,lit_dep_part_gbm,tail_only,full_fused,lit_dep_part_fused`.
- Outcome (macro LOMO AUC / min mechanism): full_gbm 0.862 / 0.668; full_part_gbm 0.863 / 0.677; lit_dep_part_gbm 0.866 [0.859, 0.874] / 0.682; tail_only 0.774 / 0.600; full_fused 0.864 / 0.687; lit_dep_part_fused 0.866 / 0.688. full_fused raised bid_suppression 0.751 → 0.811 but lowered noisy_defecting 0.794 → 0.767 and intermittent 0.856 → 0.836. All CIs overlap full_gbm's. FA ≤ 0.065 everywhere.
- Decision: NO validation gain. The code stays (superset features, extra designs) but the reference design remains full_gbm. No-gain count for the detector workstream: 1.

## 2026-10-21 I3 (detector): pair-maximum statistics for small cartels (prototype, falsified before implementation)

- Hypothesis: a top-3 mean dilutes a 2-3 firm cartel's single pair; the maximum pair statistic (time null) should separate partial_fringe better.
- Command: scratch prototype on 120 fresh tuning-spec markets per mechanism (seed 77, split label "proto", not a benchmark split).
- Outcome: univariate AUC vs honest: pair-correlation max 0.732 (partial_fringe), 0.448 (noisy), 0.457 (suppression); "both above median and within 10 %" pair max 0.613 / 0.405 / 0.398. Top-3 pair correlation already gives about 0.75 on partial_fringe (v2 diagnostic). No gain, not implemented. No-gain count: 2.

## 2026-10-21 I4 (detector): monotone GBM constraints along the knowledge directions

- Hypothesis: constraining the GBM to be monotone in `TAIL_DIRECTIONS` stops it fitting mechanism-specific non-monotone patterns, so held-out mechanisms transfer better. Falsifier: no gain beyond the CI.
- Command: `run_validation.py --tag v3mono --designs full_gbm,lit_dep_part_gbm,full_part_mono,lit_dep_part_mono`.
- Outcome (macro / min mechanism): full_gbm 0.862 / 0.668; lit_dep_part_gbm 0.866 / 0.682; full_part_mono 0.865 / 0.674; lit_dep_part_mono 0.867 [0.859, 0.874] / 0.678. Per mechanism (lit_dep_part_mono): noisy_defecting 0.806, bid_suppression 0.760, partial_fringe 0.674. No significant gain.
- No-gain count for the detector workstream: 3 (I2, I3, I4). STOP RULE reached for this workstream.

## 2026-10-21 Design freeze

- On validation, C1 is unmet for partial_fringe (0.67-0.69 across designs), bid_suppression (0.75-0.81) and noisy_defecting (0.77-0.81). C2, C3 and C4 pass on validation for full_gbm.
- Freeze rule (keep a change only if validation improves): I1 is the only kept change. The frozen design is **full_gbm** (groups lit, dep, change, seq; GBM_PARAMS as committed; feature v3, whose groups are identical to v2's for these columns). Only this design gets a locked run; the others are not run on test, so test results cannot be used to choose among them.

## 2026-10-21 LOCKED run 1: full_gbm (hash 6cc622540eac3929)

- Command: `build_features.py --seeds 0-9 --splits test`, then `run_locked_test.py --design full_gbm --note "frozen after validation v1-v3mono; only kept change I1 (time null)"` (seeds 0-9, 1000 bootstraps, 551 s). Files: `results/locked_1_full_gbm.{json,md}`, `results/locked_runs.jsonl`.
- Macro LOMO AUC 0.870 [0.862, 0.878]; lit_gbm 0.803 [0.792, 0.812]; difference +0.067 [0.060, 0.074]. Against the single screens: +0.196 (rdalt) to +0.382 (gap), all CIs above 0.
- Per held-out mechanism, pooled AUC (worst of in / size / spread): rotation 0.961 (0.939), price_fixing 0.971 (0.963), complementary 0.971 (0.947), cover_bidding 0.946 (0.921), intermittent 0.871 (0.849, spread), noisy_defecting 0.823 (0.775), bid_suppression 0.744 (0.677), partial_fringe 0.672 (0.621).
- FA, all-mechanism model, calib 5 % threshold: in 0.053, size 0.075, spread 0.072, markup_high 0.043, capacity 0.048, noisy 0.013. Worst LOMO-model FA: 0.116 (size).
- Precision@10 % (base rate 0.167): 0.378 [0.335, 0.418] (partial_fringe) to 0.983 (price_fixing); macro 0.766.
- Uniform-markup control: AUC 0.478 [0.450, 0.503], alarm 0.047. Undetectable by design, as expected.
- Criteria: C1 FAIL (4 of 8 mechanisms below 0.85 in at least one condition; intermittent misses by 0.001 at spread), C2 PASS, C3 PASS, C4 PASS.
- No further locked runs. Another design would need a new freeze based on validation only.

## 2026-10-21 Workstream W2: within-market change detection (targets C1 for intermittent: 0.849 at unseen spreads on the locked test)

Prototypes use 150 fresh markets per mechanism (seed 78, split label "proto"; half in-distribution, half with cost spread 0.18-0.22). Univariate AUC vs honest; the reference is the existing `chg_res_sd` bridge scan.

- W2-1. Hypothesis: a bridge scan on the per-tender closeness of the market's top-5 close pairs jumps inside the cartel window. Result: intermittent 0.731 vs reference 0.794. Falsified; not implemented. No-gain count 1.
- W2-2. Hypothesis: the top-3 close-pair share recomputed in contiguous segments (halves, thirds, middle half), minus its full-history value, concentrates an intermittent cartel. Result: 0.682 vs 0.794. Falsified; not implemented. No-gain count 2.
- W2-3. Hypothesis: an intermittent cartel is an epidemic change (start and stop), so the epidemic scan max |Σz|/√(L(1−L/n)) beats the single-change bridge scan. Prototype: res_sd 0.818 vs 0.794; close 0.552 vs 0.555; gap 0.563 vs 0.581. Implemented as `chg_epi_*` (feature v4, test against brute force) and sent to full validation (`--tag v4`).
- W2-3 validation (`run_validation.py --tag v4 --designs full_gbm,change_gbm,seq_gbm,knowledge_gbm,full_minus_change`, feature v4): full_gbm macro 0.861 [0.852, 0.870] (v3: 0.862); intermittent 0.856 (v3: 0.856); change_gbm 0.500 (v3: 0.494); full_minus_change 0.864. No gain. REVERTED (`bidrig/features.py`, `bidrig/datasets.py` and the test restored to feature v3; `results/validation_v4.*` kept). No-gain count 3. STOP RULE reached for W2.

## 2026-10-21 Final status

- Both workstreams that target the unmet criterion C1 (detector features and within-market change detection) ended by the three-iterations-without-gain rule. Making the simulator's mechanisms easier to detect would change the benchmark to fit the detector, so it was not done. Real data needs a labelled per-bid dataset from the user (rule 5).
- Locked result stands as run 1 (full_gbm): C1 not met (documented as a limitation), C2-C4 met. The source after the revert matches the locked design's source.

## 2026-10-22 D1: real data source (user approved the download)

- Source: supplementary file of García Rodríguez, Rodríguez-Montequín, Ballesteros-Pérez, Love, Signor (2022), "Collusion detection in public procurement auctions with machine learning algorithms", Automation in Construction 133, 104047. URL `https://ars.els-cdn.com/content/image/1-s2.0-S0926580521004982-mmc2.zip`, 1,530,786 bytes, sha256 bcf4bc280227326bfa3e761c4163173fd29c5452c37d65d08269a00a5fa209f0.
- Stored outside the repo at `C:/Users/Lohit Vijayabaskar/real_data/rodriguez2022/` (`raw/` zip; `data/` extracted CSVs and README). The bundled `Collusion_Detection_Code.py` was not extracted or run. The data is not committed (licence not checked).
- Inspection (`python -I`, read-only):

| dataset | bids | tenders | median bids/tender | firm IDs | pre-tender estimate | per-bid member label (original) | tenders mixing members and non-members |
|---|---|---|---|---|---|---|---|
| Japan (Okinawa) | 13,515 | 1,080 | 12 | yes, 1,665 firms (median 6 tenders per firm) | yes | 8 % of bids | 216 |
| Italy (Turin) | 20,286 | 278 | 70 | yes, 821 firms (median 11) | yes | 40 % | 275 |
| Brazil (Petrobras) | 683 | 101 | 5 | yes, 272 firms (median 1) | yes | 19 % | 6 |
| USA | 7,004 | 3,754 | 2 | yes, 120 firms (median 9) | no | 12 % | 252 |
| Switzerland GR / See-Gaster | 21,231 | 4,344 | 4 | no | no | 59 % | 1,717 |
| Switzerland Ticino | 1,629 | 224 | 7 | no (consortium only) | no | tender label only | 0 |

- `Collusive_competitor` (the paper's thresholded label) is constant within every tender, so it is a tender label. `Collusive_competitor_original` is per bid (cartel membership).
- Fit to this project: Japan, Italy and Brazil have everything the detector uses (bids, persistent firm IDs, an estimate). The USA lacks an estimate, and the Swiss sets lack firm IDs (literature screens only).
- Gaps: no cartel identifier within a dataset. Italy's `Cartel_name` is 0/1, which looks like a membership flag, not a cartel identity; this is unverified. Leave-one-cartel-out therefore becomes leave-one-dataset-out. The data has no natural "market" unit of repeated tenders, so one has to be defined (e.g. windows of consecutive tenders), and that choice changes what the market-level score means. Date values in Brazil include a sentinel (-922337203).

## 2026-10-22 D2: real-data loader and the window rule (descriptive only; no detector run)

- Change: `bidrig/realdata.py` (`load`, `standardise`, `make_windows`) and `tests/test_realdata.py` (4 tests; the real-file test skips when the data is absent).
- Facts found while building it:
  - Italy is an average-bid auction: the winner is never the lowest bid (0 of 276 tenders), and `Bid_value` = 100 × PTE × (1 − rebate/100) (ratio median 1.0000). The first-price, winner-based features (cover tracking, win/loss, repeat wins, gap screens) are misspecified for Italy.
  - Japan: the lowest bid wins in 84 % of tenders (consistent with a price floor); tender id order equals date order.
  - Brazil: the lowest bid always wins; id order and date order disagree (Spearman −0.245), and 2 dates are invalid.
- Requested rule, consecutive non-overlapping windows (descriptive counts):

| dataset | W | windows | collusive share ≥ 0.5 | ≤ 0.1 | between | median firms per window | median firms with ≥ 6 bids in window |
|---|---|---|---|---|---|---|---|
| Japan | 20 / 30 / 40 | 54 / 36 / 27 | 0 / 0 / 0 | 32 / 21 / 16 | 22 / 15 / 11 | 204 / 290 / 368 | 0 / 0 / 0 |
| Italy | 20 / 30 / 40 | 13 / 9 / 6 | 7 / 5 / 3 | 1 / 1 / 0 | 5 / 3 / 3 | 244 / 268 / 315 | 102 / 125 / 167 |
| Brazil | 20 / 30 / 40 | 5 / 3 / 2 | 1 / 1 / 0 | 2 / 1 / 1 | 2 / 1 / 1 | 81 / 112 / 144 | 2 / 6 / 10 |

- Consequence: Japan has no collusive window at any size and no firm bids ≥ 6 times in a window, so the identity features are undefined and Japan's held-out AUC cannot be computed. Brazil gives 3-5 windows. A market-level leave-one-dataset-out evaluation is not feasible with consecutive windows.
- Alternative checked: label-free average-linkage clustering of tenders on bidder-set Jaccard distance. Japan: 2 clusters of ≥ 20 tenders (48 tenders, both non-collusive); Brazil: none; Italy (cut 0.6): 7 clusters, 176 tenders, collusive shares 0.59, 1.0, 1.0, 0.4, 0.08, 0.0, 0.0. Only Italy has real repeated markets.
- Status: waiting for the user's choice of real-data unit before any evaluation runs.

## 2026-10-22 D3: tender-level LODO design proposed (user chose option 2); not run

- Change: `bidrig/realeval.py` (screen and leave-tender-out pair features, percentile scaling, LODO, block bootstrap), `tests/test_realeval.py` (6 tests: brute-force check of the leave-tender-out statistics, label independence, scaling, bootstrap, end-to-end smoke on synthetic data), `scripts/run_real_lodo.py` (single run, refuses repeats by hash).
- Design: `docs/real_data_design.md`, hash 1cec4feb56110f6c. Waiting for the user's approval before the one run.

## 2026-10-22 REAL run 1: tender-level LODO (hash 1cec4feb56110f6c, approved by the user, run once)

- Command: `run_real_lodo.py --note "approved by user 2026-10-22 as proposed (hash 1cec4feb56110f6c)"` (33 s; 1000 block-bootstrap resamples). Files: `results/real_lodo_1.{json,md}`, `results/real_runs.jsonl`.
- Pre-registered main question: does real_full_gbm beat real_lit_gbm on each held-out dataset (paired AUC CI above 0)?

| held out | full AUC | lit AUC | full − lit | full FA | full prec@10 % (base) | best single screen |
|---|---|---|---|---|---|---|
| Italy (278, 143 collusive) | 0.779 [0.668, 0.876] | 0.635 [0.492, 0.770] | **+0.144 [0.078, 0.234]** | 0.007 | 0.929 [0.749, 1.000] (0.514) | spread 0.676 |
| Japan (1080, 123) | 0.691 [0.627, 0.758] | 0.674 [0.606, 0.737] | +0.018 [−0.032, 0.069] | **0.265** | 0.259 [0.148, 0.380] (0.114) | CV 0.739 |
| Brazil (101, 33) | 0.742 [0.661, 0.850] | 0.755 [0.623, 0.849] | −0.013 [−0.109, 0.177] | 0.088 | 0.500 [0.200, 0.800] (0.327) | CV 0.873 |

- Answer: yes for Italy only. Japan: no significant difference. Brazil: no significant difference vs lit_gbm, and real_full_gbm is worse than the CV screen alone (−0.131 [−0.247, −0.013]).
- False alarms at the transferred 5 % threshold: Italy 0.007, Brazil 0.088, Japan 0.265 (lit_gbm on Japan: 0.212). The threshold does not transfer to Japan.
- Precision@10 % CI above the base rate: Italy and Japan yes; Brazil no.
- Pair features alone (real_pair_gbm): 0.679 / 0.630 / 0.685 (Italy / Japan / Brazil). Adding them to the screens helps where repeated bidders exist (Italy, the only dataset with repeated markets in D2) and not elsewhere.
- Single CV screen: 0.630 / 0.739 / 0.873. A single screen beats every learned model on Japan and Brazil, so the learned models do not transfer across countries better than the simplest screen.
- No further real-data runs under this hash. Any change needs a new proposal and the user's approval.

## 2026-11-01 VAR-1: variant `previous` (full_gbm minus the within-firm null columns `_t`), validation

- Goal and hypothesis: `previous` = same model and feature groups as full_gbm with every column ending in `_t` removed. Hypothesis: the second null is what raised validation AUC in I1, so removing it should lower the result. Falsifier: the paired difference to full_gbm lies inside its CI.
- Change: new package `bidrig/detectors/` (`base.py` interface, registry and `GBMOn`; `previous.py`; `__init__.py`), a lazy registry lookup at the end of `models.make_design` (built-in names resolve first, so existing designs are unchanged), `tests/test_detectors.py` (9 tests), `scripts/run_variant_validation.py` (variant vs reference on the validation split only; `protocol.py` and `metrics.py` untouched).
- Tests: 57 pass (full suite), black and isort clean. Smoke (`--seeds 0-1 --scale 0.15 --n-boot 30`): previous 0.771 vs full_gbm 0.789, paired -0.018 [-0.042, 0.004]; too small to conclude.
- Command: `cd scripts && python run_variant_validation.py --variant previous --tag previous_v1 --seeds 0-9 --n-boot 300`. Files: `results/variant_previous_v1.{json,md}`, `results/logs/variant_previous_v1.log`. Feature version 3, cached tables.
- Outcome (validation, 10 seeds, 95 % CIs):
  - macro LOMO AUC: previous 0.848 [0.839, 0.855]; full_gbm 0.862 [0.853, 0.871]. Paired difference (previous - full_gbm) -0.014 [-0.019, -0.008]; the CI excludes 0.
  - macro precision@10 %: previous 0.788 [0.764, 0.805]; full_gbm 0.810 [0.794, 0.826].
  - Per scheme, paired differences with CIs excluding 0 (previous lower): complementary -0.021, price_fixing -0.019, partial_fringe -0.021, intermittent -0.015. CIs including 0: bid_suppression, cover_bidding, noisy_defecting, rotation. No scheme improved.
  - False alarms at the calibration threshold are indistinguishable: in 0.049 vs 0.051, size 0.075 vs 0.073, spread 0.051 vs 0.053.
- Result for the hypothesis: NOT falsified. The second null contributes, but the effect on the full model is small (about 0.014 macro AUC). The earlier large I1 gain (0.771 to 0.826) was for the dependence-only model; the full model already carried most of that signal through the literature screens.
- Correction to the stated expectation: the brief said removing the second null should give "about 0.77". That figure is the dependence-only model. For the full model the expected value is the v1 result (0.847), and the measured 0.848 matches it.
- Decision: the code is KEPT. Rule 8 (keep only a change that improves on the reference) does not fit an ablation variant whose job is to be compared with the reference; read literally it would revert `previous` because it is worse. The user should confirm this reading.
- Open point for freezing: `scripts/run_locked_test.py` hashes `bidrig/*.py` only, so edits to `bidrig/detectors/*.py` would not change a frozen design's hash. Before any locked run, the hash should include `bidrig/detectors/*.py` (an edit to the locked-run script, so it needs the user's approval).
- No locked run was made. Next: user approval to freeze `previous` (after the hash fix), then one locked run.

## 2026-11-01 FREEZE-1: design `previous` frozen (hash fix approved by the user)

- User approved: extend the design hash to include `bidrig/detectors/*.py`, then freeze `previous`. The locked run was NOT approved and was not made.
- Change: `scripts/run_locked_test.py` `design_hash` now hashes `bidrig/*.py` and `bidrig/detectors/*.py` (docstring updated); `tests/test_locked_hash.py` (3 tests: a detector-file edit changes the hash, a package-file edit and a different design name change it, and it works with no detectors folder). 60 tests pass; black and isort clean.
- Frozen hash for `previous`: 90b0cbe63f309f0e. Design: `docs/frozen_design_previous.md`. Git tag `freeze-previous`.
- Found while checking: the `full_gbm` hash recorded in locked run 1 (6cc622540eac3929) no longer matches `design_hash("full_gbm")` (4a954c909acec0c8) because source files changed after that run. The hash-based one-run guard would not refuse a repeat `full_gbm` locked run. A name-based guard would close this; not implemented (outside the approved edit). The locked test results for `full_gbm` are unaffected: that run used the earlier source.

## 2026-11-01 LOCKED run 2: previous (hash 90b0cbe63f309f0e, approved by the user, run once)

- Command: `cd scripts && python run_locked_test.py --design previous --note "frozen design previous (hash 90b0cbe63f309f0e), user approved the locked run 2026-11-01"` (seeds 0-9, 1000 bootstraps, 969 s). Files: `results/locked_2_previous.{json,md}`, `results/locked_runs.jsonl`, `results/logs/locked_previous.log`. The hash matched the freeze before the run.
- Macro LOMO AUC 0.856 [0.846, 0.866]; lit_gbm 0.803 [0.792, 0.812]; difference +0.053 [0.045, 0.060]. Against the single screens: +0.182 (rdalt) to +0.368 (gap), all CIs above 0. Macro precision@10 % 0.738 [0.719, 0.758].
- Criteria: C1 FAIL (same four schemes below 0.85 in at least one condition), C2 PASS (highest false-alarm rate 0.096 [0.066, 0.130] on unseen sizes; the point estimate is under 10 %, the interval is not), C3 PASS, C4 PASS.
- Per scheme, pooled AUC (worst condition), next to locked run 1 `full_gbm`: bid_suppression 0.728 (0.650) vs 0.744; complementary 0.955 (0.921) vs 0.971; cover_bidding 0.943 (0.916) vs 0.946; intermittent 0.850 (0.796 at unseen spreads) vs 0.871 (0.849); noisy_defecting 0.798 (0.729) vs 0.823; partial_fringe 0.664 (0.614) vs 0.672; price_fixing 0.956 (0.935) vs 0.971; rotation 0.952 (0.920) vs 0.961. The point estimate is lower than `full_gbm` for every scheme.
- False alarms, all-mechanism model: in 0.046, size 0.096, spread 0.079, markup_high 0.052, capacity 0.043, noisy 0.016. `full_gbm` run 1 had size 0.075 and spread 0.072. The worst leave-one-out models reach 0.123 (size) and 0.115 (spread).
- Uniform-markup control: AUC 0.487 [0.456, 0.515], alarm 0.044. Undetectable, as expected.
- Reading: consistent with validation (previous lower than `full_gbm` by about 0.014 macro AUC). The unpaired locked-test CIs overlap slightly (previous [0.846, 0.866], `full_gbm` [0.862, 0.878]), so they do not by themselves separate the two; a paired locked comparison would need `full_gbm` scored again on the test split, which is a repeat locked run and was not done.
- No further locked runs for this design.

## 2026-11-01 VAR-2 (pre-registered before any run): variant `stat_auc`, validation-tuned logistic model

- Goal: a plain logistic regression on standardised features, with the feature subset and penalty chosen to maximise validation AUC.
- Hypothesis: a simple linear model can match `full_gbm`, which would mean the boosting step adds little.
- Design: median imputation then standardisation (both fitted on the tuning split only), L2 logistic regression (lbfgs, max_iter 2000). Grid: subsets dep, lit_dep, lit_dep_seq, all (groups of FEATURE_GROUPS) x penalty C in {0.01, 0.1, 1, 10} = 16 configurations.
- Selection rule: the configuration with the highest macro leave-one-scheme-out AUC on the VALIDATION split, seeds 0-9, computed by the existing protocol (`score_split`, `evaluate`); ties go to the first configuration in grid order. The test split is never read during selection. The choice is made once by `scripts/select_stat_auc.py` and then frozen as constants in `bidrig/detectors/stat_auc.py` (so it is part of the design hash). `protocol.py` and `metrics.py` are untouched.
- Falsifier: at validation, `stat_auc` is below `full_gbm` by more than the paired CI. At the locked stage the criterion from the loop prompt applies: the locked-test AUC within the paired CI of `full_gbm`; a larger gap in either direction is reported as is.
- Caveat (mandatory): the settings are chosen by validation AUC and the same validation data then reports the result, so the validation number is optimistic and must be labelled "validation-tuned". The locked test is the unbiased estimate.
- Interpretation rule for step F, fixed in advance: as with `previous`, this is a comparison variant, so rule 8 is read as "keep the code; report whether the hypothesis survived".

## 2026-11-01 VAR-2 result: `stat_auc` (validation-tuned), selection and validation

- Selection (validation only, seeds 0-9, 16 configurations; `scripts/select_stat_auc.py --tag stat_auc_select_v1`; files `results/stat_auc_selection_stat_auc_select_v1.{json,md}`, log `results/logs/stat_auc_select_v1.log`): chosen `lit_dep`, C = 0.1, macro LOMO AUC 0.8639. Neighbours: lit_dep_seq 0.1 -> 0.8626, all 0.1 -> 0.8595, lit_dep 0.01 -> 0.8621, lit_dep 1 -> 0.8601; the best `dep`-only configuration reaches 0.8166. The optimum is an interior point of the grid, not a boundary. The values are point estimates (20 bootstrap resamples, unused); differences between the top configurations (about 0.001-0.004) are too small to read as meaningful.
- Frozen into `bidrig/detectors/stat_auc.py` as `CHOSEN = ("lit_dep", 0.1)` (43 features: lit and dep groups). A test ties the constant to the selection record.
- Tests: 69 pass (full suite); black and isort clean. Smoke selection (`--seeds 0-1 --scale 0.15`) ran end to end; its choice (all, 0.01) was a plumbing check and is not used.
- Validation vs reference (`cd scripts && python run_variant_validation.py --variant stat_auc --tag stat_auc_v1 --seeds 0-9 --n-boot 300`; files `results/variant_stat_auc_v1.{json,md}`, `results/logs/variant_stat_auc_v1.log`):
  - macro LOMO AUC: stat_auc 0.864 [0.856, 0.871]; full_gbm 0.862 [0.853, 0.871]. Paired difference +0.002 [-0.005, 0.009].
  - macro precision@10 %: 0.818 [0.797, 0.834] vs 0.810 [0.794, 0.826].
  - Per scheme, paired difference: rotation +0.011 [0.001, 0.021] (the only CI excluding 0); complementary -0.025 [-0.054, 0.001]; every other scheme's CI includes 0.
  - False alarms at the calibration threshold: in 0.054 vs 0.051, size 0.085 vs 0.073, spread 0.055 vs 0.053.
- Result for the hypothesis: NOT falsified. A logistic model on the lit and dep features matches `full_gbm` on validation (paired gap inside the CI), which suggests the boosting step adds little for these features.
- Caveat (mandatory): the configuration was chosen by validation AUC and the same validation data reports it, so 0.864 is optimistic ("validation-tuned"). The gap to `full_gbm` is selection-affected in this variant's favour. The locked test is the unbiased estimate.
- Decision: KEEP the code (comparison variant; rule 8 read as fixed in the pre-registration).
- Side finding: `design_hash` covers every file in `bidrig/detectors/`, so adding `stat_auc.py` changed the hash of `previous` (now 8be1ddcc72085b4d; locked as 90b0cbe63f309f0e). The locked run of `previous` is unaffected (it ran on the earlier source), but a repeat run of `previous` would not be refused by hash. This is the same gap as for `full_gbm`; a name-based guard, or hashing only a variant's own module plus the shared files, would fix it. Not implemented (needs approval).
- Proposed frozen design for `stat_auc`: hash 6a64930fd6bed238 (`docs/frozen_design_stat_auc.md`). No locked run made.

## 2026-11-01 LOCKED run 3: stat_auc (hash 6a64930fd6bed238, approved by the user, run once)

- Command: `cd scripts && python run_locked_test.py --design stat_auc --note "frozen design stat_auc (hash 6a64930fd6bed238), user approved the locked run 2026-11-01"` (seeds 0-9, 1000 bootstraps, 860 s). Files: `results/locked_3_stat_auc.{json,md}`, `results/locked_runs.jsonl`, `results/logs/locked_stat_auc.log`. The hash matched the proposal before the run.
- Macro LOMO AUC 0.862 [0.854, 0.869]; lit_gbm 0.803 [0.792, 0.812]; difference +0.059 [0.051, 0.068]. Against the single screens: +0.188 (rdalt) to +0.374 (gap), all CIs above 0. Macro precision@10 % 0.805 [0.789, 0.818].
- Criteria: C1 FAIL (the same schemes below 0.85 in at least one condition), C2 PASS, C3 PASS, C4 PASS.
- False alarms, all-mechanism model: in 0.041, size 0.044, spread 0.053, markup_high 0.048, capacity 0.055, noisy 0.012; worst leave-one-out models 0.057-0.071. Highest 0.055, below `full_gbm` run 1 (size 0.075).
- Per scheme, pooled AUC (worst condition) next to run 1 `full_gbm`: bid_suppression 0.751 (0.736) vs 0.744; complementary 0.939 (0.919) vs 0.971; cover_bidding 0.950 (0.937) vs 0.946; intermittent 0.863 (0.849) vs 0.871; noisy_defecting 0.821 (0.814) vs 0.823; partial_fringe 0.636 (0.624) vs 0.672; price_fixing 0.969 (0.961) vs 0.971; rotation 0.964 (0.959) vs 0.961. Lowest scheme: partial_fringe 0.636 (`full_gbm` 0.672); complementary is lower by 0.032.
- Uniform-markup control: AUC 0.541 [0.515, 0.567] (`full_gbm`: 0.478), alarm 0.049 [0.030, 0.071]. The AUC interval excludes 0.5 but the alarm rate equals the false-alarm target, so this is not detection; noted as a small difference from `full_gbm`.
- Validation-tuned check: the selection-time validation macro AUC was 0.864; the locked result is 0.862. The optimism from tuning on validation is about 0.002 here, negligible.
- Reading: versus run 1 `full_gbm` (0.870 [0.862, 0.878]) the point estimate is lower by 0.008 and the unpaired CIs overlap, so the locked test does not separate them; macro precision@10 % is higher for `stat_auc` (0.805 vs 0.766, CIs do not overlap). A paired comparison needs `full_gbm` rescored on the test split (a repeat locked run), not done. The loop-prompt criterion (locked AUC within the paired CI of `full_gbm`) cannot be evaluated exactly without that.
- No further locked runs for this design. Three locked runs now exist (`full_gbm`, `previous`, `stat_auc`); all three must be reported together.

## 2026-11-01 VAR-3 (pre-registered before any run): variant `rl_screens`, screen-subset selection by a bandit

- Goal: a detector that chooses its screen subset with a bandit, using the TUNING split only.
- Scope note (deviation from the brief's "RL" wording, flagged for the user): version 0 is a pure-exploration multi-armed bandit (successive halving) over a fixed family of six subsets, not PPO, and it selects ONE subset for all markets, not a per-market policy. Per-market selection and a PPO policy are later versions and need separate approval.
- Arms: lit; dep; lit+dep; lit+dep+change; lit+dep+seq; all groups (groups of FEATURE_GROUPS).
- Bandit: reward = out-of-fold AUC of the standard GBM (shared `GBM_PARAMS`) on the arm's features, 3-fold stratified CV on the training data passed to `fit`. Round 1 evaluates all six arms and keeps the best three; round 2 evaluates the survivors on a different CV split; the chosen arm has the highest mean reward over the rounds it was evaluated in (ties: first in arm order). The chosen arm is refitted on all training data. `fit` sees only the training split it is given; under leave-one-scheme-out the bandit therefore re-runs inside every fold, on data that excludes the held-out scheme. Validation and test are never used to choose the arm.
- Cost estimate: about 28 GBM fits per `fit`; validation needs 9 folds x 10 seeds x 28, about 2,500 fits, expected about 45 minutes (under the 2-hour approval threshold). The locked run would be similar.
- Hypothesis: a subset chosen from tuning data generalises to held-out schemes at least as well as the fixed `full_gbm` feature set.
- Falsifier: on validation, `rl_screens` is below `full_gbm` by more than the paired CI. If so, adaptivity loses and the variant is dropped.
- Interpretation rule for step F, fixed in advance: as with `previous` and `stat_auc`, this is a comparison variant, so the code is kept and the hypothesis outcome is reported.
- Diagnostic to run after validation: which arm the bandit picks on the full tuning split for each seed (`scripts/rl_screens_arms.py`; tuning split only).

## 2026-11-01 REF-1: modularisation (behaviour-preserving) and cleanup

- Request: clean the folder and modularise the code. No result files or docs were deleted.
- Before changing anything, reference outputs were saved from the unmodified code: `tests/golden/reference.json` (feature names and groups, all 53 feature values of four simulated markets, and the scores of 11 detectors on fixed synthetic data); `tests/test_golden.py` compares the refactored code with it (tolerance 1e-8).
- Features: `bidrig/features.py` (459 lines) became a package `bidrig/features/` with one module per group (`literature`, `dependence`, `participation`, `change`, `sequential`), plus `common`, `groups` (column order, `group_features`) and `extract` (`market_features`). Function bodies were moved verbatim; the public names are re-exported from `bidrig.features`.
- Detectors: the designs in `bidrig/models.py` moved into `bidrig/detectors/` as one registered module each (`gbm_groups`, `single_screen`, `tail_fused`) plus `config` (GBM settings) and `priors` (tail directions); `GBMOn` gained an optional monotone constraint so the monotone designs share it. 42 detectors are now registered by name. `bidrig/models.py` is a compatibility layer with no logic, so the frozen `protocol.py` and `metrics.py` are untouched.
- Verification: the golden test passes (feature values and the scores of all 11 reference detectors unchanged); full suite 90 passed, 1 skipped (the real-file test skips without the data; with `BIDRIG_REAL_DATA` set it passes, 4 passed).
- Hashes (a change to protocol scripts, flagged): `scripts/run_locked_test.py` `design_hash` now hashes every file under `bidrig/` recursively (a test covers subpackages), and `scripts/run_real_lodo.py` hashes the `bidrig/features/` package in place of the removed `features.py`. Any code move changes every hash, so the earlier frozen hashes (`full_gbm` 6cc622540eac3929, `previous` 90b0cbe63f309f0e, `stat_auc` 6a64930fd6bed238) cannot be recomputed from the current tree; the locked results were produced by the earlier code and are unaffected. The repeat-run gap (a design name already run is not refused) is still open.
- Anonymity: no user-specific path or name remains in `bidrig/`, `scripts/` or `tests/`. The real-data default folder is `data/real` (override with `BIDRIG_REAL_DATA`); `make_figures.py` reads local copies of the old-environment results in `results/old_env/`; `tests/test_no_identifying_paths.py` guards this.
- `pyproject.toml` now finds subpackages (`bidrig*`).
- Cleanup: Python caches removed (regenerable). Kept: the feature cache `cache/` (53 MB, gitignored, needed by every run), all of `results/` and `docs/`. Files that were deleted or added outside this session (`docs/Bid-Rigging_Detection_Project_Report.pdf`, `docs/report/*`, untracked `docs/bid_rigging_report.pdf`) were left as they are.

## 2026-11-01 VAR-3 result: `rl_screens` (screen-subset bandit), validation

- Tests: `tests/test_rl_screens.py` (6) and the full suite pass (90 passed, 1 skipped without the real data); smoke (`--seeds 0-1 --scale 0.15 --n-boot 30`) ran end to end (paired -0.014 on that tiny scale, not evidence).
- Validation (`cd scripts && python run_variant_validation.py --variant rl_screens --tag rl_screens_v1 --seeds 0-9 --n-boot 300`; files `results/variant_rl_screens_v1.{json,md}`, `results/logs/variant_rl_screens_v1.log`; about 50 minutes):
  - macro LOMO AUC: rl_screens 0.863 [0.855, 0.871]; full_gbm 0.862 [0.853, 0.871]. Paired difference +0.001 [-0.001, 0.003].
  - macro precision@10 %: 0.807 [0.795, 0.825] vs 0.810 [0.794, 0.826].
  - Per scheme, every paired difference is between -0.000 and +0.005 with CIs including 0 (partial_fringe +0.005 [-0.005, 0.014]).
  - False alarms: in 0.051 vs 0.051, size 0.078 vs 0.073, spread 0.053 vs 0.053.
- Diagnostic (`scripts/rl_screens_arms.py --tag rl_screens_arms_v1`, tuning split only; `results/rl_screens_arms_rl_screens_arms_v1.{json,md}`): over seeds 0-9 the bandit chose lit_dep in 5 seeds, lit_dep_change in 2, lit_dep_seq in 2 and all in 1. It does not simply pick the full set. The four subsets that contain both the literature and dependence groups score within about 0.01 of each other (out-of-fold AUC 0.85-0.90 by seed); single-group subsets score clearly lower (lit 0.79-0.83, dep 0.81-0.86).
- Result for the hypothesis: NOT falsified (the gap to full_gbm is inside the CI). But adaptivity shows no measurable value: the subsets it can choose between are nearly tied, in line with the earlier ablation (change scans and the sequential layer add nothing).
- Decision: KEEP the code (comparison variant, rule fixed in the pre-registration). Reported as a negative finding for "choosing screens helps".
- Limits of this version: one global subset for all markets (not per market), a bandit not PPO. A per-market policy remains untested.
- Proposed frozen design: hash 002a3d7267394d48 (`docs/frozen_design_rl_screens.md`). The locked run costs about 50 minutes (under the 2-hour threshold) and has not been made.

## LOCKED-4: rl_screens (2026-11-01)

Design hash 002a3d7267394d48 (frozen in docs/frozen_design_rl_screens.md), user approved, run once (1529 s).
Result: macro LOMO AUC 0.872 [0.863, 0.880], precision@10% 0.772 [0.757, 0.788], min mechanism AUC 0.678
(partial_fringe). Criteria: C1 fail (partial_fringe 0.678, bid_suppression 0.747), C2 pass, C3 pass, C4 pass.
Honest false alarms: all-mechanism model at most 0.082; worst leave-one-out model 0.106 on honest@size and
0.098 on honest@spread. Uniform-markup control AUC 0.478.
Validation said 0.863 (paired +0.001 vs full_gbm); the locked value is 0.872 vs 0.870 for full_gbm. The
locked runs are separate samples and were not paired, so the difference is not tested. Files:
results/locked_4_rl_screens.{json,md}.
