# Paper outline (workshop / benchmark track)

Framing chosen by the user on 2026-10-22: a benchmark paper with an honest partial negative
result. Claims used: 2-6, 8, 10 and (added 2026-10-22) 13-15 from `docs/claims_table.md`.
Claim 7 (C1 not met) is a headline limitation; claims 14-15 and the Japan/Brazil results are
limitations. Simulator numbers come from LOCKED run 1 (`results/locked_1_full_gbm.json`:
design full_gbm, seeds 0-9, two-level bootstrap 95 % CIs, 1000 resamples) unless marked
"validation". Real-data numbers come from REAL run 1 (`results/real_lodo_1.json`: hash
1cec4feb56110f6c, run once, moving-block bootstrap with block 20 and 1000 resamples).

Target length: 4-6 pages plus appendix.

## Working title

- "Which bid-rigging schemes can public bids reveal? A leave-one-mechanism-out benchmark"
- Alternative: "Beyond collusion screens: a benchmark of identity-aware bid-rigging detection
  under unseen schemes and market shifts"

## Abstract (draft)

Collusion screens built from the bid distribution (coefficient of variation, skewness,
relative distance) are standard tools for competition authorities, but they are usually
evaluated on the same kinds of cartels they were designed for. We release a benchmark of
simulated repeated procurement markets. It has persistent bidders, noisy public estimates and
honest markups of 0-25 %, plus eight imposed collusion mechanisms and a protocol that holds
out whole mechanisms, unseen market sizes and unseen cost spreads, evaluated once on a locked
test split. A detector built on identity-dependence statistics, standardised against
within-market permutation nulls, reaches a macro leave-one-mechanism-out AUC of 0.870
[0.862, 0.878]. A gradient-boosted model on the literature screens reaches 0.803
[0.792, 0.812], and the best single screen 0.674. False alarms stay at or below 7.5 % on every
honest test set. Four mechanism families (rotation, price fixing, complementary and cover
bidding) are detected at AUC ≥ 0.92 under every shift. Four are not detected at 0.85: small
cartels inside a competitive fringe (0.672), bid suppression (0.744), noisy or defecting
cartels (0.823) and intermittent cartels at unseen spreads (0.849). We report these as the
benchmark's open problems. On real bid data from three prosecuted cartel cases (Italy, Japan,
Brazil), evaluated tender by tender with each country held out in turn, pair features
computed from the other tenders improve on the screens only in Italy, the one case with
repeated bidders (AUC +0.144 [0.078, 0.234]). In Japan and Brazil a single CV screen does as
well or better, and an alarm threshold set on two countries gives 26.5 % false alarms on the
third.

## 1. Introduction

- Problem: screens are calibrated and tested on in-distribution cartels; deployment faces
  unseen schemes and markets of different size and cost dispersion.
- Gap: no public benchmark that holds out whole mechanisms and market shifts with a locked
  test and CIs.
- Contributions:
  1. Benchmark: simulator, eight in-scope mechanisms plus one negative control, splits and a
     locked protocol (Section 3).
  2. A detector whose identity-dependence features are invariant to uniform markups and to the
     estimate, standardised against two within-market nulls (Section 4; claim 10).
  3. Locked-test evidence: it beats the literature screens (claims 2, 3), keeps false alarms
     bounded under shift (claim 5), ranks markets usefully (claim 6) and detects four
     mechanism families robustly (claim 4).
  4. A measured map of what it does not detect (claim 7, Section 6) and an identification
     limit confirmed by a negative control (claim 8).
  5. A pre-registered real-data check, one run, on three cartel cases: a gain where bidders
     recur (Italy, claim 13), and no gain or threshold transfer elsewhere (claims 14-15).

## 2. Related work (to be written; no citations checked yet)

- Collusion screens: variance and distribution screens, relative-distance and gap screens.
- Machine-learning detection of bid rigging on screen features.
- Identity- and pair-based tests: conditional independence of bids, co-bidding patterns.
- Change detection and sequential monitoring in markets.
- Positioning: our contribution is the evaluation protocol and an honest account of which
  schemes are found, not a new classifier family.
- TODO: every citation needs to be checked against the source before it is used.

## 3. Benchmark

### 3.1 Simulator
- Repeated first-price procurement tenders, 6-30 persistent firms, 20-60 tenders per market.
- Firm cost = project size × persistent efficiency × type specialisation × idiosyncratic
  noise × capacity backlog (raises cost after recent wins; an honest source of rotation-like
  patterns).
- Public estimate = project size × noise, never derived from realised costs; drawn before any
  mechanism parameter, so it is identical for honest and collusive markets of the same seed.
- Honest bids: cost × (uniform market markup 0-25 % + firm markup + competition shading).
- Randomised per market: size, cost spread, estimate noise, participation, specialisation,
  capacity effect, tender count.

### 3.2 Collusion library (imposed, not equilibrium behaviour)
- Table 1: the eight mechanisms with their parameters and ranges: rotation, cover bidding
  (lowest-cost designated winner), complementary bidding (covers at fixed multiples of the
  estimate), price fixing (three levels), partial cartel with a fringe (2-3 firms),
  noisy/defecting cartel, intermittent cartel (active in one window of 30-60 % of tenders),
  bid suppression.
- Negative control: uniform-markup collusion (30-60 %), out of scope by construction.

### 3.3 Protocol
- Splits: tuning (fit), calib (honest only; sets the 5 % false-alarm threshold), validation
  (design choice; moderately shifted sizes and spreads), locked test (sizes 6-7 and 25-30,
  spreads 0.02-0.04 and 0.22-0.30, plus honest stress sets: high markup, strong capacity
  effects, noisy estimates and costs).
- Leave-one-mechanism-out (LOMO): each mechanism is scored by a model trained without it.
- Criteria fixed before any validation number was seen (C1-C4, Section 5).
- Ten seeds; two-level bootstrap (seeds, then markets), paired across detectors.
- Every metric is unit-tested on separated, reversed and tied data; determinism is checked
  across processes and worker counts (claim 1, reproducibility appendix).
- The locked test is run once per frozen design; repeat runs are refused by the runner and
  every run is logged.

## 4. Detectors

- Baselines: seven literature screens per tender (CV, skewness, kurtosis, relative distance,
  RD-alt, normalised gap, spread), aggregated per market; each alone (sign chosen on tuning)
  and a gradient-boosted model on all of them.
- Proposed (full_gbm): gradient boosting on the literature screens plus identity-dependence
  features, within-market change scans and a sequential layer (CUSUM, HMM filter).
- Identity-dependence features: residuals = log(bid / estimate) minus the tender median, which
  removes the estimate, tender shocks and uniform markups. Statistics: pair residual
  correlation, firm residual dispersion, cover-bid tracking, top-two pair recurrence,
  win/loss inconsistency, repeat wins, near-identical and close bid pairs, co-bidding.
- Two within-market nulls: shuffling bids among the participants of each tender, and
  shuffling each firm's residuals across its own tenders (keeps every firm's marginal
  behaviour and breaks within-tender coordination). Claim 10 (validation): adding the second
  null raised the dependence-only model from 0.771 to 0.826 and the full model from 0.847 to
  0.862.
- State plainly: change scans and the sequential layer add no measurable power (validation:
  change-only 0.494, sequential-only 0.549; dropping either changes the full model by
  ≤ 0.002). They stay in the frozen design because it was frozen with them.

## 5. Results (locked test)

### 5.1 Against the literature screens (claims 2, 3)
- Table 2: macro LOMO AUC and macro precision@10 % with CIs.
  - full_gbm 0.870 [0.862, 0.878]; precision@10 % 0.766 [0.753, 0.785].
  - lit_gbm 0.803 [0.792, 0.812]; 0.714 [0.695, 0.728]. Paired difference +0.067
    [0.060, 0.074].
  - Single screens: RD-alt 0.674, RD 0.641, CV 0.621, spread 0.610, kurtosis 0.609, skewness
    0.584, gap 0.488. Paired differences +0.196 to +0.382, all CIs above 0.

### 5.2 Per held-out mechanism and shift (claim 4; claim 7 in Section 6)
- Table 3: LOMO AUC pooled and per condition (in-distribution, unseen sizes, unseen spreads),
  precision@10 % and alarm rate at 5 % false alarms.
- Detected robustly (worst condition): rotation 0.939, price fixing 0.963, complementary
  0.947, cover bidding 0.921.
- Figure 1 (to make): per-mechanism AUC by condition with CIs and the 0.85 line.

### 5.3 False alarms under shift (claim 5)
- Table 4: alarm rate at the calib 5 % threshold, model trained on all mechanisms: in 0.053,
  unseen sizes 0.075 [0.050, 0.105], unseen spreads 0.072, high markup 0.043, capacity 0.048,
  noisy 0.013.
- Caveat in text: the LOMO models reach 0.116 [0.090, 0.155] on unseen sizes and 0.103 on
  unseen spreads, so the bound holds for the deployable model only.
- The literature GBM's false-alarm rate on unseen sizes is 0.106 [0.067, 0.146].

### 5.4 Ranking (claim 6)
- Precision@10 % above the 0.167 base rate for every held-out mechanism; lowest 0.378
  [0.335, 0.418] (partial cartel with fringe), highest 0.983 (price fixing).

### 5.5 Negative control (claim 8)
- Uniform-markup collusion: AUC 0.478 [0.450, 0.503], alarm rate 0.047 [0.030, 0.068], which
  is indistinguishable from honest markets. This is expected: without cost data a common
  markup looks like higher costs. The residual construction makes the detector blind to it by
  design.

### 5.6 Real-data case study (claim 13)

- Data: García Rodríguez et al. (2022) bid-level datasets with firm IDs and pre-tender
  estimates: Italy (Turin roads, 278 tenders, 143 collusive), Japan (Okinawa, 1,080, 123),
  Brazil (Petrobras, 101, 33). Switzerland and the USA are excluded (no firm IDs or no
  estimate).
- Why tender-level: windows of consecutive tenders do not form repeated markets. In Japan no
  window is majority-collusive and no firm bids 6 or more times in any window; Brazil gives 3-5
  windows (log D2). The market-level simulator detector therefore cannot be applied as is.
- Design (`docs/real_data_design.md`, frozen and approved before the run):
  - the tender's own literature screens;
  - pair features for the tender's bidders, computed on all other tenders (exact
    leave-tender-out): residual correlation, share of close bids, co-bidding;
  - within-country percentile scaling;
  - the same GBM settings as the simulator;
  - train on two countries, test on the third;
  - threshold at 5 % false alarms on the training countries.
- Table 5: per held-out country, AUC, paired difference to the screens-only model,
  false-alarm rate, and precision@10 % with base rate.
- Italy (case study): combined 0.779 [0.668, 0.876] vs screens-only 0.635 [0.492, 0.770],
  paired difference +0.144 [0.078, 0.234]; false alarms 0.007; precision@10 % 0.929
  [0.749, 1.000] against a base rate of 0.514. It beats every single screen (best: spread
  0.676).
- Reading: Italy is the only country where the same firms meet repeatedly (seven
  bidder-overlap clusters of 20 or more tenders, log D2). That is where identity information
  exists to use, which matches the simulator's mechanism. One case, so this is illustrative,
  not a general result.
- Caveat in text: Turin uses an average-bid auction, so the gap and relative-distance screens
  mean something different there. The pair features do not depend on who wins.

## 6. Limitations and open problems (headline: claim 7, C1 not met)

- C1 (every held-out mechanism ≥ 0.85, including unseen sizes and spreads) fails for four of
  eight mechanisms:
  - partial cartel with a fringe: 0.672 (worst condition 0.621);
  - bid suppression: 0.744 (0.677);
  - noisy/defecting cartel: 0.823 (0.775);
  - intermittent cartel: 0.871 pooled, but 0.849 at unseen spreads.
- What was tried and failed on validation (report all of it, briefly):
  - knowledge-directed tail score fused with the model, plus participation features;
  - maximum-pair statistics (prototype);
  - monotone constraints;
  - three change-detection variants (pair-coordination scan, segment excess, epidemic scan).
  None moved the worst mechanisms beyond the CI.
- Why (hypotheses, labelled as such): a 2-3 firm cartel leaves a single pair-level trace among
  up to 435 pairs in 20-60 tenders. Bid suppression's own signal (missing bids) cannot be
  learned when it is held out. Noisy and defecting cartels dilute the cover pattern.
- Real data (claims 14, 15):
  - Japan: no gain over the screens-only model (+0.018 [−0.032, 0.069]). A single CV screen
    (0.739) is at least as good as the combined model (0.691); difference −0.047
    [−0.102, 0.010].
  - Brazil: no gain (−0.013 [−0.109, 0.177]). The combined model is worse than CV alone
    (0.742 vs 0.873, difference −0.131 [−0.247, −0.013]), and precision@10 % (0.500
    [0.200, 0.800]) does not clear the 0.327 base rate.
  - Threshold transfer fails: at the threshold set on two countries, false alarms on the held
    -out country are 0.007 (Italy), 0.088 (Brazil) and 0.265 (Japan; screens-only 0.212). A
    deployed alarm would need recalibration per market.
  - Hypotheses (labelled as such): Japan's and Brazil's firms rarely recur, so pair features
    have little history (Brazil: median 1 tender per firm). Labels are the paper's
    thresholded tender labels (216 Japanese tenders mix members and non-members). Each country
    is a single cartel case.
- Other limitations:
  - real-data evidence is three cartel cases at tender level; the market-level benchmark
    itself is synthetic;
  - cartels are imposed, not equilibrium behaviour, and the parameter ranges are design
    choices;
  - market-level scores only, no identification of members;
  - the design was frozen with components that add nothing.

## 7. Conclusion

- The benchmark separates schemes that public bids reveal from schemes that they do not
  reveal at useful power.
- Release: code, splits, locked-run log, and the rule that the test split runs once per
  frozen design.

## Appendix plan

- A: mechanism parameter ranges (from `bidrig/collusion.py`).
- B: full validation history (V1, I1-I4, W2-1 to W2-3) from `docs/experiment_log.md`,
  including the reverted changes.
- C: metric sanity tests and determinism check.
- D: per-condition tables for all baselines from `results/locked_1_full_gbm.md`.
- E: real-data design (`docs/real_data_design.md`), window and clustering diagnostics
  (log D2), and the full per-country tables from `results/real_lodo_1.md`, including all
  single screens.

## Do not claim

- "Publication-ready" or "solves bid-rigging detection": C1 failed on the locked test.
- Any effect of the change or sequential layers.
- Any general real-world detection rate. The real data is three cases; Italy is one case
  study.
- That the detector beats simple screens on real data in general (false for Japan and
  Brazil).
- A transferable alarm threshold.
- Validation numbers presented as test results.
