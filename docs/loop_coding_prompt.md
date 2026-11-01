# Loop-coding prompt for the detector variants

Paste the whole block below into a new session. Fill in only the `VARIANT` line. Each variant
has its own filled-in block at the end of this file; copy that block into the prompt.

```
You are coding one detector variant for the bidrig project in "Jayati ma'am project-v2".
Work in a loop, one variant at a time.

Read first: docs/paper_outline.md, docs/claims_table.md, docs/experiment_log.md (last 20
entries), and the existing bidrig/ modules, especially bidrig/models.py, bidrig/protocol.py
and bidrig/features.py. Do not modify any file in the frozen protocol or in logged results.

VARIANT: <fill in from the blocks below>

Rules (non-negotiable):
1. Choices use the validation split only. Never read or tune on the test split.
2. No global random; pass an rng. Sort sets. Same seed must give identical output.
3. One locked run per frozen design. Refuse to run a design twice.
4. Every component gets a unit test. Commit only when all tests pass.
5. black and isort clean.
6. Any run expected over about 2 hours needs my approval first. Downloads need my approval.
7. Never delete files in results/ or docs/.
8. Keep the change only if validation improves beyond the paired CI; otherwise revert it
   and log the revert.

Modularity:
- One interface per detector type (fit / score), registered by name.
- One module per variant. A variant module may depend on the shared interface and on
  bidrig.features, never on another variant.
- No hardcoded paths. Config comes from arguments or environment variables.

Loop steps (repeat until a stop rule):
A. Implement the smallest change. Show the diff.
B. Run unit tests for the new code.
C. Smoke test on small seeds and scale, e.g.
   cd scripts && python run_validation.py --tag smoke_<name> --seeds 0-1 --scale 0.15 --n-boot 30
D. Full validation (seeds 0-9) in the background. Confirm the log is non-empty after
   launch. Report macro LOMO AUC with CI, and the paired difference to the reference
   detector (full_gbm).
E. Log the result in docs/experiment_log.md: date, hypothesis, change, command, outcome,
   failures.
F. Decide: keep (commit) or revert (log the revert). Give the reason in one line.

Stop rules:
- Three iterations without validation gain for this variant: stop and report.
- The variant clears the validation bar: propose the frozen design and ask me before any
  locked run.

End each loop with: what changed, validation numbers with CIs, the decision, and the next
single step.
```

---

## Block 1: VARIANT "previous"

```
VARIANT: previous (the detector before iteration I1)
Goal: a detector with the same model and features as full_gbm, minus the second
      within-firm null (the columns that end in "_t").
Hypothesis: the second null is what raised validation AUC from 0.771 to 0.826 (I1).
            Removing it should bring the dependence-only result back to about 0.77.
Smallest change: a feature selector that drops every column ending in "_t", applied to
      the full_gbm model. No change to training.
Falsifying test: on validation, previous must be below full_gbm by more than the paired
      CI. If the gap is inside the CI, the second null does not matter and I1 was not a
      real gain.
Reference: docs/experiment_log.md entries V1 and I1.
```

## Block 2: VARIANT "rl_screens" (screen-subset bandit)

```
VARIANT: rl_screens (screen-subset selection by a bandit)
Goal: a detector that chooses its screen subset with a bandit, using tuning data only.
Scope note: version 0 is a bandit over a fixed family of screen subsets, not PPO, and it
      picks one subset for all markets (not per market). Per-market selection and PPO are
      later versions and need separate approval.
Arms: lit; dep; lit+dep; lit+dep+change; lit+dep+seq; all groups.
Selection: successive halving on out-of-fold AUC over 5 folds of the TUNING split only.
      The validation split is used only to report the chosen arm, never to choose it.
Hypothesis: a subset chosen from tuning data generalises to held-out mechanisms at least
      as well as the fixed full_gbm model.
Smallest change: the bandit over the six arms, reusing the existing GBM settings.
Falsifying test: on validation, the chosen arm must not be worse than full_gbm by more than
      the paired CI. If it is worse, adaptivity loses and the variant is dropped.
Flag for the user: this is a deviation from the brief's "RL" wording. Say so in the log
      and in any write-up.
```

## Block 3: VARIANT "stat_auc" (validation-tuned logistic model)

```
VARIANT: stat_auc (purely statistical, validation-tuned)
Goal: a logistic regression on standardised features, with the feature subset and penalty
      chosen to maximise AUC on the validation split.
Grid: subsets dep; lit+dep; lit+dep+seq; all groups. Penalties C in {0.01, 0.1, 1, 10}.
Hypothesis: a simple linear model can match full_gbm, which would mean the boosting step
      adds little.
Smallest change: the logistic model and the grid, with fitted standardisation.
Falsifying test: the locked-test AUC must be within the paired CI of full_gbm. A larger gap
      in either direction is reported as is.
Mandatory caveat: the choices are made by AUC, so this variant is optimistic. Report it as
      "validation-tuned", and say that its validation AUC is not an unbiased estimate.
```

---

## What to keep consistent across variants

- Every variant gets its own locked run. All four are reported, including the reference.
- The reference, full_gbm, is frozen with locked run 1, which is already done.
- Each variant's validation results decide whether it proceeds to a frozen design.
