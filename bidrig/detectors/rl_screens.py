"""Variant ``rl_screens``: screen-subset selection by a bandit (version 0).

Scope. Each arm is a subset of feature groups. A pure-exploration bandit (successive halving)
chooses ONE subset for all markets; this is not PPO and not a per-market policy. Those are
later versions.

Reward. Out-of-fold AUC of the standard gradient-boosted model on the arm's features, computed
by stratified cross-validation on the training data handed to ``fit``. Nothing else is read:
the validation and test splits never influence the choice, and under leave-one-scheme-out the
bandit re-runs inside every fold on data that excludes the held-out scheme.

Rounds. Round 1 evaluates every arm and keeps the best half; round 2 evaluates the survivors on a
different CV split. The chosen arm has the highest mean reward over the rounds it was evaluated
in (ties go to the first arm in ``ARMS`` order). The chosen arm is then refitted on all data.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bidrig.detectors.base import GBMOn, register
from bidrig.features import FEATURE_GROUPS
from bidrig.metrics import auc

ARMS = {
    "lit": ("lit",),
    "dep": ("dep",),
    "lit_dep": ("lit", "dep"),
    "lit_dep_change": ("lit", "dep", "change"),
    "lit_dep_seq": ("lit", "dep", "seq"),
    "all": ("lit", "dep", "change", "seq"),
}
FOLDS = 3
ROUNDS = 2
SEED = 0


def arm_features(arm: str) -> tuple[str, ...]:
    return tuple(f for g in ARMS[arm] for f in FEATURE_GROUPS[g])


def oof_auc(features, X: np.ndarray, y: np.ndarray, round_index: int) -> float:
    """Out-of-fold AUC of a GBM on ``features`` using only ``X`` and ``y``."""
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + round_index)
    s = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        s[te] = GBMOn(features).fit(X[tr], y[tr]).score(X[te])
    return float(auc(s[y == 1], s[y == 0]))


@register("rl_screens")
class RLScreens:
    name = "rl_screens"

    def __init__(self):
        self.arm: str | None = None
        self.arm_rewards: dict[str, list[float]] = {}
        self._gbm: GBMOn | None = None

    def select_arm(self, X: np.ndarray, y: np.ndarray) -> str:
        alive = list(ARMS)  # arm order is the tie-break
        self.arm_rewards = {a: [] for a in ARMS}
        for r in range(ROUNDS):
            for a in alive:
                self.arm_rewards[a].append(oof_auc(arm_features(a), X, y, r))
            if r < ROUNDS - 1:
                mean = {a: float(np.mean(self.arm_rewards[a])) for a in alive}
                ranked = sorted(alive, key=lambda a: (-mean[a], alive.index(a)))
                alive = sorted(ranked[: max(1, len(alive) // 2)], key=list(ARMS).index)
        mean = {a: float(np.mean(self.arm_rewards[a])) for a in alive}
        return max(alive, key=lambda a: (mean[a], -list(ARMS).index(a)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RLScreens":
        self.arm = self.select_arm(X, y)
        self._gbm = GBMOn(arm_features(self.arm)).fit(X, y)
        return self

    @property
    def features(self) -> tuple[str, ...]:
        return arm_features(self.arm)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._gbm.score(X)
