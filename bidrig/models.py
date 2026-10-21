"""Detector designs: single literature screens and gradient boosting on feature groups.

A design is fitted on a tuning ``FeatureTable`` and returns suspicion scores (higher = more
suspicious) for any other table. Designs are named so a locked run can record exactly
what was frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bidrig.features import ALL_FEATURES, FEATURE_GROUPS
from bidrig.metrics import auc

SINGLE_SCREENS = tuple(
    f"lit_{s}_mean" for s in ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread")
)

GBM_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


def _columns(names) -> np.ndarray:
    index = {f: i for i, f in enumerate(ALL_FEATURES)}
    return np.array([index[n] for n in names])


@dataclass
class SingleScreen:
    """One literature screen; its sign is chosen on the tuning data (AUC >= 0.5)."""

    feature: str
    sign: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleScreen":
        col = X[:, _columns([self.feature])[0]]
        self.sign = 1.0 if auc(col[y == 1], col[y == 0]) >= 0.5 else -1.0
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.sign * X[:, _columns([self.feature])[0]]


@dataclass
class GBM:
    groups: tuple[str, ...]
    model: HistGradientBoostingClassifier | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(f for g in self.groups for f in FEATURE_GROUPS[g])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBM":
        self.model = HistGradientBoostingClassifier(**GBM_PARAMS)
        self.model.fit(X[:, _columns(self.features)], y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, _columns(self.features)])[:, 1]


DESIGNS = {
    "lit_gbm": ("lit",),
    "dep_gbm": ("dep",),
    "change_gbm": ("change",),
    "seq_gbm": ("seq",),
    "knowledge_gbm": ("dep", "change", "seq"),
    "full_gbm": ("lit", "dep", "change", "seq"),
    "full_minus_dep": ("lit", "change", "seq"),
    "full_minus_change": ("lit", "dep", "seq"),
    "full_minus_seq": ("lit", "dep", "change"),
}
BASELINES = ("lit_gbm",) + tuple(f"single:{f}" for f in SINGLE_SCREENS)


def make_design(name: str):
    if name.startswith("single:"):
        return SingleScreen(name.split(":", 1)[1])
    return GBM(DESIGNS[name])
