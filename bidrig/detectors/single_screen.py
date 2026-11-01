"""Baseline detectors: one literature screen on its own."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bidrig.detectors.base import column_index, register
from bidrig.metrics import auc

SINGLE_SCREENS = tuple(
    f"lit_{s}_mean" for s in ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread")
)


@dataclass
class SingleScreen:
    """One literature screen; its sign is chosen on the tuning data (AUC >= 0.5)."""

    feature: str
    sign: float = 1.0

    def _col(self, X: np.ndarray) -> np.ndarray:
        from bidrig.features import ALL_FEATURES

        return X[:, column_index(ALL_FEATURES, [self.feature])[0]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleScreen":
        col = self._col(X)
        self.sign = 1.0 if auc(col[y == 1], col[y == 0]) >= 0.5 else -1.0
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.sign * self._col(X)


for _feature in SINGLE_SCREENS:
    register(f"single:{_feature}")(lambda f=_feature: SingleScreen(f))
