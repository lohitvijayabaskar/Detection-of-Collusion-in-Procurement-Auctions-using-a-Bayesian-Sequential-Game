"""Knowledge-directed tail score, and its fusion with the gradient-boosted model.

``TailScore`` is a sum of one-sided exceedances of honest-standardised features beyond
``TAIL_MARGIN``. ``Fused`` combines it with a GBM through honest p-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bidrig.detectors.base import GBMOn, column_index, register
from bidrig.detectors.priors import TAIL_DIRECTIONS, TAIL_MARGIN
from bidrig.features import ALL_FEATURES, group_features

FUSED = {
    "full_fused": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_fused": ("lit", "dep", "part"),
}


@dataclass
class TailScore:
    """Sum of one-sided exceedances of honest-standardised features beyond ``TAIL_MARGIN``.

    Standardisation (median, MAD) uses the honest tuning markets only; labels of collusive
    markets are never used, so the score does not depend on which mechanisms were seen.
    """

    directions: dict = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    def _cols(self):
        return column_index(ALL_FEATURES, list(self.directions))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TailScore":
        self.directions = dict(TAIL_DIRECTIONS)
        H = X[y == 0][:, self._cols()]
        self.center = np.nanmedian(H, axis=0)
        mad = np.nanmedian(np.abs(H - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > 1e-9, mad, np.nanstd(H, axis=0) + 1e-9)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, self._cols()] - self.center) / self.scale
        z = z * np.array(list(self.directions.values()), dtype=float)
        return np.nansum(np.maximum(z - TAIL_MARGIN, 0.0), axis=1)


def _upper_pvalue(ref: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(1 + #ref >= s) / (n + 1) for each score in ``s``."""
    ref = np.sort(np.nan_to_num(ref, nan=-np.inf))
    s = np.nan_to_num(s, nan=-np.inf)
    ge = ref.size - np.searchsorted(ref, s, side="left")
    return (1.0 + ge) / (ref.size + 1.0)


@dataclass
class Fused:
    """GBM on feature groups fused with ``TailScore``: score = -log min(p_gbm, p_tail).

    p-values are against honest tuning markets: out-of-fold GBM scores (5-fold, fixed
    split) and in-sample tail scores (the tail score uses no labels).
    """

    groups: tuple[str, ...]
    gbm: GBMOn | None = None
    tail: TailScore | None = None
    ref_gbm: np.ndarray | None = None
    ref_tail: np.ndarray | None = None

    def _new_gbm(self) -> GBMOn:
        return GBMOn(group_features(self.groups))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Fused":
        oof = np.full(len(y), np.nan)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in folds.split(X, y):
            oof[te] = self._new_gbm().fit(X[tr], y[tr]).score(X[te])
        self.ref_gbm = oof[y == 0]
        self.gbm = self._new_gbm().fit(X, y)
        self.tail = TailScore().fit(X, y)
        self.ref_tail = self.tail.score(X[y == 0])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        p1 = _upper_pvalue(self.ref_gbm, self.gbm.score(X))
        p2 = _upper_pvalue(self.ref_tail, self.tail.score(X))
        return -np.log(np.minimum(p1, p2))


register("tail_only")(TailScore)
for _name, _groups in FUSED.items():
    register(_name)(lambda g=_groups: Fused(g))
