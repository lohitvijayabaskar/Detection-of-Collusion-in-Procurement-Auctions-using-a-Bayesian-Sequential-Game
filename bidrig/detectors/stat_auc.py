"""Variant ``stat_auc``: a validation-tuned logistic regression.

A plain L2 logistic regression on median-imputed, standardised features. The feature subset and
the penalty are chosen once, by ``scripts/select_stat_auc.py``, to maximise macro leave-one-scheme-
out AUC on the VALIDATION split. The choice is then frozen in ``CHOSEN`` below, so it is part of
the design hash. ``fit`` itself sees the tuning split only: imputation medians, standardisation
and the model are all fitted on tuning data.

Because the settings are chosen by AUC and the same validation data reports the result, the
validation AUC of this variant is optimistic. Report it as "validation-tuned".
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from bidrig.detectors.base import column_index, register
from bidrig.features import ALL_FEATURES, FEATURE_GROUPS

SUBSETS = {
    "dep": ("dep",),
    "lit_dep": ("lit", "dep"),
    "lit_dep_seq": ("lit", "dep", "seq"),
    "all": ("lit", "dep", "change", "seq"),
}
PENALTIES = (0.01, 0.1, 1.0, 10.0)
GRID = tuple((s, c) for s in SUBSETS for c in PENALTIES)  # grid order breaks ties

# Frozen choice, set after the validation selection (see docs/experiment_log.md).
CHOSEN: tuple[str, float] | None = ("lit_dep", 0.1)  # results/stat_auc_selection_stat_auc_select_v1


def subset_features(subset: str) -> tuple[str, ...]:
    return tuple(f for g in SUBSETS[subset] for f in FEATURE_GROUPS[g])


def grid_name(subset: str, C: float) -> str:
    return f"stat_auc:{subset}:{C:g}"


class StatAUC:
    """Imputed, standardised L2 logistic regression on one feature subset."""

    def __init__(self, subset: str, C: float, names=ALL_FEATURES):
        if subset not in SUBSETS:
            raise KeyError(f"unknown subset {subset!r}; known: {sorted(SUBSETS)}")
        self.subset, self.C = subset, float(C)
        self.name = "stat_auc" if (subset, C) == CHOSEN else grid_name(subset, C)
        self.features = subset_features(subset)
        self._cols = column_index(names, self.features)
        self._median = self._mean = self._sd = None
        self.model: LogisticRegression | None = None

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        Z = np.where(np.isnan(X[:, self._cols]), self._median, X[:, self._cols])
        return (Z - self._mean) / self._sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatAUC":
        raw = X[:, self._cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are handled below
            med = np.nanmedian(raw, axis=0)
        self._median = np.where(np.isnan(med), 0.0, med)  # an all-NaN column becomes constant
        Z = np.where(np.isnan(raw), self._median, raw)
        self._mean = Z.mean(axis=0)
        self._sd = Z.std(axis=0) + 1e-9
        self.model = LogisticRegression(C=self.C, max_iter=2000)
        self.model.fit((Z - self._mean) / self._sd, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._prepare(X))[:, 1]


for _subset, _C in GRID:
    register(grid_name(_subset, _C))(lambda s=_subset, c=_C: StatAUC(s, c))

if CHOSEN is not None:
    register("stat_auc")(lambda: StatAUC(*CHOSEN))
