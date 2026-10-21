"""Detector designs: single literature screens and gradient boosting on feature groups.

A design is fitted on a tuning ``FeatureTable`` and returns suspicion scores (higher = more
suspicious) for any other table. Designs are named so a locked run can record exactly
what was frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

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
    monotone: dict | None = None  # feature -> +1/-1 constraint (knowledge prior)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(f for g in self.groups for f in FEATURE_GROUPS[g])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBM":
        params = dict(GBM_PARAMS)
        if self.monotone:
            params["monotonic_cst"] = [self.monotone.get(f, 0) for f in self.features]
        self.model = HistGradientBoostingClassifier(**params)
        self.model.fit(X[:, _columns(self.features)], y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, _columns(self.features)])[:, 1]


# Knowledge prior: direction in which collusion moves each feature (+1 up, -1 down). Fixed
# from the economics of each mechanism family, never fitted on labels. Only scale-free
# identity and participation features are used, so price level and cost spread do not enter.
TAIL_DIRECTIONS = {
    "dep_pair_corr_pos_t": +1,
    "dep_close_pair_t": +1,
    "dep_near_tie_pair_t": +1,
    "dep_firm_sd_min_t": +1,
    "dep_cover_sd_min_t": +1,
    "dep_win_loss_corr": +1,
    "dep_cobid_min": +1,
    "part_group": +1,
    "part_rate": -1,
}
TAIL_MARGIN = 2.0


@dataclass
class TailScore:
    """Sum of one-sided exceedances of honest-standardised features beyond ``TAIL_MARGIN``.

    Standardisation (median, MAD) uses the honest tuning markets only; labels of collusive
    markets are never used, so the score does not depend on which mechanisms were seen.
    """

    directions: dict = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TailScore":
        self.directions = dict(TAIL_DIRECTIONS)
        H = X[y == 0][:, _columns(self.directions)]
        self.center = np.nanmedian(H, axis=0)
        mad = np.nanmedian(np.abs(H - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > 1e-9, mad, np.nanstd(H, axis=0) + 1e-9)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, _columns(self.directions)] - self.center) / self.scale
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
    gbm: GBM | None = None
    tail: TailScore | None = None
    ref_gbm: np.ndarray | None = None
    ref_tail: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Fused":
        oof = np.full(len(y), np.nan)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in folds.split(X, y):
            oof[te] = GBM(self.groups).fit(X[tr], y[tr]).score(X[te])
        self.ref_gbm = oof[y == 0]
        self.gbm = GBM(self.groups).fit(X, y)
        self.tail = TailScore().fit(X, y)
        self.ref_tail = self.tail.score(X[y == 0])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        p1 = _upper_pvalue(self.ref_gbm, self.gbm.score(X))
        p2 = _upper_pvalue(self.ref_tail, self.tail.score(X))
        return -np.log(np.minimum(p1, p2))


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
    "full_part_gbm": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_gbm": ("lit", "dep", "part"),
}
MONOTONE = {
    "full_part_mono": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_mono": ("lit", "dep", "part"),
}
FUSED = {
    "full_fused": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_fused": ("lit", "dep", "part"),
}
BASELINES = ("lit_gbm",) + tuple(f"single:{f}" for f in SINGLE_SCREENS)


def make_design(name: str):
    if name.startswith("single:"):
        return SingleScreen(name.split(":", 1)[1])
    if name == "tail_only":
        return TailScore()
    if name in FUSED:
        return Fused(FUSED[name])
    if name in MONOTONE:
        return GBM(MONOTONE[name], monotone=dict(TAIL_DIRECTIONS))
    return GBM(DESIGNS[name])
