"""Re-implementations of published detection ideas, for the literature comparison.

Each detector is one idea reduced to a market-level score on our features; none is the
authors' code or data pipeline. Detectors that use a single statistic choose its sign on the
tuning split (AUC >= 0.5), which can only favour the baseline. Unsupervised detectors ignore
the labels. Methods that need data we hide (costs, rebids, secret reserve prices) or heavy
structural / deep-learning machinery are cited, not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from bidrig.detectors.base import GBMOn, column_index, register
from bidrig.features import TABLE_FEATURES, group_features
from bidrig.metrics import auc

LIT_MEANS = tuple(f"lit_{s}_mean" for s in ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread"))


@dataclass
class PublishedScreen:
    """One statistic of the table as the score; sign chosen on the tuning split."""

    feature: str
    sign: float = 1.0

    def _col(self, X: np.ndarray) -> np.ndarray:
        return X[:, column_index(TABLE_FEATURES, [self.feature])[0]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PublishedScreen":
        col = self._col(X)
        ok = np.isfinite(col)
        self.sign = 1.0 if auc(col[ok & (y == 1)], col[ok & (y == 0)]) >= 0.5 else -1.0
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        s = self.sign * self._col(X)
        return np.where(np.isfinite(s), s, np.nanmin(s) if np.isfinite(s).any() else 0.0)


class _Prepared:
    """Median-impute and standardise the literature screens on the tuning rows."""

    def _prepare(self, X: np.ndarray, fit: bool) -> np.ndarray:
        cols = column_index(TABLE_FEATURES, LIT_MEANS)
        Z = X[:, cols].astype(float)
        if fit:
            self.med = np.where(np.isfinite(np.nanmedian(Z, axis=0)), np.nanmedian(Z, axis=0), 0.0)
        Z = np.where(np.isfinite(Z), Z, self.med)
        if fit:
            self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0) + 1e-9
        return (Z - self.mu) / self.sd


class MixtureCluster(_Prepared):
    """Unsupervised mixture model on the screens (idea of Silveira et al., 2022).

    The suspicious cluster is the one with the lowest mean coefficient of variation.
    """

    def __init__(self, components: int = 3):
        self.components = components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MixtureCluster":
        Z = self._prepare(X, fit=True)
        self.gm = GaussianMixture(self.components, covariance_type="full", random_state=0).fit(Z)
        self.target = int(np.argmin(self.gm.means_[:, 0]))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.gm.predict_proba(self._prepare(X, fit=False))[:, self.target]


class OutlierForest(_Prepared):
    """Unsupervised isolation forest on the screens (anomaly-detection baseline)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OutlierForest":
        Z = self._prepare(X, fit=True)
        self.forest = IsolationForest(n_estimators=200, random_state=0).fit(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.forest.score_samples(self._prepare(X, fit=False))


PUBLISHED = {
    "pub:variance_screen": "lit_cv_mean",
    "pub:relative_distance": "lit_rd_mean",
    "pub:benford": "bl_benford_chi2",
    "pub:randomisation_groups": "bl_cd_group",
    "pub:cobidding_network": "bl_wk_group",
    "pub:bayesian_sequential": "seq_hmm_max",
}
for _name, _feature in PUBLISHED.items():
    register(_name)(lambda f=_feature: PublishedScreen(f))
register("pub:ml_screens")(lambda: GBMOn(group_features(["lit"]), TABLE_FEATURES))
register("pub:mixture_cluster")(MixtureCluster)
register("pub:isolation_forest")(OutlierForest)

PUBLISHED_DETECTORS = tuple(PUBLISHED) + (
    "pub:ml_screens",
    "pub:mixture_cluster",
    "pub:isolation_forest",
)
