"""Common interface and registry for detector variants.

A detector maps a feature matrix (rows = markets, columns = ``names``) to suspicion scores
(higher = more suspicious). ``fit`` receives the tuning split only; a variant that needs the
validation split must say so in its own module and take it as an explicit argument.

Variant modules depend on this file and on ``bidrig.features`` only, never on each other.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bidrig.detectors.config import GBM_PARAMS
from bidrig.features import ALL_FEATURES


class Detector(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Detector": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...


def column_index(names: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """Positions of ``wanted`` in ``names``; raises ``KeyError`` for a missing feature."""
    index = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f"features not in table: {missing}")
    return [index[w] for w in wanted]


class GBMOn:
    """Gradient-boosted classifier on a named subset of features, using the shared settings."""

    name = "gbm_on"

    def __init__(
        self,
        features: Sequence[str],
        names: Sequence[str] = ALL_FEATURES,
        monotone: dict | None = None,
    ):
        self.features = tuple(features)
        self.monotone = monotone  # feature -> +1 / -1 constraint (knowledge prior), or None
        self._cols = column_index(names, self.features)
        self.model: HistGradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMOn":
        params = dict(GBM_PARAMS)
        if self.monotone:
            params["monotonic_cst"] = [self.monotone.get(f, 0) for f in self.features]
        self.model = HistGradientBoostingClassifier(**params)
        self.model.fit(X[:, self._cols], y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self._cols])[:, 1]


REGISTRY: dict[str, Callable[[], Detector]] = {}


def register(name: str):
    """Decorator: ``@register("previous")`` adds a zero-argument factory to the registry."""

    def wrap(factory):
        if name in REGISTRY:
            raise ValueError(f"detector {name!r} is already registered")
        REGISTRY[name] = factory
        return factory

    return wrap


def make(name: str) -> Detector:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]()
