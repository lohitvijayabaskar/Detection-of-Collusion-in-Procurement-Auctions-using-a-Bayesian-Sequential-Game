"""All features of one market, from public data only (bids, bidder ids, estimates)."""

from __future__ import annotations

import warnings

import numpy as np

from bidrig.features.baselines import baseline_features
from bidrig.features.change import change_features
from bidrig.features.dependence import dependence_features
from bidrig.features.literature import literature_features
from bidrig.features.participation import participation_features
from bidrig.features.sequential import sequential_features


def market_features(bids: np.ndarray, estimate: np.ndarray, rng: np.random.Generator) -> dict:
    feats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats.update(literature_features(bids))
        feats.update(dependence_features(bids, estimate, rng))
        feats.update(participation_features(~np.isnan(bids)))
        feats.update(change_features(bids, estimate))
        feats.update(sequential_features(bids, estimate))
        base_rng = np.random.default_rng(int(rng.integers(0, 2**62)))
        feats.update(baseline_features(bids, estimate, base_rng))
    return feats
