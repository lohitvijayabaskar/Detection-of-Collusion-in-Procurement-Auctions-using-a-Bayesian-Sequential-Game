"""Golden-output test: a refactor must not change features, groups or detector scores.

The reference file ``tests/golden/reference.json`` was written from the code BEFORE the
modularisation. To (re)write it on purpose, run ``PYTHONPATH=. python tests/test_golden.py --write``; do that
only when a behaviour change is intended and logged.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bidrig.collusion import MECHANISMS
from bidrig.detectors import REGISTRY, make
from bidrig.features import ALL_FEATURES, FEATURE_GROUPS, market_features
from bidrig.models import make_design
from bidrig.simulator import MarketSpec, sample_config, simulate_market

REF = Path(__file__).resolve().parent / "golden" / "reference.json"
MARKETS = ("honest", "rotation", "partial_fringe", "bid_suppression")
DETECTORS = (
    "full_gbm",
    "lit_gbm",
    "dep_gbm",
    "full_part_gbm",
    "lit_dep_part_mono",
    "single:lit_cv_mean",
    "tail_only",
    "full_fused",
    "previous",
    "stat_auc",
    "rl_screens",
)


def synthetic(seed=11, n=240):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    y = (np.arange(n) % 4 == 0).astype(int)
    X[y == 1, ALL_FEATURES.index("dep_pair_corr_pos")] += 2.5
    X[rng.random(X.shape) < 0.05] = np.nan
    return X, y


def clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 10)


def build():
    feats = {}
    for i, name in enumerate(MARKETS):
        rng = np.random.default_rng([7, i])
        m = simulate_market(sample_config(MarketSpec(), rng), MECHANISMS[name], rng)
        f = market_features(m.bids, m.estimate, np.random.default_rng([7, i, 1]))
        feats[name] = {k: clean(f[k]) for k in ALL_FEATURES}
    X, y = synthetic()
    scores = {}
    for name in DETECTORS:
        d = make_design(name).fit(X, y)
        scores[name] = [clean(s) for s in d.score(X)]
    return {
        "feature_names": list(ALL_FEATURES),
        "groups": {g: list(v) for g, v in FEATURE_GROUPS.items()},
        "market_features": feats,
        "scores": scores,
    }


@pytest.fixture(scope="module")
def reference():
    return json.loads(REF.read_text())


@pytest.fixture(scope="module")
def current():
    return build()


def test_feature_names_and_groups_unchanged(reference, current):
    assert current["feature_names"] == reference["feature_names"]
    assert current["groups"] == reference["groups"]


def test_market_features_unchanged(reference, current):
    for name in MARKETS:
        ref, cur = reference["market_features"][name], current["market_features"][name]
        for k in ALL_FEATURES:
            if ref[k] is None:
                assert cur[k] is None, (name, k)
            else:
                assert cur[k] == pytest.approx(ref[k], abs=1e-8), (name, k)


@pytest.mark.parametrize("name", DETECTORS)
def test_detector_scores_unchanged(reference, current, name):
    ref, cur = reference["scores"][name], current["scores"][name]
    assert len(ref) == len(cur)
    for a, b in zip(ref, cur):
        if a is None:
            assert b is None
        else:
            assert b == pytest.approx(a, abs=1e-8)


def test_every_reference_detector_is_resolvable():
    for name in DETECTORS:
        assert make_design(name) is not None
    assert {"previous", "stat_auc", "rl_screens"} <= set(REGISTRY)
    assert make("previous").name == "previous"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("pass --write to overwrite the reference")
    REF.parent.mkdir(exist_ok=True)
    REF.write_text(json.dumps(build()))
    print("wrote", REF)
