import numpy as np
import pytest

import bidrig.detectors as detectors
from bidrig.detectors.base import GBMOn, column_index, make, register
from bidrig.detectors.previous import drop_time_null, full_gbm_features
from bidrig.features import ALL_FEATURES
from bidrig.models import DESIGNS, GBM, make_design


def synthetic(seed=0, n=240):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    y = (np.arange(n) % 4 == 0).astype(int)
    X[y == 1, ALL_FEATURES.index("dep_pair_corr_pos")] += 3.0
    return X, y


def test_registry_contains_previous():
    assert "previous" in detectors.REGISTRY
    assert make("previous").name == "previous"


def test_register_rejects_duplicates_and_unknown_names_fail():
    with pytest.raises(ValueError):
        register("previous")(lambda: None)
    with pytest.raises(KeyError):
        make("no_such_detector")


def test_drop_time_null_removes_exactly_the_t_columns():
    full = full_gbm_features()
    prev = drop_time_null(full)
    removed = set(full) - set(prev)
    assert removed and all(f.endswith("_t") for f in removed)
    assert all(not f.endswith("_t") for f in prev)
    assert set(prev) <= set(full) and len(prev) == len(full) - len(removed)
    # nothing outside the dependence group carries the suffix
    assert all(f.startswith("dep_") for f in removed)


def test_full_gbm_features_match_the_full_gbm_design():
    assert full_gbm_features() == GBM(DESIGNS["full_gbm"]).features


def test_previous_fit_score_shape_and_determinism():
    X, y = synthetic()
    a = make("previous").fit(X, y).score(X)
    b = make("previous").fit(X, y).score(X)
    assert a.shape == (len(y),) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert a[y == 1].mean() > a[y == 0].mean()


def test_previous_ignores_the_time_null_columns():
    X, y = synthetic()
    X2 = X.copy()
    cols = [i for i, f in enumerate(ALL_FEATURES) if f.endswith("_t")]
    X2[:, cols] = np.random.default_rng(9).normal(size=(len(y), len(cols))) * 50
    a = make("previous").fit(X, y).score(X)
    b = make("previous").fit(X2, y).score(X2)
    assert np.array_equal(a, b)


def test_make_design_resolves_registered_and_builtin_names():
    assert make_design("previous").name == "previous"
    assert isinstance(make_design("full_gbm"), GBM)
    with pytest.raises(KeyError):
        make_design("definitely_not_a_design")


def test_column_index_reports_missing_features():
    assert column_index(("a", "b", "c"), ("c", "a")) == [2, 0]
    with pytest.raises(KeyError):
        column_index(("a",), ("z",))


def test_gbmon_matches_gbm_on_the_same_columns():
    X, y = synthetic(seed=3)
    feats = GBM(DESIGNS["full_gbm"]).features
    a = GBMOn(feats).fit(X, y).score(X)
    b = GBM(DESIGNS["full_gbm"]).fit(X, y).score(X)
    assert np.array_equal(a, b)
