import numpy as np
import pytest

from bidrig.detectors import (
    ALL_COMBINATIONS,
    METRIC_COMBINATIONS_BY_TIER,
    MultiDetectorSuite,
    MultiMetricGBM,
    make,
)
from bidrig.features import ALL_FEATURES, FEATURE_GROUPS
from bidrig.models import make_design


def synthetic(seed=42, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    y = (np.arange(n) % 3 == 0).astype(int)
    # Add signal to dependence and literature features
    X[y == 1, ALL_FEATURES.index("dep_pair_corr_pos")] += 2.0
    X[y == 1, ALL_FEATURES.index("lit_cv_mean")] -= 1.5
    return X, y


def test_metric_tiers_structure():
    assert set(METRIC_COMBINATIONS_BY_TIER.keys()) == {
        "all_metrics",
        "single_metric",
        "pair_metrics",
        "triplet_metrics",
        "quad_metrics",
    }
    assert "all_metrics_gbm" in METRIC_COMBINATIONS_BY_TIER["all_metrics"]
    all_groups = METRIC_COMBINATIONS_BY_TIER["all_metrics"]["all_metrics_gbm"]
    assert set(all_groups) == set(FEATURE_GROUPS.keys())


def test_every_combination_resolves_via_make():
    for name in ALL_COMBINATIONS:
        d = make(name)
        assert d is not None
        assert hasattr(d, "fit")
        assert hasattr(d, "score")


def test_all_metrics_uses_all_defined_features():
    d = make("all_metrics_gbm")
    assert len(d.features) == len(ALL_FEATURES)
    assert set(d.features) == set(ALL_FEATURES)


def test_suite_initialization_and_filtering():
    # All combinations
    suite_all = MultiDetectorSuite()
    assert len(suite_all.detectors) == len(ALL_COMBINATIONS)

    # By tier
    suite_pairs = MultiDetectorSuite(tiers=["pair_metrics"])
    assert len(suite_pairs.detectors) == len(METRIC_COMBINATIONS_BY_TIER["pair_metrics"])
    for d in suite_pairs.detectors.values():
        assert len(d.groups) == 2

    # Custom
    suite_custom = MultiDetectorSuite(combinations={"test_det": ("lit", "seq")})
    assert len(suite_custom.detectors) == 1
    assert suite_custom.detectors["test_det"].groups == ("lit", "seq")


def test_suite_describe():
    suite = MultiDetectorSuite(tiers=["all_metrics", "single_metric"])
    desc = suite.describe()
    assert len(desc) == len(METRIC_COMBINATIONS_BY_TIER["all_metrics"]) + len(
        METRIC_COMBINATIONS_BY_TIER["single_metric"]
    )
    all_met = next(item for item in desc if item["name"] == "all_metrics_gbm")
    assert all_met["num_metric_groups"] == 5
    assert all_met["num_features"] == len(ALL_FEATURES)


def test_suite_simultaneous_fit_and_score():
    X, y = synthetic()
    subset_names = [
        "all_metrics_gbm",
        "single_lit_gbm",
        "pair_lit_dep_gbm",
        "triplet_lit_dep_seq_gbm",
    ]
    subset_combinations = {name: ALL_COMBINATIONS[name] for name in subset_names}
    suite = MultiDetectorSuite(combinations=subset_combinations)

    # Simultaneously fit
    suite.fit(X, y)

    # Simultaneously score
    scores = suite.score(X)
    assert set(scores.keys()) == set(subset_names)
    for name, s in scores.items():
        assert s.shape == (len(y),)
        assert np.isfinite(s).all()
        assert (s >= 0.0).all() and (s <= 1.0).all()
        # Ensure model learned signal
        assert s[y == 1].mean() > s[y == 0].mean()


def test_make_design_resolves_multi_detector_names():
    assert make_design("all_metrics_gbm").name == "all_metrics_gbm"
    assert make_design("pair_lit_dep_gbm").name == "pair_lit_dep_gbm"
    assert make_design("single_part_gbm").name == "single_part_gbm"
