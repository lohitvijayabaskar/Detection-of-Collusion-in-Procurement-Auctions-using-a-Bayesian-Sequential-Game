import numpy as np
import pytest

import bidrig.detectors as detectors
from bidrig.detectors.base import make
from bidrig.detectors.stat_auc import (
    CHOSEN,
    GRID,
    PENALTIES,
    SUBSETS,
    StatAUC,
    grid_name,
    subset_features,
)
from bidrig.features import ALL_FEATURES, FEATURE_GROUPS


def synthetic(seed=0, n=300, nan_share=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    y = (np.arange(n) % 4 == 0).astype(int)
    X[y == 1, ALL_FEATURES.index("dep_pair_corr_pos")] += 2.5
    X[rng.random(X.shape) < nan_share] = np.nan
    return X, y


def test_grid_is_registered_and_complete():
    assert len(GRID) == len(SUBSETS) * len(PENALTIES) == 16
    for s, c in GRID:
        assert grid_name(s, c) in detectors.REGISTRY
        assert make(grid_name(s, c)).subset == s


def test_subset_feature_counts_follow_groups():
    assert set(subset_features("dep")) < set(subset_features("lit_dep"))
    assert set(subset_features("lit_dep")) < set(subset_features("all"))
    assert subset_features("all") == tuple(
        f for g in ("lit", "dep", "change", "seq") for f in FEATURE_GROUPS[g]
    )


def test_fit_score_with_nans_finite_deterministic():
    X, y = synthetic()
    a = StatAUC("lit_dep", 1.0).fit(X, y).score(X)
    b = StatAUC("lit_dep", 1.0).fit(X, y).score(X)
    assert a.shape == (len(y),) and np.isfinite(a).all() and np.array_equal(a, b)
    assert a[y == 1].mean() > a[y == 0].mean()


def test_all_nan_column_is_handled():
    X, y = synthetic(nan_share=0.0)
    col = ALL_FEATURES.index(subset_features("dep")[0])
    X[:, col] = np.nan
    s = StatAUC("dep", 1.0).fit(X, y).score(X)
    assert np.isfinite(s).all()


def test_scaling_uses_training_statistics_only():
    X, y = synthetic(nan_share=0.0)
    m = StatAUC("dep", 1.0).fit(X, y)
    Xt = X.copy()
    Xt[:, m._cols] += 100.0  # a shifted test table must not re-centre the model
    assert not np.allclose(m.score(X), m.score(Xt))


def test_stronger_penalty_shrinks_scores_toward_the_base_rate():
    X, y = synthetic(nan_share=0.0)
    weak = StatAUC("dep", 10.0).fit(X, y).score(X)
    strong = StatAUC("dep", 0.01).fit(X, y).score(X)
    assert strong.std() < weak.std()


def test_unknown_subset_rejected():
    with pytest.raises(KeyError):
        StatAUC("nope", 1.0)


def test_frozen_choice_is_a_grid_member_and_registered_when_set():
    if CHOSEN is None:
        assert "stat_auc" not in detectors.REGISTRY
    else:
        assert CHOSEN in GRID and "stat_auc" in detectors.REGISTRY
        assert make("stat_auc").name == "stat_auc"


def test_frozen_choice_matches_the_logged_selection():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "stat_auc_selection_stat_auc_select_v1.json"
    )
    if not path.exists():
        pytest.skip("selection record not present")
    chosen = json.loads(path.read_text())["chosen"]
    assert (chosen["subset"], chosen["C"]) == CHOSEN
