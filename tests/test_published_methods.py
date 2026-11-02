import numpy as np
import pytest

from bidrig.detectors import REGISTRY, make
from bidrig.detectors.published_methods import PUBLISHED_DETECTORS
from bidrig.features import ALL_FEATURES, BASE_FEATURES, TABLE_FEATURES, market_features
from bidrig.features.baselines import (
    benford_chi2,
    cd_group,
    first_digits,
    group_score,
    jaccard,
    pair_distance,
    shuffle_among_participants,
    tightest_group,
    wk_group,
)


def test_table_columns_extend_the_main_features():
    assert TABLE_FEATURES[: len(ALL_FEATURES)] == ALL_FEATURES
    assert TABLE_FEATURES[len(ALL_FEATURES) :] == BASE_FEATURES


def test_first_digits():
    x = np.array([0.0123, 4.5, 920.0, 1.0, np.nan, -3.0, 0.0])
    assert first_digits(x).tolist() == [1, 4, 9, 1]


def bids_with_gaps(gaps):
    rows = [np.r_[100.0, 100.0 + g, 100.0 + 2 * g] for g in gaps]
    return np.array(rows)


def test_benford_statistic_separates_benford_from_uniform_digits():
    rng = np.random.default_rng(0)
    benford_like = 10 ** rng.uniform(0, 3, size=200)  # log-uniform: first digits follow Benford
    uniform_digit = rng.integers(1, 10, size=200) * 10.0 ** rng.integers(0, 3, size=200)
    assert benford_chi2(bids_with_gaps(benford_like)) < benford_chi2(bids_with_gaps(uniform_digit))


def test_benford_returns_nan_with_too_few_gaps():
    assert np.isnan(benford_chi2(bids_with_gaps([1.0, 2.0])))


def test_shuffle_keeps_mask_and_each_tenders_bids():
    rng = np.random.default_rng(1)
    bids = rng.uniform(1, 2, size=(10, 6))
    bids[rng.random(bids.shape) < 0.3] = np.nan
    out = shuffle_among_participants(bids, np.random.default_rng(2))
    assert np.array_equal(np.isnan(out), np.isnan(bids))
    for a, b in zip(bids, out):
        assert np.allclose(np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)]))


def planted_market(n_tenders=40, n_firms=8, seed=3):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.05, size=(n_tenders, n_firms))
    r[:, 1] = r[:, 0] + rng.normal(0, 0.002, n_tenders)  # firms 0, 1, 2 bid almost alike
    r[:, 2] = r[:, 0] + rng.normal(0, 0.002, n_tenders)
    estimate = np.full(n_tenders, 100.0)
    return 100.0 * np.exp(r), estimate


def test_tightest_group_finds_the_planted_trio():
    bids, estimate = planted_market()
    from bidrig.features import residuals

    d = pair_distance(residuals(bids, estimate), ~np.isnan(bids))
    assert tightest_group(d) < 0.01


def test_cd_group_is_large_for_a_planted_group_and_deterministic():
    bids, estimate = planted_market()
    a = cd_group(bids, estimate, np.random.default_rng(4))
    b = cd_group(bids, estimate, np.random.default_rng(4))
    assert a == b
    honest = np.exp(np.random.default_rng(9).normal(0, 0.05, size=bids.shape)) * 100.0
    assert a > cd_group(honest, estimate, np.random.default_rng(4))


def test_jaccard_is_symmetric_and_zero_on_the_diagonal():
    mask = np.random.default_rng(5).random((30, 6)) < 0.6
    w = jaccard(mask)
    assert np.allclose(w, w.T) and np.all(np.diag(w) == 0)
    assert w.min() >= 0 and w.max() <= 1


def test_group_score_prefers_a_planted_group():
    rng = np.random.default_rng(6)
    mask = rng.random((60, 8)) < 0.5
    mask[:, 1] = mask[:, 0]
    mask[:, 2] = mask[:, 0]  # firms 0, 1, 2 always bid together
    plain = rng.random((60, 8)) < 0.5
    assert group_score(jaccard(mask)) > group_score(jaccard(plain))


def test_wk_group_is_deterministic():
    mask = np.random.default_rng(7).random((40, 7)) < 0.6
    assert wk_group(mask, np.random.default_rng(8)) == wk_group(mask, np.random.default_rng(8))


def test_market_features_include_baseline_columns():
    rng = np.random.default_rng(10)
    bids = np.exp(rng.normal(0, 0.05, size=(30, 8))) * 100.0
    f = market_features(bids, np.full(30, 100.0), np.random.default_rng(11))
    assert all(k in f for k in TABLE_FEATURES)


def table(n=300, seed=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(TABLE_FEATURES)))
    y = (np.arange(n) % 4 == 0).astype(int)
    X[y == 1, TABLE_FEATURES.index("bl_cd_group")] -= 3.0  # signal in the wrong direction
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y


@pytest.mark.parametrize("name", PUBLISHED_DETECTORS)
def test_every_published_detector_fits_scores_and_is_deterministic(name):
    X, y = table()
    a = make(name).fit(X, y).score(X)
    b = make(name).fit(X, y).score(X)
    assert a.shape == (len(y),) and np.all(np.isfinite(a))
    assert np.array_equal(a, b)


def test_single_statistic_detectors_learn_the_sign_on_tuning_data():
    X, y = table()
    s = make("pub:randomisation_groups").fit(X, y).score(X)
    assert s[y == 1].mean() > s[y == 0].mean()


@pytest.mark.parametrize("name", ("pub:mixture_cluster", "pub:isolation_forest"))
def test_unsupervised_detectors_ignore_the_labels(name):
    X, y = table()
    a = make(name).fit(X, y).score(X)
    b = make(name).fit(X, 1 - y).score(X)
    assert np.array_equal(a, b)


def test_published_detectors_are_registered():
    assert set(PUBLISHED_DETECTORS) <= set(REGISTRY)
