import numpy as np
import pytest

from bidrig.metrics import (
    alarm_rate,
    auc,
    bootstrap_ci,
    bootstrap_many,
    precision_at_frac,
    resample_seed,
    threshold_at_far,
)


def test_auc_separated_reversed_tied():
    pos, neg = np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])
    assert auc(pos, neg) == 1.0
    assert auc(neg, pos) == 0.0
    assert auc(np.ones(5), np.ones(7)) == 0.5


def test_auc_partial_and_nan():
    assert auc(np.array([1.0, 3.0]), np.array([2.0])) == 0.5
    # NaN scores rank lowest
    assert auc(np.array([np.nan]), np.array([0.0])) == 0.0


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 30).astype(float)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auc(pos, neg) == pytest.approx(brute)


def test_threshold_far():
    neg = np.arange(100, dtype=float)
    thr = threshold_at_far(neg, 0.05)
    assert alarm_rate(neg, thr) == 0.05
    tied = np.zeros(50)
    assert alarm_rate(tied, threshold_at_far(tied, 0.05)) == 0.0


def test_precision_separated_reversed_tied():
    y = np.r_[np.ones(10), np.zeros(90)]
    s = np.r_[np.ones(10) * 2, np.zeros(90)]
    assert precision_at_frac(s, y, 0.10) == 1.0
    assert precision_at_frac(-s, y, 0.10) == 0.0
    assert precision_at_frac(np.zeros(100), y, 0.10) == pytest.approx(0.10)
    # order of tied inputs must not matter
    perm = np.random.default_rng(1).permutation(100)
    assert precision_at_frac(np.zeros(100), y[perm], 0.10) == pytest.approx(0.10)


def test_resample_paired_keys_share_indices():
    g = {"a|x": np.arange(10), "a|y": np.arange(10) * 2, "b|x": np.arange(5)}
    r = resample_seed(g, np.random.default_rng(0))
    assert np.array_equal(r["a|y"], r["a|x"] * 2)


def test_bootstrap_ci_covers_truth_and_orders():
    rng = np.random.default_rng(0)
    groups = [{"x|a": rng.normal(1.0, 1.0, 200)} for _ in range(10)]
    stat = lambda gs: float(np.mean([g["x|a"].mean() for g in gs]))  # noqa: E731
    point, lo, hi = bootstrap_ci(groups, stat, np.random.default_rng(1), n_boot=300)
    assert lo < 1.0 < hi and lo < point < hi
    many = bootstrap_many(groups, {"m": stat}, np.random.default_rng(1), n_boot=300)
    assert many["m"] == (point, lo, hi)
