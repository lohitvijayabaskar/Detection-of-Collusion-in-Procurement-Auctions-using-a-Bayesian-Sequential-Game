import numpy as np
import pytest

from bidrig.collusion import MECHANISMS
from bidrig.features import (
    ALL_FEATURES,
    bridge_scan,
    cobid_features,
    cusum_max,
    hmm_filter,
    identity_stats,
    market_features,
    pair_correlation_z,
    permute_within_tenders,
    residuals,
    tender_screens,
)
from bidrig.simulator import MarketSpec, sample_config, simulate_market


def market(mech="honest", seed=0):
    rng = np.random.default_rng(seed)
    cfg = sample_config(MarketSpec(), rng)
    return simulate_market(cfg, MECHANISMS[mech], rng)


def test_tender_screens_known_values():
    bids = np.array([[1.0, 2.0, 3.0, np.nan], [1.0, 1.0, np.nan, np.nan]])
    s = tender_screens(bids)
    assert s["cv"][0] == pytest.approx(1.0 / 2.0)
    assert s["skew"][0] == pytest.approx(0.0)
    assert s["gap"][0] == pytest.approx(1.0)
    assert s["spread"][0] == pytest.approx(2.0)
    assert s["rd"][0] == pytest.approx(1.0 / np.std([2.0, 3.0], ddof=1))
    assert s["rdalt"][0] == pytest.approx(1.0)
    assert all(np.isnan(v[1]) for v in s.values())  # fewer than 3 bids


def test_residuals_invariant_to_uniform_markup_and_estimate_scale():
    m = market(seed=1)
    r1 = residuals(m.bids, m.estimate)
    r2 = residuals(m.bids * 1.3, m.estimate * 0.7)
    assert np.allclose(r1, r2, equal_nan=True)


def test_pair_correlation_matches_numpy():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(40, 4))
    r[:, 1] = r[:, 0] + 0.1 * rng.normal(size=40)
    mask = np.ones_like(r, bool)
    mask[:5, 2] = False
    z = pair_correlation_z(r, mask)
    c01 = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    assert z[0, 1] == pytest.approx(np.arctanh(c01) * np.sqrt(37))
    keep = mask[:, 2]
    c02 = np.corrcoef(r[keep, 0], r[keep, 2])[0, 1]
    assert z[0, 2] == pytest.approx(np.arctanh(c02) * np.sqrt(keep.sum() - 3))
    assert np.isnan(z[1, 0])  # lower triangle unused


def test_permutation_keeps_participation_and_bid_multisets():
    m = market(seed=2)
    r = residuals(m.bids, m.estimate)
    bp, rp = permute_within_tenders(m.bids, r, np.random.default_rng(0))
    assert np.array_equal(np.isnan(bp), np.isnan(m.bids))
    for t in range(m.bids.shape[0]):
        assert np.allclose(
            np.sort(bp[t][~np.isnan(bp[t])]), np.sort(m.bids[t][~np.isnan(m.bids[t])])
        )
    # residuals move with their bids
    idx = ~np.isnan(m.bids)
    a = np.log(bp[idx] / np.repeat(m.estimate, idx.sum(axis=1))) - rp[idx]
    b = np.log(m.bids[idx] / np.repeat(m.estimate, idx.sum(axis=1))) - r[idx]
    assert np.allclose(np.sort(a), np.sort(b))


def test_identity_stats_on_scripted_extremes():
    t, n = 40, 6
    rng = np.random.default_rng(3)
    base = 100 * np.exp(rng.normal(0, 0.3, t))
    bids = base[:, None] * np.exp(rng.normal(0, 0.1, (t, n)))
    est = base.copy()
    tied = bids.copy()
    tied[:, 1] = tied[:, 0] * 1.001  # firm 1 shadows firm 0
    a = identity_stats(bids, residuals(bids, est))
    b = identity_stats(tied, residuals(tied, est))
    assert b["near_tie_pair"] > a["near_tie_pair"]
    assert b["pair_corr_pos"] > a["pair_corr_pos"]


def test_cobid_detects_mutual_exclusion():
    t = 60
    mask = np.ones((t, 5), bool)
    mask[::2, 0] = False
    mask[1::2, 1] = False  # firms 0 and 1 never bid together
    assert cobid_features(mask)["cobid_min"] > 5


def test_change_and_sequential_respond_to_shift():
    flat = np.random.default_rng(0).normal(size=60)
    shifted = flat.copy()
    shifted[30:] += 2.0
    assert bridge_scan(shifted) > bridge_scan(flat)
    assert cusum_max(shifted) > cusum_max(flat)
    assert np.nanmean(hmm_filter(shifted)[35:]) > np.nanmean(hmm_filter(flat)[35:])


def test_market_features_complete_and_deterministic():
    m = market("cover_bidding", seed=4)
    f1 = market_features(m.bids, m.estimate, np.random.default_rng(9))
    f2 = market_features(m.bids, m.estimate, np.random.default_rng(9))
    assert set(f1) == set(ALL_FEATURES)
    assert all((np.isnan(f1[k]) and np.isnan(f2[k])) or f1[k] == f2[k] for k in ALL_FEATURES)


def test_dependence_features_invariant_to_uniform_markup():
    m = market(seed=5)
    f1 = market_features(m.bids, m.estimate, np.random.default_rng(1))
    f2 = market_features(m.bids * 1.2, m.estimate, np.random.default_rng(1))
    for k in ALL_FEATURES:
        if k.startswith(("dep_", "chg_", "seq_")) or "cv" in k or "skew" in k:
            assert f1[k] == pytest.approx(f2[k], nan_ok=True, rel=1e-6, abs=1e-9), k


def test_time_null_keeps_each_firms_residuals():
    from bidrig.features import permute_within_firms

    m = market(seed=6)
    r = residuals(m.bids, m.estimate)
    rp = permute_within_firms(r, np.random.default_rng(0))
    assert np.array_equal(np.isnan(rp), np.isnan(r))
    for j in range(r.shape[1]):
        assert np.allclose(np.sort(rp[~np.isnan(rp[:, j]), j]), np.sort(r[~np.isnan(r[:, j]), j]))
    assert not np.allclose(rp, r, equal_nan=True)
