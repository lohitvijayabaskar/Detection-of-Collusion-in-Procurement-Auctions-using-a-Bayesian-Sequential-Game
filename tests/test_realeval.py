"""Tender-level real-data evaluation: checked on synthetic frames only."""

import numpy as np
import pandas as pd
import pytest

from bidrig.realeval import (
    CLOSE_TOL,
    MIN_CO_PAIR,
    PAIR_FEATURES,
    block_indices,
    dataset_table,
    lodo,
    pair_features,
    percentile,
)


def synth_bids(seed=0, t=40, n=8, p=0.7):
    rng = np.random.default_rng(seed)
    est = 100 * np.exp(rng.normal(0, 0.3, t))
    bids = est[:, None] * np.exp(rng.normal(0, 0.08, (t, n)))
    bids[rng.random((t, n)) > p] = np.nan
    bids[:, 1] = np.where(np.isnan(bids[:, 0]), np.nan, bids[:, 0] * 1.01)  # a shadow pair
    return bids, est


def brute(bids, est, t):
    """Pair statistics for tender t recomputed from scratch on the other tenders."""
    keep = np.arange(len(est)) != t
    b, e = bids[keep], est[keep]
    x = np.log(b / e[:, None])
    r = x - np.nanmedian(x, axis=1)[:, None]
    m = ~np.isnan(b)
    idx = np.flatnonzero(~np.isnan(bids[t]))
    corr, close = [], []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            co = m[:, i] & m[:, j]
            if co.sum() < MIN_CO_PAIR:
                continue
            c = np.corrcoef(r[co, i], r[co, j])[0, 1]
            corr.append(np.arctanh(np.clip(c, -0.999999, 0.999999)) * np.sqrt(co.sum() - 3))
            close.append(np.mean(np.abs(r[co, i] - r[co, j]) < CLOSE_TOL))
    return max(corr), max(close)


def test_leave_tender_out_pair_stats_match_bruteforce():
    bids, est = synth_bids()
    f = pair_features(bids, est)
    for t in (0, 7, 23, 39):
        c, cl = brute(bids, est, t)
        assert f["pair_corr_max"][t] == pytest.approx(c, rel=1e-6)
        assert f["pair_close_max"][t] == pytest.approx(cl, rel=1e-9)


def test_shadow_pair_raises_pair_features():
    bids, est = synth_bids(seed=1)
    f = pair_features(bids, est)
    both = ~np.isnan(bids[:, 0])
    assert np.nanmean(f["pair_close_max"][both]) > np.nanmean(f["pair_close_max"][~both])


def test_percentile_scale_free_and_nan():
    v = np.array([3.0, 1.0, np.nan, 2.0])
    assert np.allclose(percentile(v), [1.0, 1 / 3, np.nan, 2 / 3], equal_nan=True)
    assert np.allclose(percentile(v * 1e6), percentile(v), equal_nan=True)


def test_block_indices():
    idx = block_indices(95, np.random.default_rng(0), block=20)
    assert len(idx) == 95 and idx.min() >= 0 and idx.max() < 95


def frame(seed, collusive_rows):
    bids, est = synth_bids(seed=seed, t=60, n=10)
    rows = []
    for t in range(bids.shape[0]):
        for j in range(bids.shape[1]):
            if not np.isnan(bids[t, j]):
                rows.append((t, t, j, bids[t, j], est[t], int(t in collusive_rows)))
    return pd.DataFrame(
        rows, columns=["tender", "order", "firm", "bid", "estimate", "tender_label"]
    )


def test_features_do_not_use_labels():
    a = dataset_table(frame(3, set(range(0, 60, 3))))
    b = dataset_table(frame(3, set(range(1, 60, 4))))
    assert np.allclose(a["X"], b["X"], equal_nan=True)
    assert set(a["names"]) >= set(PAIR_FEATURES)


def test_lodo_smoke():
    tables = {f"d{k}": dataset_table(frame(k, set(range(0, 60, 3)))) for k in range(3)}
    res = lodo(tables, ["real_lit_gbm", "real_full_gbm", "single:scr_cv"], n_boot=20)
    assert set(res) == {"d0", "d1", "d2"}
    r = res["d0"]["real_full_gbm"]
    assert 0 <= r["auc"]["point"] <= 1 and len(r["auc"]["ci"]) == 2
    assert "real_full_gbm-real_lit_gbm" in res["d0"]
