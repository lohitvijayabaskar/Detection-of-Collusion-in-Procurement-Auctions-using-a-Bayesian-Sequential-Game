"""Group ``change``: within-market change scans (Brownian-bridge statistics) on per-tender
series."""

from __future__ import annotations

import numpy as np

from bidrig.features.common import EPS, residuals

SERIES = ("res_sd", "gap", "nbids", "close")
CHANGE_FEATURES = tuple(f"chg_{k}" for k in SERIES)


def tender_series(bids: np.ndarray, estimate: np.ndarray) -> dict[str, np.ndarray]:
    mask = ~np.isnan(bids)
    r = residuals(bids, estimate)
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        res_sd = np.nanstd(np.where(mask, r, np.nan), axis=1)
        s = np.sort(bids, axis=1)
        gap = np.log(s[:, 1] / s[:, 0])
        close = (np.where(mask, bids, np.inf) <= s[:, :1] * 1.03).sum(axis=1) - 1
        close = close / np.maximum(n - 1, 1)
    res_sd[n < 3] = np.nan
    gap[n < 2] = np.nan
    return {"res_sd": res_sd, "gap": gap, "nbids": n / bids.shape[1], "close": close}


def bridge_scan(x: np.ndarray) -> float:
    """max |partial sum of standardised deviations| / sqrt(n): a change-in-mean scan."""
    v = x[np.isfinite(x)]
    if v.size < 8 or v.std() < EPS:
        return np.nan
    z = (v - v.mean()) / v.std()
    return float(np.abs(np.cumsum(z)).max() / np.sqrt(v.size))


def change_features(bids: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return {f"chg_{k}": bridge_scan(v) for k, v in tender_series(bids, estimate).items()}
