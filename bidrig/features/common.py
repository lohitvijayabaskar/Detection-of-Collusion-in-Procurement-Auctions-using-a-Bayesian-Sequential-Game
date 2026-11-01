"""Shared constants and small helpers for the feature modules.

Residuals are ``log(bid / estimate)`` minus the tender median, which removes the estimate,
tender-level shocks and any uniform markup, so no feature measures the price level.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
MIN_CO = 6  # co-bid tenders needed for a pair correlation
MIN_BIDS = 6  # bids needed for a firm dispersion
MIN_WINS = 4  # wins needed for cover tracking
N_PERM = 30
TIE_TOL = 0.005
CLOSE_TOL = 0.03


def residuals(bids: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log(bids / estimate[:, None])
    med = np.nanmedian(x, axis=1)
    return x - med[:, None]


def topk_mean(values: np.ndarray, k: int = 3) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    v = np.sort(v)[::-1]
    return float(v[: min(k, v.size)].mean())


def robust_z(x: np.ndarray) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size < 5:
        return np.full_like(x, np.nan)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    scale = mad if mad > EPS else (v.std() + EPS)
    return (x - med) / scale
