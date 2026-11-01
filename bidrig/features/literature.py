"""Group ``lit``: standard literature screens per tender (CV, skewness, kurtosis, relative
distance, RD-alt, normalised gap, spread), aggregated over a market's tenders."""

from __future__ import annotations

import numpy as np

from bidrig.features.common import EPS

LIT_SCREENS = ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread")
LIT_FEATURES = tuple(f"lit_{s}_{a}" for s in LIT_SCREENS for a in ("mean", "median", "sd"))


def tender_screens(bids: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tender literature screens. Tenders with fewer than three bids give NaN."""
    mask = ~np.isnan(bids)
    n = mask.sum(axis=1)
    ok = n >= 3
    s = np.sort(bids, axis=1)  # NaN sorted to the end
    b0 = np.where(mask, bids, 0.0)
    mean = b0.sum(axis=1) / np.maximum(n, 1)
    dev = np.where(mask, bids - mean[:, None], 0.0)
    m2 = (dev**2).sum(axis=1) / np.maximum(n, 1)
    m3 = (dev**3).sum(axis=1) / np.maximum(n, 1)
    m4 = (dev**4).sum(axis=1) / np.maximum(n, 1)
    sd = np.sqrt(m2 * n / np.maximum(n - 1, 1))
    first, second = s[:, 0], s[:, 1]
    gap12 = second - first
    idx = np.arange(s.shape[1])[None, :]
    losing = np.where((idx >= 1) & (idx < n[:, None]), s, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lose_sd = np.nanstd(losing, axis=1, ddof=1) if s.shape[1] > 2 else np.full(len(n), np.nan)
        diffs = np.diff(losing, axis=1)
        lose_gap = np.nanmean(diffs, axis=1)
        top = s[np.arange(len(n)), np.maximum(n - 1, 0)]
        out = {
            "cv": sd / mean,
            "skew": m3 / np.maximum(m2, EPS) ** 1.5,
            "kurt": m4 / np.maximum(m2, EPS) ** 2 - 3.0,
            "rd": gap12 / lose_sd,
            "rdalt": gap12 / lose_gap,
            "gap": gap12 / first,
            "spread": (top - first) / first,
        }
    for k in out:
        v = out[k]
        v[~ok | ~np.isfinite(v)] = np.nan
    return out


def _agg(values: np.ndarray) -> tuple[float, float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(v.mean()), float(np.median(v)), float(v.std())


def literature_features(bids: np.ndarray) -> dict[str, float]:
    feats = {}
    for name, v in tender_screens(bids).items():
        mean, med, sd = _agg(v)
        feats[f"lit_{name}_mean"] = mean
        feats[f"lit_{name}_median"] = med
        feats[f"lit_{name}_sd"] = sd
    return feats
