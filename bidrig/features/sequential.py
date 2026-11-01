"""Group ``seq``: sequential layer, a CUSUM and a two-state HMM filter on a per-tender suspicion
series standardised against the market's own history."""

from __future__ import annotations

import numpy as np

from bidrig.features.change import tender_series
from bidrig.features.common import EPS, robust_z

SEQ_FEATURES = ("seq_cusum", "seq_hmm_max", "seq_hmm_mean")


def suspicion_series(bids: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Per-tender suspicion: low dispersion, small gap, many bids close to the winner."""
    s = tender_series(bids, estimate)
    parts = [-robust_z(s["res_sd"]), -robust_z(s["gap"]), robust_z(s["close"])]
    stack = np.vstack(parts)
    with np.errstate(invalid="ignore"):
        u = np.nanmean(stack, axis=0)
    return np.clip(u, -5.0, 5.0)


def cusum_max(u: np.ndarray, drift: float = 0.5) -> float:
    s, best = 0.0, 0.0
    for x in u:
        if not np.isfinite(x):
            continue
        s = max(0.0, s + x - drift)
        best = max(best, s)
    return best


def hmm_filter(u: np.ndarray, mu1: float = 1.5, stay: float = 0.95, prior: float = 0.1):
    """Forward filter of a two-state HMM with N(0,1) / N(mu1,1) emissions; P(state 1 | u_1:t)."""
    post = np.full(len(u), np.nan)
    p = prior
    for t, x in enumerate(u):
        p = stay * p + (1 - stay) * (1 - p)
        if np.isfinite(x):
            l1 = np.exp(-0.5 * (x - mu1) ** 2)
            l0 = np.exp(-0.5 * x**2)
            p = p * l1 / (p * l1 + (1 - p) * l0 + EPS)
        post[t] = p
    return post


def sequential_features(bids: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    u = suspicion_series(bids, estimate)
    if np.isfinite(u).sum() < 5:
        return {"seq_cusum": np.nan, "seq_hmm_max": np.nan, "seq_hmm_mean": np.nan}
    post = hmm_filter(u)
    return {
        "seq_cusum": cusum_max(u) / np.sqrt(np.isfinite(u).sum()),
        "seq_hmm_max": float(np.nanmax(post)),
        "seq_hmm_mean": float(np.nanmean(post)),
    }
