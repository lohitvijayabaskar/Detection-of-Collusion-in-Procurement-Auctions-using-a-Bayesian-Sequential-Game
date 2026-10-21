"""Evaluation metrics and bootstrap confidence intervals.

Higher scores mean "more suspicious". Every metric is sanity-tested on separated, reversed
and tied data (tests/test_metrics.py) before use.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score_pos > score_neg) + 0.5 P(tie). NaN scores count as lowest."""
    pos = np.nan_to_num(np.asarray(pos, float), nan=-np.inf)
    neg = np.nan_to_num(np.asarray(neg, float), nan=-np.inf)
    if pos.size == 0 or neg.size == 0:
        return np.nan
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def threshold_at_far(neg_scores: np.ndarray, far: float = 0.05) -> float:
    """Smallest threshold with empirical false-alarm rate (score > thr) <= far on ``neg``."""
    s = np.sort(np.nan_to_num(np.asarray(neg_scores, float), nan=-np.inf))
    if s.size == 0:
        return np.inf
    allowed = int(np.floor(far * s.size))  # number of negatives allowed above threshold
    return float(s[s.size - allowed - 1])


def alarm_rate(scores: np.ndarray, thr: float) -> float:
    s = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    return float((s > thr).mean()) if s.size else np.nan


def precision_at_frac(scores: np.ndarray, labels: np.ndarray, frac: float = 0.10) -> float:
    """Share of positives among the top ``frac`` of markets. Ties at the cut are averaged.

    With ties straddling the cut, each tied item contributes its label weighted by the
    share of the tied block that falls inside the top-k, so the result does not depend on
    the input order.
    """
    s = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    y = np.asarray(labels, float)
    k = max(1, int(round(frac * s.size)))
    order = np.argsort(-s, kind="stable")
    cut = s[order[k - 1]]
    above = s > cut
    tied = s == cut
    need = k - above.sum()
    total = y[above].sum() + y[tied].mean() * need
    return float(total / k)


def bootstrap_ci(
    groups: Sequence[dict[str, np.ndarray]],
    stat: Callable[[list[dict[str, np.ndarray]]], float],
    rng: np.random.Generator,
    n_boot: int = 1000,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Two-level bootstrap: resample seeds, then markets within each array of each seed.

    ``groups`` is one dict per seed mapping a name to an array whose first axis is markets
    (arrays sharing a first-axis length within a seed and listed in ``paired`` are resampled
    together, see ``resample_seed``). ``stat`` maps a list of per-seed dicts to a number.
    Returns (point estimate, lower, upper).
    """
    return bootstrap_many(groups, {"s": stat}, rng, n_boot, level)["s"]


def bootstrap_many(
    groups: Sequence[dict[str, np.ndarray]],
    stats: dict[str, Callable[[list[dict[str, np.ndarray]]], float]],
    rng: np.random.Generator,
    n_boot: int = 1000,
    level: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """``bootstrap_ci`` for several statistics sharing the same resamples (paired CIs)."""
    names = sorted(stats)
    point = {k: stats[k](list(groups)) for k in names}
    vals = np.empty((n_boot, len(names)))
    n = len(groups)
    for b in range(n_boot):
        seeds = rng.integers(0, n, n)
        sample = [resample_seed(groups[i], rng) for i in seeds]
        vals[b] = [stats[k](sample) for k in names]
    q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    out = {}
    for j, k in enumerate(names):
        lo, hi = np.nanpercentile(vals[:, j], q)
        out[k] = (float(point[k]), float(lo), float(hi))
    return out


def resample_seed(group: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Resample markets within each set. Keys sharing a prefix before '|' are resampled with
    the same indices (paired scores of different detectors on the same markets)."""
    out, idx_cache = {}, {}
    for key in sorted(group):
        arr = group[key]
        base = key.split("|")[0]
        if base not in idx_cache:
            idx_cache[base] = rng.integers(0, len(arr), len(arr))
        out[key] = arr[idx_cache[base]]
    return out
