"""Group ``part``: participation structure (bidding rate, thin tenders, and a greedy group of
firms that rarely bid together)."""

from __future__ import annotations

import numpy as np

PART_FEATURES = ("part_rate", "part_low", "part_group")


def exclusive_group(z: np.ndarray, join: float = -1.0) -> float:
    """Greedy group of firms that rarely bid together.

    ``z`` is the symmetric phi*sqrt(T) participation matrix (NaN = undefined). Start from
    the most negative pair and add the firm with the most negative mean z to the group
    while that mean is below ``join``. Score = -(sum of within-group z) / sqrt(#pairs).
    """
    zz = np.where(np.isfinite(z), z, 0.0)
    zz = np.minimum(zz, zz.T) if not np.allclose(zz, zz.T) else zz
    n = zz.shape[0]
    if n < 2:
        return np.nan
    off = zz + np.diag(np.full(n, np.inf))
    i, j = np.unravel_index(int(np.argmin(off)), off.shape)
    group = [int(i), int(j)]
    while len(group) < n:
        rest = np.array([k for k in range(n) if k not in group])
        means = zz[np.ix_(rest, group)].mean(axis=1)
        best = int(np.argmin(means))
        if means[best] >= join:
            break
        group.append(int(rest[best]))
    g = np.array(sorted(group))
    sub = zz[np.ix_(g, g)][np.triu_indices(len(g), k=1)]
    return float(-sub.sum() / np.sqrt(sub.size))


def participation_features(mask: np.ndarray) -> dict[str, float]:
    t_count, n = mask.shape
    nb = mask.sum(axis=1)
    m = mask.astype(float)
    p = m.mean(axis=0)
    sd = np.sqrt(p * (1 - p))
    ok = sd > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = (((m - p).T @ (m - p)) / t_count) / np.outer(sd, sd)
    phi[~np.outer(ok, ok)] = np.nan
    np.fill_diagonal(phi, np.nan)
    return {
        "part_rate": float(nb.mean() / n),
        "part_low": float((nb <= 2).mean()),
        "part_group": exclusive_group(phi * np.sqrt(t_count)),
    }
