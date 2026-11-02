"""Group ``base``: statistics adapted from published detectors that the main features lack.

These are our own adaptations of three published ideas, each reduced to one market-level
number; none is the authors' code. Compared with the original papers they differ as follows.

``bl_benford_chi2``  Benford's-law digit test (practitioner use in procurement): chi-square
    distance, per gap, between the first-digit distribution of the gaps from the lowest bid to
    every other bid and Benford's law.
``bl_cd_group``  randomisation test on bidder groups (idea of Conley and Decarolis, 2016): how
    much closer than chance the tightest group of three firms bids, standardised against
    permutations that shuffle bids among the participants of each tender.
``bl_wk_group``  co-bidding network groups (idea of Wachs and Kertesz, 2019): coherence times
    exclusivity of the best group in the Jaccard co-participation network, standardised
    against permutations that shuffle participants within each tender's size.
"""

from __future__ import annotations

import numpy as np

from bidrig.features.common import EPS, MIN_CO, N_PERM, residuals

BASE_FEATURES = ("bl_benford_chi2", "bl_cd_group", "bl_wk_group")
BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))
GROUP_SIZE = 3
MIN_GAP_COUNT = 20
MAX_NETWORK_GROUP = 8


def first_digits(x: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x) & (x > 0)]
    e = np.floor(np.log10(x))
    return np.floor(x / 10.0**e).astype(int).clip(1, 9)


def benford_chi2(bids: np.ndarray) -> float:
    gaps = []
    for row in bids:
        b = np.sort(row[~np.isnan(row)])
        if b.size >= 3:
            gaps.append(b[1:] - b[0])
    if not gaps:
        return np.nan
    g = np.concatenate(gaps)
    g = g[g > EPS]
    if g.size < MIN_GAP_COUNT:
        return np.nan
    counts = np.bincount(first_digits(g), minlength=10)[1:10]
    expected = BENFORD * counts.sum()
    return float(((counts - expected) ** 2 / expected).sum() / counts.sum())


def pair_distance(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean absolute residual difference over the tenders two firms both bid in (NaN if few)."""
    n_firms = r.shape[1]
    d = np.full((n_firms, n_firms), np.nan)
    for i in range(n_firms):
        both = mask[:, [i]] & mask
        cnt = both.sum(axis=0)
        diff = np.where(both, np.abs(r[:, [i]] - r), 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            row = diff / cnt
        row[cnt < MIN_CO] = np.nan
        d[i] = row
    np.fill_diagonal(d, np.nan)
    return d


def tightest_group(d: np.ndarray, size: int = GROUP_SIZE) -> float:
    """Smallest mean pairwise distance of a greedily grown group of ``size`` firms."""
    finite = np.where(np.isfinite(d), d, np.inf)
    np.fill_diagonal(finite, np.inf)
    if not np.isfinite(finite).any():
        return np.nan
    i, j = np.unravel_index(int(np.argmin(finite)), finite.shape)
    group = [int(i), int(j)]
    while len(group) < size:
        rest = [k for k in range(d.shape[0]) if k not in group]
        if not rest:
            return np.nan
        means = finite[np.ix_(rest, group)].mean(axis=1)
        best = int(np.argmin(means))
        if not np.isfinite(means[best]):
            return np.nan
        group.append(rest[best])
    sub = finite[np.ix_(group, group)][np.triu_indices(size, k=1)]
    return float(sub.mean())


def shuffle_among_participants(bids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.full_like(bids, np.nan)
    for t in range(bids.shape[0]):
        idx = np.flatnonzero(~np.isnan(bids[t]))
        out[t, idx] = bids[t, rng.permutation(idx)]
    return out


def cd_group(bids: np.ndarray, estimate: np.ndarray, rng: np.random.Generator) -> float:
    mask = ~np.isnan(bids)
    obs = tightest_group(pair_distance(residuals(bids, estimate), mask))
    if not np.isfinite(obs):
        return np.nan
    null = []
    for _ in range(N_PERM):
        sb = shuffle_among_participants(bids, rng)
        v = tightest_group(pair_distance(residuals(sb, estimate), mask))
        if np.isfinite(v):
            null.append(v)
    if len(null) < 5 or np.std(null) < EPS:
        return np.nan
    return float((np.mean(null) - obs) / np.std(null))


def jaccard(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(float)
    inter = m.T @ m
    count = m.sum(axis=0)
    union = count[:, None] + count[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(w, 0.0)
    return w


def group_score(w: np.ndarray) -> float:
    """Best coherence x exclusivity over groups grown greedily from the strongest pairs."""
    n = w.shape[0]
    if n < 3:
        return np.nan
    strength = w.sum(axis=1)
    pairs = np.dstack(np.unravel_index(np.argsort(-w, axis=None), w.shape))[0]
    seen, seeds = set(), []
    for i, j in pairs:
        if i < j and (i, j) not in seen:
            seen.add((i, j))
            seeds.append((int(i), int(j)))
        if len(seeds) == 5:
            break

    def value(group):
        sub = w[np.ix_(group, group)][np.triu_indices(len(group), k=1)]
        if (sub <= 0).any():
            return 0.0
        coherence = float(np.exp(np.log(sub).mean()) / sub.mean())
        inside = 2.0 * sub.sum()
        exclusivity = inside / max(strength[group].sum(), EPS)
        return coherence * exclusivity

    cap = min(MAX_NETWORK_GROUP, max(3, n // 3))  # groups stay small, as in the paper
    best = 0.0
    for seed in seeds:
        group = list(seed)
        cur = value(group)
        while len(group) < cap:
            cand = [(value(group + [k]), k) for k in range(n) if k not in group]
            v, k = max(cand)
            if v <= cur:
                break
            group.append(k)
            cur = v
        best = max(best, cur)
    return best


def wk_group(mask: np.ndarray, rng: np.random.Generator) -> float:
    obs = group_score(jaccard(mask))
    if not np.isfinite(obs):
        return np.nan
    sizes = mask.sum(axis=1)
    null = []
    for _ in range(N_PERM):
        sm = np.zeros_like(mask)
        for t, k in enumerate(sizes):
            sm[t, rng.choice(mask.shape[1], size=int(k), replace=False)] = True
        null.append(group_score(jaccard(sm)))
    null = np.array([v for v in null if np.isfinite(v)])
    if null.size < 5 or null.std() < EPS:
        return np.nan
    return float((obs - null.mean()) / null.std())


def baseline_features(bids: np.ndarray, estimate: np.ndarray, rng: np.random.Generator) -> dict:
    return {
        "bl_benford_chi2": benford_chi2(bids),
        "bl_cd_group": cd_group(bids, estimate, rng),
        "bl_wk_group": wk_group(~np.isnan(bids), rng),
    }
