"""Market-level features computed from public data only (bids, bidder ids, estimates).

Groups:
  lit     standard literature screens per tender (CV, skewness, kurtosis, relative distance,
          RD-alt, normalised gap, spread), aggregated over tenders.
  dep     identity-dependence statistics (pair residual correlation, firm residual dispersion,
          cover-bid tracking, top-two pair recurrence, win/loss inconsistency, repeat wins,
          near-identical bid pairs), each standardised against a within-market null that
          permutes bids among the participants of each tender. Co-bidding (participation
          dependence) uses phi coefficients scaled by sqrt(T).
  change  within-market change scans (Brownian-bridge statistics) on per-tender series.
  seq     sequential layer: CUSUM and a two-state HMM filter on a per-tender suspicion
          series standardised against the market's own history.

Residuals are ``log(bid / estimate)`` minus the tender median, which removes the estimate,
tender-level shocks and any uniform markup, so no feature measures the price level.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata

EPS = 1e-12
MIN_CO = 6  # co-bid tenders needed for a pair correlation
MIN_BIDS = 6  # bids needed for a firm dispersion
MIN_WINS = 4  # wins needed for cover tracking
N_PERM = 30
TIE_TOL = 0.005
CLOSE_TOL = 0.03
# Identity statistics are standardised against two nulls: the tender null shuffles bids among
# the participants of each tender; the time null shuffles each firm's residuals across the
# tenders it bid in (keeps each firm's marginal behaviour, breaks within-tender coordination).
RAW_STATS = ("firm_sd_min", "firm_sd_low")  # invariant under the time null: kept raw

LIT_SCREENS = ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread")
DEP_STATS = (
    "pair_corr_pos",
    "pair_corr_neg",
    "firm_sd_min",
    "firm_sd_low",
    "cover_sd_min",
    "top2_pair",
    "win_loss_corr",
    "repeat_win",
    "near_tie_pair",
    "close_pair",
)
SERIES = ("res_sd", "gap", "nbids", "close")


# ---------------------------------------------------------------- literature screens
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


# ---------------------------------------------------------------- identity statistics
def residuals(bids: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log(bids / estimate[:, None])
    med = np.nanmedian(x, axis=1)
    return x - med[:, None]


def _topk_mean(values: np.ndarray, k: int = 3) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    v = np.sort(v)[::-1]
    return float(v[: min(k, v.size)].mean())


def pair_correlation_z(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fisher-z * sqrt(n-3) residual correlation for each pair; NaN where n < MIN_CO."""
    m = mask.astype(float)
    r0 = np.where(mask, r, 0.0)
    n = m.T @ m
    a = r0.T @ m  # a[i, j] = sum of r_i over tenders where j bids too (and i bids)
    q = (r0**2).T @ m
    sij = r0.T @ r0
    num = n * sij - a * a.T
    den = np.sqrt(np.maximum(n * q - a**2, 0.0) * np.maximum(n * q.T - a.T**2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.clip(num / den, -0.999999, 0.999999)
        z = np.arctanh(corr) * np.sqrt(np.maximum(n - 3, 0.0))
    iu = np.triu(np.ones_like(n, dtype=bool), k=1)
    z[(n < MIN_CO) | ~iu | ~np.isfinite(z)] = np.nan
    return z


def identity_stats(bids: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Statistics that depend on which firm submitted which bid."""
    mask = ~np.isnan(bids)
    t_count, n_firms = bids.shape
    out = {}

    z = pair_correlation_z(r, mask)
    out["pair_corr_pos"] = _topk_mean(z.ravel())
    out["pair_corr_neg"] = _topk_mean(-z.ravel())

    nb = mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(np.where(mask, r, np.nan), axis=0)
    sd = sd[nb >= MIN_BIDS]
    if sd.size >= 3 and np.median(sd) > EPS:
        ratio = sd / np.median(sd)
        out["firm_sd_min"] = float(-np.log(ratio.min() + 1e-6))
        out["firm_sd_low"] = float((ratio < 0.5).mean())
    else:
        out["firm_sd_min"] = out["firm_sd_low"] = np.nan

    win = np.nanargmin(bids, axis=1)
    win_onehot = np.zeros_like(mask, dtype=float)
    win_onehot[np.arange(t_count), win] = 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(bids / bids[np.arange(t_count), win][:, None])
    lose_mask = mask & (win_onehot == 0)
    l0 = np.where(lose_mask, lr, 0.0)
    cnt = win_onehot.T @ lose_mask.astype(float)
    s1 = win_onehot.T @ l0
    s2 = win_onehot.T @ (l0**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        csd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))
    csd = csd[cnt >= MIN_WINS]
    if csd.size >= 2 and np.median(csd) > EPS:
        out["cover_sd_min"] = float(-np.log(csd.min() / np.median(csd) + 1e-6))
    else:
        out["cover_sd_min"] = np.nan

    order = np.argsort(np.where(mask, bids, np.inf), axis=1)
    two = mask.sum(axis=1) >= 2
    a, b = np.minimum(order[two, 0], order[two, 1]), np.maximum(order[two, 0], order[two, 1])
    if a.size:
        counts = np.bincount(a * n_firms + b, minlength=n_firms * n_firms)
        out["top2_pair"] = float(counts.max() / a.size)
    else:
        out["top2_pair"] = np.nan

    wins = win_onehot.sum(axis=0)
    rate = wins / np.maximum(nb, 1)
    with np.errstate(invalid="ignore"):
        loss_res = np.nanmean(np.where(lose_mask, r, np.nan), axis=0)
    keep = (nb >= MIN_BIDS) & np.isfinite(loss_res)
    if keep.sum() >= 4 and np.ptp(rate[keep]) > 0 and np.ptp(loss_res[keep]) > 0:
        out["win_loss_corr"] = float(_spearman(rate[keep], loss_res[keep]))
    else:
        out["win_loss_corr"] = np.nan

    out["repeat_win"] = float(-(win[1:] == win[:-1]).mean())

    if n_firms <= 60:
        with np.errstate(invalid="ignore", divide="ignore"):
            lb = np.log(bids)
            diff = np.abs(lb[:, :, None] - lb[:, None, :])
        both = mask[:, :, None] & mask[:, None, :]
        ties = (diff < TIE_TOL) & both
        co = both.sum(axis=0)
        frac = np.where(co >= MIN_CO, ties.sum(axis=0) / np.maximum(co, 1), np.nan)
        frac[np.tril(np.ones_like(co, dtype=bool))] = np.nan
        out["near_tie_pair"] = float(np.nanmax(frac)) if np.isfinite(frac).any() else np.nan
        close = (diff < CLOSE_TOL) & both
        cfrac = np.where(co >= MIN_CO, close.sum(axis=0) / np.maximum(co, 1), np.nan)
        cfrac[np.tril(np.ones_like(co, dtype=bool))] = np.nan
        out["close_pair"] = _topk_mean(cfrac.ravel())
    else:
        out["near_tie_pair"] = out["close_pair"] = np.nan
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def permute_within_tenders(
    bids: np.ndarray, r: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle which participant submitted which bid in each tender; participation is kept."""
    mask = ~np.isnan(bids)
    big = np.iinfo(np.int64).max
    pos = np.argsort(np.where(mask, np.arange(bids.shape[1])[None, :], big), axis=1, kind="stable")
    keys = np.where(mask, rng.random(bids.shape), np.inf)
    src = np.argsort(keys, axis=1, kind="stable")
    bp = np.empty_like(bids)
    rp = np.empty_like(r)
    np.put_along_axis(bp, pos, np.take_along_axis(bids, src, axis=1), axis=1)
    np.put_along_axis(rp, pos, np.take_along_axis(r, src, axis=1), axis=1)
    return bp, rp


def permute_within_firms(r: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each firm's residuals across the tenders it bid in; participation is kept."""
    mask = ~np.isnan(r)
    big = np.iinfo(np.int64).max
    pos = np.argsort(np.where(mask, np.arange(r.shape[0])[:, None], big), axis=0, kind="stable")
    keys = np.where(mask, rng.random(r.shape), np.inf)
    src = np.argsort(keys, axis=0, kind="stable")
    rp = np.empty_like(r)
    np.put_along_axis(rp, pos, np.take_along_axis(r, src, axis=0), axis=0)
    return rp


def cobid_features(mask: np.ndarray) -> dict[str, float]:
    m = mask.astype(float)
    t_count = m.shape[0]
    p = m.mean(axis=0)
    sd = np.sqrt(p * (1 - p))
    ok = sd > 0
    if ok.sum() < 2:
        return {"cobid_min": np.nan, "cobid_bottom3": np.nan}
    c = ((m - p).T @ (m - p)) / t_count
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = c / np.outer(sd, sd)
    phi[~np.outer(ok, ok) | np.tril(np.ones_like(phi, dtype=bool))] = np.nan
    zz = phi * np.sqrt(t_count)
    return {"cobid_min": float(-np.nanmin(zz)), "cobid_bottom3": _topk_mean(-zz.ravel())}


def _zscore(obs: float, null: list[float]) -> float:
    nv = np.array(null, dtype=float)
    nv = nv[np.isfinite(nv)]
    if not np.isfinite(obs) or nv.size < 5:
        return np.nan
    return float((obs - nv.mean()) / (nv.std() + 1e-3))


def dependence_features(
    bids: np.ndarray, estimate: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM
) -> dict[str, float]:
    """Identity statistics as z-scores against both nulls.

    ``dep_<stat>`` uses the tender null, ``dep_<stat>_t`` the time null. Statistics that the
    time null leaves unchanged (``RAW_STATS``) are reported raw instead of ``_t``.
    """
    r = residuals(bids, estimate)
    obs = identity_stats(bids, r)
    null_tender = {k: [] for k in obs}
    null_time = {k: [] for k in obs}
    for _ in range(n_perm):
        bp, rp = permute_within_tenders(bids, r, rng)
        for k, v in identity_stats(bp, rp).items():
            null_tender[k].append(v)
        rp = permute_within_firms(r, rng)
        for k, v in identity_stats(np.exp(rp), rp).items():
            null_time[k].append(v)
    feats = {}
    for k in DEP_STATS:
        feats[f"dep_{k}"] = _zscore(obs[k], null_tender[k])
        feats[f"dep_{k}_t"] = obs[k] if k in RAW_STATS else _zscore(obs[k], null_time[k])
    feats.update({f"dep_{k}": v for k, v in cobid_features(~np.isnan(bids)).items()})
    return feats


# ---------------------------------------------------------------- participation
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


# ---------------------------------------------------------------- per-tender series
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


def _robust_z(x: np.ndarray) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size < 5:
        return np.full_like(x, np.nan)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    scale = mad if mad > EPS else (v.std() + EPS)
    return (x - med) / scale


def bridge_scan(x: np.ndarray) -> float:
    """max |partial sum of standardised deviations| / sqrt(n): a change-in-mean scan."""
    v = x[np.isfinite(x)]
    if v.size < 8 or v.std() < EPS:
        return np.nan
    z = (v - v.mean()) / v.std()
    return float(np.abs(np.cumsum(z)).max() / np.sqrt(v.size))


def change_features(bids: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return {f"chg_{k}": bridge_scan(v) for k, v in tender_series(bids, estimate).items()}


def suspicion_series(bids: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Per-tender suspicion: low dispersion, small gap, many bids close to the winner."""
    s = tender_series(bids, estimate)
    parts = [-_robust_z(s["res_sd"]), -_robust_z(s["gap"]), _robust_z(s["close"])]
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


# ---------------------------------------------------------------- all features
def market_features(bids: np.ndarray, estimate: np.ndarray, rng: np.random.Generator) -> dict:
    feats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats.update(literature_features(bids))
        feats.update(dependence_features(bids, estimate, rng))
        feats.update(participation_features(~np.isnan(bids)))
        feats.update(change_features(bids, estimate))
        feats.update(sequential_features(bids, estimate))
    return feats


FEATURE_GROUPS = {
    "lit": tuple(f"lit_{s}_{a}" for s in LIT_SCREENS for a in ("mean", "median", "sd")),
    "dep": tuple(f"dep_{k}{sfx}" for k in DEP_STATS for sfx in ("", "_t"))
    + ("dep_cobid_min", "dep_cobid_bottom3"),
    "part": ("part_rate", "part_low", "part_group"),
    "change": tuple(f"chg_{k}" for k in SERIES),
    "seq": ("seq_cusum", "seq_hmm_max", "seq_hmm_mean"),
}
ALL_FEATURES = tuple(f for g in FEATURE_GROUPS.values() for f in g)
