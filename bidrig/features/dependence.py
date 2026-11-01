"""Group ``dep``: identity-dependence statistics (pair residual correlation, firm residual
dispersion, cover-bid tracking, top-two pair recurrence, win/loss inconsistency, repeat wins,
near-identical and close bid pairs, co-bidding), each standardised against two within-market
nulls. The tender null shuffles bids among the participants of each tender; the time null
shuffles each firm's residuals across the tenders it bid in (keeps each firm's marginal
behaviour, breaks within-tender coordination).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from bidrig.features.common import (
    CLOSE_TOL,
    EPS,
    MIN_BIDS,
    MIN_CO,
    MIN_WINS,
    N_PERM,
    TIE_TOL,
    residuals,
    topk_mean,
)

RAW_STATS = ("firm_sd_min", "firm_sd_low")  # invariant under the time null: kept raw
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
DEP_FEATURES = tuple(f"dep_{k}{sfx}" for k in DEP_STATS for sfx in ("", "_t")) + (
    "dep_cobid_min",
    "dep_cobid_bottom3",
)


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
    out["pair_corr_pos"] = topk_mean(z.ravel())
    out["pair_corr_neg"] = topk_mean(-z.ravel())

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
        out["close_pair"] = topk_mean(cfrac.ravel())
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
    return {"cobid_min": float(-np.nanmin(zz)), "cobid_bottom3": topk_mean(-zz.ravel())}


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
