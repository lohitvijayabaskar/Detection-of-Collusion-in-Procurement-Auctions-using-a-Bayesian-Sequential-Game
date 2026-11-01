"""Tender-level leave-one-dataset-out (LODO) evaluation on the real bid data.

Design (proposed in docs/real_data_design.md; frozen by ``design_hash``):
  unit      one tender; label = ``tender_label`` (paper's thresholded tender label)
  screens   literature screens of the tender's own bids (``features.tender_screens``) + log
            number of bids
  pairs     for every pair of the tender's bidders, statistics computed from all OTHER
            tenders of the same dataset (exact leave-tender-out): residual correlation
            (Fisher z), share of close co-bids, co-bidding phi; aggregated per tender
            (max, top-3 mean) plus coverage (share of pairs with >= MIN_CO_PAIR co-bids)
  scaling   every feature is replaced by its within-dataset percentile (label-free), so
            currency, project size and market density do not differ across datasets
  models    GBM (``models.GBM_PARAMS``) on screens (baseline), on pairs, on both; seven
            single screens with sign chosen on the training datasets
  folds     train on two datasets, test on the third
  threshold 5 % false alarms on out-of-fold scores of the training datasets' competitive
            tenders (5-fold, fixed split)
  CIs       moving-block bootstrap over tenders in time order (block 20, 1000 resamples),
            paired across designs

Residuals are ``log(bid / estimate)`` minus the tender median, as in the simulator features.
No feature uses labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from bidrig.features import tender_screens
from bidrig.metrics import alarm_rate, auc, precision_at_frac, threshold_at_far

MIN_CO_PAIR = 4
CLOSE_TOL = 0.03
SCREEN_NAMES = ("cv", "skew", "kurt", "rd", "rdalt", "gap", "spread")
SCREEN_FEATURES = tuple(f"scr_{s}" for s in SCREEN_NAMES) + ("scr_log_nbids",)
PAIR_FEATURES = (
    "pair_corr_max",
    "pair_corr_top3",
    "pair_close_max",
    "pair_close_top3",
    "pair_cobid_max",
    "pair_cobid_top3",
    "pair_coverage",
)
DESIGNS = {
    "real_lit_gbm": SCREEN_FEATURES,
    "real_pair_gbm": PAIR_FEATURES + ("scr_log_nbids",),
    "real_full_gbm": SCREEN_FEATURES + PAIR_FEATURES,
}
SINGLE = tuple(f"scr_{s}" for s in SCREEN_NAMES)
FAR = 0.05
BLOCK = 20


def tender_matrix(df):
    """(T, N) bids in time order, estimates, tender labels, tender ids, firm ids."""
    orders = np.sort(df["order"].unique())
    firms = np.sort(df["firm"].unique())
    row = {o: i for i, o in enumerate(orders)}
    col = {f: j for j, f in enumerate(firms)}
    bids = np.full((len(orders), len(firms)), np.nan)
    bids[df["order"].map(row).to_numpy(), df["firm"].map(col).to_numpy()] = df["bid"].to_numpy()
    g = df.groupby("order")
    est = g.estimate.first().reindex(orders).to_numpy()
    lab = g.tender_label.first().reindex(orders).to_numpy().astype(int)
    tid = g.tender.first().reindex(orders).to_numpy()
    return bids, est, lab, tid, firms


def _topk(v: np.ndarray, k: int = 3) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.sort(v)[::-1][: min(k, v.size)].mean())


def pair_features(bids: np.ndarray, estimate: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tender aggregates of leave-tender-out pair statistics among the tender's bidders."""
    mask = ~np.isnan(bids)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log(bids / estimate[:, None])
    r = x - np.nanmedian(x, axis=1)[:, None]
    m = mask.astype(float)
    r0 = np.where(mask, r, 0.0)
    n_all = m.T @ m
    a_all = r0.T @ m  # sum of r_i over tenders where i and j both bid
    q_all = (r0**2).T @ m
    s_all = r0.T @ r0
    t_count = bids.shape[0]
    close_all = np.zeros_like(n_all)
    for t in range(t_count):
        idx = np.flatnonzero(mask[t])
        c = (np.abs(r[t, idx][:, None] - r[t, idx][None, :]) < CLOSE_TOL).astype(float)
        close_all[np.ix_(idx, idx)] += c
    cnt = m.sum(axis=0)

    out = {k: np.full(t_count, np.nan) for k in PAIR_FEATURES}
    for t in range(t_count):
        idx = np.flatnonzero(mask[t])
        if idx.size < 2:
            continue
        rt = r[t, idx]
        iu = np.triu_indices(idx.size, k=1)
        sub = np.ix_(idx, idx)
        n = (n_all[sub] - 1.0)[iu]
        a = (a_all[sub] - rt[:, None])[iu]  # sum r_i over other co-bid tenders
        b = (a_all[sub].T - rt[None, :])[iu]  # sum r_j
        qa = (q_all[sub] - rt[:, None] ** 2)[iu]
        qb = (q_all[sub].T - rt[None, :] ** 2)[iu]
        sab = (s_all[sub] - np.outer(rt, rt))[iu]
        close = (close_all[sub] - (np.abs(rt[:, None] - rt[None, :]) < CLOSE_TOL))[iu]
        ok = n >= MIN_CO_PAIR
        with np.errstate(invalid="ignore", divide="ignore"):
            num = n * sab - a * b
            den = np.sqrt(np.maximum(n * qa - a**2, 0) * np.maximum(n * qb - b**2, 0))
            corr = np.clip(num / den, -0.999999, 0.999999)
            zc = np.arctanh(corr) * np.sqrt(np.maximum(n - 3, 0))
            share = close / n
        zc[~ok | ~np.isfinite(zc)] = np.nan
        share[~ok] = np.nan
        # co-bidding phi over the other T-1 tenders
        tt = t_count - 1.0
        ci = (cnt[idx] - 1.0)[:, None] * np.ones(idx.size)[None, :]
        cj = ci.T
        pi, pj = ci[iu] / tt, cj[iu] / tt
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = (n / tt - pi * pj) / np.sqrt(pi * (1 - pi) * pj * (1 - pj))
        zphi = phi * np.sqrt(tt)
        zphi[~np.isfinite(zphi)] = np.nan
        out["pair_corr_max"][t] = np.nanmax(zc) if np.isfinite(zc).any() else np.nan
        out["pair_corr_top3"][t] = _topk(zc)
        out["pair_close_max"][t] = np.nanmax(share) if np.isfinite(share).any() else np.nan
        out["pair_close_top3"][t] = _topk(share)
        out["pair_cobid_max"][t] = np.nanmax(zphi) if np.isfinite(zphi).any() else np.nan
        out["pair_cobid_top3"][t] = _topk(zphi)
        out["pair_coverage"][t] = float(ok.mean())
    return out


def screen_features(bids: np.ndarray) -> dict[str, np.ndarray]:
    s = tender_screens(bids)
    out = {f"scr_{k}": s[k] for k in SCREEN_NAMES}
    out["scr_log_nbids"] = np.log((~np.isnan(bids)).sum(axis=1).astype(float))
    return out


def percentile(v: np.ndarray) -> np.ndarray:
    """Within-dataset percentile in (0, 1]; ties get the average rank; NaN stays NaN."""
    from scipy.stats import rankdata

    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = rankdata(v[ok]) / ok.sum()
    return out


def dataset_table(df) -> dict:
    """Feature matrix (percentile-scaled), labels and order for one dataset."""
    import warnings

    bids, est, lab, tid, _ = tender_matrix(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = {**screen_features(bids), **pair_features(bids, est)}
    names = SCREEN_FEATURES + PAIR_FEATURES
    X = np.column_stack([percentile(feats[k]) for k in names])
    return {"X": X, "y": lab, "tender": tid, "names": names}


def _cols(names, wanted):
    index = {n: i for i, n in enumerate(names)}
    return [index[w] for w in wanted]


def fit_score(design: str, train: dict, test: dict):
    """Fit on the training table, return (test scores, threshold from OOF honest scores)."""
    from bidrig.detectors.config import GBM_PARAMS

    X, y = train["X"], train["y"]
    if design.startswith("single:"):
        j = _cols(train["names"], [design.split(":", 1)[1]])[0]
        sign = 1.0 if auc(X[y == 1, j], X[y == 0, j]) >= 0.5 else -1.0
        thr = threshold_at_far(sign * X[y == 0, j], FAR)
        return sign * test["X"][:, j], thr
    cols = _cols(train["names"], DESIGNS[design])
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mdl = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X[tr][:, cols], y[tr])
        oof[te] = mdl.predict_proba(X[te][:, cols])[:, 1]
    thr = threshold_at_far(oof[y == 0], FAR)
    model = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X[:, cols], y)
    return model.predict_proba(test["X"][:, cols])[:, 1], thr


def concat(tables):
    return {
        "X": np.vstack([t["X"] for t in tables]),
        "y": np.concatenate([t["y"] for t in tables]),
        "names": tables[0]["names"],
    }


def block_indices(n: int, rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    """Moving-block bootstrap indices of length n over a time-ordered sequence."""
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def fold_metrics(scores: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pos, neg = scores[y == 1], scores[y == 0]
    return {
        "auc": auc(pos, neg),
        "prec10": precision_at_frac(scores, y, 0.10),
        "tpr": alarm_rate(pos, thr),
        "fa": alarm_rate(neg, thr),
        "base_rate": float(y.mean()),
    }


def bootstrap_fold(scores: dict, y: np.ndarray, thr: dict, rng, n_boot: int = 1000, pairs=()):
    """Point estimates and block-bootstrap CIs for each design, plus paired AUC differences."""
    point = {d: fold_metrics(s, y, thr[d]) for d, s in scores.items()}
    samples = {d: {k: [] for k in point[d]} for d in scores}
    diffs = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(n_boot):
        idx = block_indices(len(y), rng)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        cur = {d: fold_metrics(s[idx], yb, thr[d]) for d, s in scores.items()}
        for d in scores:
            for k, v in cur[d].items():
                samples[d][k].append(v)
        for a, b in pairs:
            diffs[f"{a}-{b}"].append(cur[a]["auc"] - cur[b]["auc"])
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]  # noqa: E731
    out = {
        d: {k: {"point": point[d][k], "ci": ci(samples[d][k])} for k in point[d]} for d in scores
    }
    for a, b in pairs:
        key = f"{a}-{b}"
        out[key] = {"point": point[a]["auc"] - point[b]["auc"], "ci": ci(diffs[key])}
    return out


def lodo(tables: dict, designs, n_boot: int = 1000, seed: int = 2026, compare=("real_full_gbm",)):
    """Leave-one-dataset-out over ``tables`` (name -> dataset_table)."""
    rng = np.random.default_rng(seed)
    results = {}
    for held in sorted(tables):
        train = concat([tables[k] for k in sorted(tables) if k != held])
        test = tables[held]
        scores, thr = {}, {}
        for d in designs:
            scores[d], thr[d] = fit_score(d, train, test)
        pairs = [(a, b) for a in compare for b in designs if b != a]
        results[held] = bootstrap_fold(scores, test["y"], thr, rng, n_boot, pairs)
    return results
