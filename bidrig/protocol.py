"""Evaluation protocol: leave-one-mechanism-out folds, 5 % alarm threshold, bootstrap CIs.

For every seed, design and fold the model is fitted on the tuning split (without the held-out
mechanism, or with all mechanisms for fold ``all``), the alarm threshold is set at a 5 %
false-alarm rate on the honest ``calib`` split, and the target split (validation or test) is
scored. Metrics, all with two-level bootstrap 95 % CIs (seeds, then markets):

  auc[m]          held-out mechanism m (all conditions) vs all honest target sets, LOMO model
  auc[m@c]        m under condition c vs honest under the same condition (c = in/size/spread)
  prec10[m]       precision among the top 10 % of (m markets + honest markets), LOMO model;
                  base rate = n_m / (n_m + n_honest)
  tpr[m]          alarm rate on m at the calib threshold, LOMO model
  fa[h]           alarm rate on honest set h, fold ``all`` (the deployable detector)
  fa_lomo_max[h]  worst alarm rate on h over the LOMO models
  macro_auc, macro_prec10   means over mechanisms
  control_auc / control_alarm   uniform-markup negative control vs honest, fold ``all``
"""

from __future__ import annotations

import numpy as np

from bidrig.collusion import IN_SCOPE
from bidrig.datasets import load_table
from bidrig.metrics import alarm_rate, auc, bootstrap_many, precision_at_frac, threshold_at_far

FAR = 0.05
FOLDS = ("all",) + IN_SCOPE


def score_split(designs, split: str, seeds, scale: float = 1.0, workers=None):
    """Return (groups, sets): per-seed dicts of score arrays and the target set names."""
    from bidrig.models import make_design

    groups = []
    sets = None
    for seed in seeds:
        tune = load_table(seed, "tuning", scale, workers)
        calib = load_table(seed, "calib", scale, workers)
        target = load_table(seed, split, scale, workers)
        names = sorted(set(target.set_name.tolist()))
        sets = names if sets is None else sets
        g = {}
        for design in designs:
            for fold in FOLDS:
                keep = np.ones(len(tune.y), bool) if fold == "all" else tune.mechanism != fold
                model = make_design(design).fit(tune.X[keep], tune.y[keep])
                thr = threshold_at_far(model.score(calib.X), FAR)
                g[f"thr|{design}|{fold}"] = np.array([thr])
                s = model.score(target.X)
                for name in names:
                    g[f"{name}|{design}|{fold}"] = s[target.set_name == name]
        groups.append(g)
    return groups, sets


def _honest_sets(sets):
    return [s for s in sets if s.startswith("honest@")]


def _mech_sets(sets, m):
    return [s for s in sets if s.split("@")[0] == m]


def metric_functions(design: str, sets) -> dict:
    honest = _honest_sets(sets)
    conds = sorted({s.split("@")[1] for s in sets if s.split("@")[0] in IN_SCOPE})
    fns = {}

    def cat(g, names, fold):
        return np.concatenate([g[f"{n}|{design}|{fold}"] for n in names])

    def per_seed(f):
        return lambda groups: float(np.nanmean([f(g) for g in groups]))

    for m in IN_SCOPE:
        ms = _mech_sets(sets, m)
        fns[f"auc[{m}]"] = per_seed(lambda g, ms=ms, m=m: auc(cat(g, ms, m), cat(g, honest, m)))
        for c in conds:
            if f"honest@{c}" in sets and f"{m}@{c}" in sets:
                fns[f"auc[{m}@{c}]"] = per_seed(
                    lambda g, m=m, c=c: auc(
                        g[f"{m}@{c}|{design}|{m}"], g[f"honest@{c}|{design}|{m}"]
                    )
                )

        def prec(g, ms=ms, m=m):
            pos, neg = cat(g, ms, m), cat(g, honest, m)
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            return precision_at_frac(np.r_[pos, neg], y, 0.10)

        fns[f"prec10[{m}]"] = per_seed(prec)
        fns[f"tpr[{m}]"] = per_seed(
            lambda g, ms=ms, m=m: alarm_rate(cat(g, ms, m), g[f"thr|{design}|{m}"][0])
        )

    fns["macro_auc"] = lambda groups: float(np.mean([fns[f"auc[{m}]"](groups) for m in IN_SCOPE]))
    fns["macro_prec10"] = lambda groups: float(
        np.mean([fns[f"prec10[{m}]"](groups) for m in IN_SCOPE])
    )
    for h in honest:
        fns[f"fa[{h}]"] = per_seed(
            lambda g, h=h: alarm_rate(g[f"{h}|{design}|all"], g[f"thr|{design}|all"][0])
        )
        fns[f"fa_lomo_max[{h}]"] = per_seed(
            lambda g, h=h: max(
                alarm_rate(g[f"{h}|{design}|{m}"], g[f"thr|{design}|{m}"][0]) for m in IN_SCOPE
            )
        )
    if "uniform_markup@in" in sets:
        fns["control_auc"] = per_seed(
            lambda g: auc(g[f"uniform_markup@in|{design}|all"], cat(g, honest, "all"))
        )
        fns["control_alarm"] = per_seed(
            lambda g: alarm_rate(g[f"uniform_markup@in|{design}|all"], g[f"thr|{design}|all"][0])
        )
    return fns


def base_rate(groups, sets, design: str, m: str) -> float:
    g = groups[0]
    n_pos = sum(len(g[f"{s}|{design}|{m}"]) for s in _mech_sets(sets, m))
    n_neg = sum(len(g[f"{s}|{design}|{m}"]) for s in _honest_sets(sets))
    return n_pos / (n_pos + n_neg)


class _Memo:
    """Single-sample cache: statistics sharing sub-results reuse them within one resample."""

    def __init__(self):
        self.sample_id, self.values = None, {}

    def wrap(self, key, f):
        def g(groups):
            if id(groups) != self.sample_id:
                self.sample_id, self.values = id(groups), {}
            if key not in self.values:
                self.values[key] = f(groups)
            return self.values[key]

        return g


def evaluate(
    groups, sets, designs, baselines=(), rng=None, n_boot: int = 1000, compare=None
) -> dict:
    """Metrics with CIs for each design, plus paired differences (``compare`` - baseline)."""
    rng = rng or np.random.default_rng(12345)
    stats = {}
    memo = _Memo()
    per_design = {}
    for d in designs:
        fns = metric_functions(d, sets)
        for k in list(fns):  # in place: macro statistics look entries up in this dict
            fns[k] = memo.wrap((d, k), fns[k])
        per_design[d] = fns
    for d, fns in per_design.items():
        stats.update({f"{d}::{k}": f for k, f in fns.items()})
    for d in compare if compare is not None else designs:
        for b in baselines:
            if b == d:
                continue
            fd, fb = per_design[d], per_design[b]
            stats[f"{d}-{b}::macro_auc"] = lambda groups, fd=fd, fb=fb: fd["macro_auc"](
                groups
            ) - fb["macro_auc"](groups)
            for m in IN_SCOPE:
                stats[f"{d}-{b}::auc[{m}]"] = lambda groups, fd=fd, fb=fb, m=m: fd[f"auc[{m}]"](
                    groups
                ) - fb[f"auc[{m}]"](groups)
    res = bootstrap_many(groups, stats, rng, n_boot)
    out = {k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in res.items()}
    for d in designs:
        for m in IN_SCOPE:
            out[f"{d}::base_rate[{m}]"] = {"point": base_rate(groups, sets, d, m)}
    return out


def criteria(results: dict, design: str, baselines, sets) -> dict:
    """Check the four pass criteria for ``design``. Returns per-criterion pass flags + detail."""
    r = lambda k: results[f"{design}::{k}"]  # noqa: E731
    c1 = {}
    for m in IN_SCOPE:
        keys = [f"auc[{m}]"] + [f"auc[{m}@{c}]" for c in ("in", "size", "spread")]
        c1[m] = {k: r(k)["point"] for k in keys if f"{design}::{k}" in results}
    c1_pass = all(v >= 0.85 for d in c1.values() for v in d.values())
    fa = {
        k.split("::")[1]: v["point"] for k, v in results.items() if k.startswith(f"{design}::fa[")
    }
    c2_pass = all(v <= 0.10 for v in fa.values())
    c3 = {b: results[f"{design}-{b}::macro_auc"] for b in baselines if b != design}
    c3_pass = all(v["lo"] > 0 for v in c3.values())
    c4 = {m: (r(f"prec10[{m}]"), r(f"base_rate[{m}]")["point"]) for m in IN_SCOPE}
    c4_pass = all(p["lo"] > br for p, br in c4.values())
    return {
        "C1_lomo_auc>=0.85": c1_pass,
        "C2_fa<=0.10_every_honest_set": c2_pass,
        "C3_beats_baselines_ci": c3_pass,
        "C4_prec10_above_base_ci": c4_pass,
        "detail": {"C1": c1, "C2": fa, "C3": c3, "C4": {m: [p, b] for m, (p, b) in c4.items()}},
    }
