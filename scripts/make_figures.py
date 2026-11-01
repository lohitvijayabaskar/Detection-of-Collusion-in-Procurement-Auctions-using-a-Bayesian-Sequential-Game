"""Regenerate all paper figures from logged results (no experiment is re-run).

python scripts/make_figures.py

Inputs: results/locked_1_full_gbm.json, results/validation_v{1,2,3,3mono,4}.json,
results/real_lodo_1.json, and (read-only) the old repo's results/new_detector_on_old_env and
results/new_detector_recalibrated. Outputs: docs/figures/*.png, plus one CSV per figure in
docs/figures/data/ (the table view of each chart).

Palette: the dataviz reference categorical slots 1-3 (validated all-pairs, light mode) plus a
neutral gray for de-emphasised marks. Slot 3 (aqua) is below 3:1 contrast on white, so every
series is also told apart by marker shape and a legend or direct label.
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _common import RESULTS, ROOT  # noqa: E402

from bidrig.collusion import IN_SCOPE  # noqa: E402

FIG = ROOT / "docs" / "figures"
DATA = FIG / "data"
OLD = ROOT / "results" / "old_env"  # copies of the old-environment transfer results

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "savefig.dpi": 200,
    }
)
NICE = {
    "bid_suppression": "bid suppression",
    "complementary": "complementary",
    "cover_bidding": "cover bidding",
    "intermittent": "intermittent",
    "noisy_defecting": "noisy / defecting",
    "partial_fringe": "partial cartel + fringe",
    "price_fixing": "price fixing",
    "rotation": "rotation",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save(fig, name, title, subtitle):
    h = fig.get_figheight()
    fig.text(
        0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=11, color=INK, fontweight="bold"
    )
    fig.text(0.01, 1 - 0.36 / h, subtitle, ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.58 / h))
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def err(p):
    return [[p["point"] - p["lo"]], [p["hi"] - p["point"]]]


# ------------------------------------------------------------------ locked test
def fig_locked_mechanisms(res):
    conds = [
        ("in", "in-distribution", BLUE, "o"),
        ("size", "unseen sizes", ORANGE, "s"),
        ("spread", "unseen spreads", AQUA, "^"),
    ]
    order = sorted(IN_SCOPE, key=lambda m: res[f"full_gbm::auc[{m}]"]["point"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = []
    for k, (c, label, col, mk) in enumerate(conds):
        for i, m in enumerate(order):
            p = res[f"full_gbm::auc[{m}@{c}]"]
            y = i + (k - 1) * 0.22
            ax.errorbar(
                p["point"],
                y,
                xerr=err(p),
                fmt=mk,
                color=col,
                ms=5,
                lw=1.2,
                capsize=0,
                label=label if i == 0 else None,
            )
            rows.append([m, c, p["point"], p["lo"], p["hi"]])
    ax.axvline(0.85, color=INK2, lw=1, ls="--")
    ax.text(0.852, len(order) - 0.45, "C1 target 0.85", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)), [NICE[m] for m in order])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("leave-one-mechanism-out AUC vs honest markets (95 % CI)")
    ax.legend(loc="lower right", fontsize=8)
    save(
        fig,
        "fig1_locked_auc_by_mechanism",
        "Four of eight held-out mechanisms miss the 0.85 target",
        "Locked test, full_gbm, 10 seeds. Each mechanism is scored by a model that never saw it.",
    )
    write_csv("fig1_locked_auc_by_mechanism", ["mechanism", "condition", "auc", "lo", "hi"], rows)


def fig_locked_baselines(res):
    designs = ["full_gbm", "lit_gbm"] + [
        f"single:lit_{s}_mean" for s in ("rdalt", "rd", "cv", "spread", "kurt", "skew", "gap")
    ]
    names = {"full_gbm": "full_gbm (proposed)", "lit_gbm": "screens GBM"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rows = []
    for i, d in enumerate(designs[::-1]):
        p = res[f"{d}::macro_auc"]
        col = BLUE if d == "full_gbm" else GRAY
        ax.barh(i, p["point"] - 0.4, left=0.4, height=0.55, color=col)
        ax.errorbar(p["point"], i, xerr=err(p), fmt="none", color=INK, lw=1)
        ax.text(p["hi"] + 0.006, i, f"{p['point']:.3f}", va="center", fontsize=8, color=INK)
        rows.append([d, p["point"], p["lo"], p["hi"]])
    labels = [
        names.get(d, d.replace("single:lit_", "").replace("_mean", "") + " (single screen)")
        for d in designs[::-1]
    ]
    ax.set_yticks(range(len(designs)), labels)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("macro LOMO AUC (95 % CI)")
    d = res["full_gbm-lit_gbm::macro_auc"]
    save(
        fig,
        "fig2_locked_vs_baselines",
        f"The full detector beats every literature screen (+{d['point']:.3f} over the screens GBM)",
        f"Locked test. Paired difference vs screens GBM {d['point']:.3f} "
        f"[{d['lo']:.3f}, {d['hi']:.3f}]; all single-screen differences have CIs above 0.",
    )
    write_csv("fig2_locked_vs_baselines", ["design", "macro_auc", "lo", "hi"], rows)


def fig_locked_false_alarms(res):
    sets = ["in", "size", "spread", "markup_high", "capacity", "noisy"]
    nice = {
        "in": "in-distribution",
        "size": "unseen sizes",
        "spread": "unseen spreads",
        "markup_high": "markup 20-25 %",
        "capacity": "strong capacity",
        "noisy": "noisy estimates",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = []
    for i, s in enumerate(sets):
        a = res[f"full_gbm::fa[honest@{s}]"]
        b = res[f"full_gbm::fa_lomo_max[honest@{s}]"]
        ax.errorbar(
            i - 0.12,
            a["point"],
            yerr=[[a["point"] - a["lo"]], [a["hi"] - a["point"]]],
            fmt="o",
            color=BLUE,
            ms=6,
            lw=1.2,
            label="deployable model (all mechanisms)" if i == 0 else None,
        )
        ax.errorbar(
            i + 0.12,
            b["point"],
            yerr=[[b["point"] - b["lo"]], [b["hi"] - b["point"]]],
            fmt="s",
            color=ORANGE,
            ms=5,
            lw=1.2,
            label="worst leave-one-out model" if i == 0 else None,
        )
        rows.append([s, a["point"], a["lo"], a["hi"], b["point"], b["lo"], b["hi"]])
    for y, lab in ((0.05, "5 % target"), (0.10, "C2 limit 10 %")):
        ax.axhline(y, color=INK2, lw=1, ls="--")
        ax.text(5.45, y + 0.003, lab, color=INK2, fontsize=8, ha="right")
    ax.set_xticks(range(len(sets)), [nice[s] for s in sets])
    ax.set_ylabel("false-alarm rate (95 % CI)")
    ax.set_ylim(0, 0.17)
    ax.legend(loc="upper left", fontsize=8)
    save(
        fig,
        "fig3_locked_false_alarms",
        "False alarms stay under 10 % on every honest test set",
        "Locked test, threshold at 5 % on calib. Leave-one-out models exceed 10 % on unseen sizes/spreads.",
    )
    write_csv(
        "fig3_locked_false_alarms",
        ["honest_set", "fa_all", "lo", "hi", "fa_lomo_max", "lo", "hi"],
        rows,
    )


def fig_locked_precision(res):
    order = sorted(IN_SCOPE, key=lambda m: res[f"full_gbm::prec10[{m}]"]["point"])
    base = res[f"full_gbm::base_rate[{order[0]}]"]["point"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rows = []
    for i, m in enumerate(order):
        p = res[f"full_gbm::prec10[{m}]"]
        ax.barh(i, p["point"], height=0.55, color=BLUE)
        ax.errorbar(p["point"], i, xerr=err(p), fmt="none", color=INK, lw=1)
        rows.append([m, p["point"], p["lo"], p["hi"], base])
    ax.axvline(base, color=INK2, lw=1, ls="--")
    ax.text(base + 0.01, len(order) - 0.5, f"base rate {base:.3f}", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)), [NICE[m] for m in order])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("precision among the top 10 % of ranked markets (95 % CI)")
    save(
        fig,
        "fig4_locked_precision10",
        "Ranking beats the base rate for every held-out mechanism",
        "Locked test, full_gbm. Pool = held-out mechanism markets + all honest test markets.",
    )
    write_csv("fig4_locked_precision10", ["mechanism", "prec10", "lo", "hi", "base_rate"], rows)


# ------------------------------------------------------------------ validation history
def fig_validation_trace():
    steps = [
        ("V1", "validation_v1", "full_gbm", "baseline"),
        ("I1 time null", "validation_v2", "full_gbm", "kept"),
        ("I2 tail fusion", "validation_v3", "full_fused", "no gain"),
        ("I4 monotone", "validation_v3mono", "lit_dep_part_mono", "no gain"),
        ("W2-3 epidemic", "validation_v4", "full_gbm", "reverted"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows, xs, macro, mins = [], [], [], []
    for i, (lab, f, d, status) in enumerate(steps):
        r = load(RESULTS / f"{f}.json")["results"]
        p = r[f"{d}::macro_auc"]
        mn = min(r[f"{d}::auc[{m}]"]["point"] for m in IN_SCOPE)
        ax.errorbar(
            i,
            p["point"],
            yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
            fmt="o",
            color=BLUE,
            ms=6,
            lw=1.2,
        )
        ax.text(i, p["hi"] + 0.012, status, ha="center", fontsize=8, color=INK2)
        xs.append(i), macro.append(p["point"]), mins.append(mn)
        rows.append([lab, d, p["point"], p["lo"], p["hi"], mn, status])
    ax.plot(xs, macro, color=BLUE, lw=2, label="macro LOMO AUC (95 % CI)")
    ax.plot(xs, mins, color=ORANGE, lw=2, marker="s", ms=5, label="worst mechanism AUC")
    ax.axhline(0.85, color=INK2, lw=1, ls="--")
    ax.set_xticks(xs, [s[0] for s in steps])
    ax.set_ylim(0.6, 0.92)
    ax.set_ylabel("validation AUC")
    ax.legend(loc="center right", fontsize=8)
    save(
        fig,
        "fig5_validation_history",
        "Only the time-shuffle null improved validation; later iterations stalled",
        "Validation split, 10 seeds. Stop rule reached in both workstreams (3 iterations without gain).",
    )
    write_csv(
        "fig5_validation_history",
        ["step", "design", "macro_auc", "lo", "hi", "min_mechanism_auc", "decision"],
        rows,
    )


def fig_ablation():
    r = load(RESULTS / "validation_v2.json")["results"]
    designs = [
        ("full_gbm", "all groups"),
        ("full_minus_change", "minus change scans"),
        ("full_minus_seq", "minus sequential layer"),
        ("full_minus_dep", "minus dependence"),
        ("dep_gbm", "dependence only"),
        ("lit_gbm", "screens only"),
        ("seq_gbm", "sequential only"),
        ("change_gbm", "change scans only"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = []
    for i, (d, lab) in enumerate(designs[::-1]):
        p = r[f"{d}::macro_auc"]
        col = BLUE if d in ("full_gbm", "dep_gbm") else GRAY
        ax.barh(i, p["point"] - 0.4, left=0.4, height=0.55, color=col)
        ax.errorbar(p["point"], i, xerr=err(p), fmt="none", color=INK, lw=1)
        ax.text(p["hi"] + 0.006, i, f"{p['point']:.3f}", va="center", fontsize=8, color=INK)
        rows.append([d, p["point"], p["lo"], p["hi"]])
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_yticks(range(len(designs)), [lab for _, lab in designs[::-1]])
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("macro LOMO AUC on validation (95 % CI)")
    save(
        fig,
        "fig6_ablation",
        "Dependence features carry the gain; change and sequential layers add nothing",
        "Validation split (feature v2), 10 seeds. Dotted line = chance.",
    )
    write_csv("fig6_ablation", ["design", "macro_auc", "lo", "hi"], rows)


# ------------------------------------------------------------------ real data
def fig_real():
    r = load(RESULTS / "real_lodo_1.json")
    res = r["results"]
    countries = ["italy", "japan", "brazil"]
    series = [
        ("real_full_gbm", "screens + pair features", BLUE, "o"),
        ("real_lit_gbm", "screens GBM", ORANGE, "s"),
        ("single:scr_cv", "CV screen alone", AQUA, "^"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    rows = []
    for k, (d, lab, col, mk) in enumerate(series):
        for i, c in enumerate(countries):
            a = res[c][d]["auc"]
            y = i + (k - 1) * 0.22
            ax.errorbar(
                a["point"],
                y,
                xerr=[[a["point"] - a["ci"][0]], [a["ci"][1] - a["point"]]],
                fmt=mk,
                color=col,
                ms=5,
                lw=1.2,
                label=lab if i == 0 else None,
            )
            fa = res[c][d]["fa"]
            rows.append([c, d, a["point"], *a["ci"], fa["point"], *fa["ci"]])
    for k, (d, lab, col, mk) in enumerate(series[:2]):
        for i, c in enumerate(countries):
            fa = res[c][d]["fa"]
            y = i + (k - 0.5) * 0.25
            ax2.errorbar(
                fa["point"],
                y,
                xerr=[[fa["point"] - fa["ci"][0]], [fa["ci"][1] - fa["point"]]],
                fmt=mk,
                color=col,
                ms=5,
                lw=1.2,
            )
    labels = [f"{c.title()} ({r['tenders'][c]}, {r['positives'][c]} coll.)" for c in countries]
    ax.set_yticks(range(3), labels)
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("held-out AUC (95 % block-bootstrap CI)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.set_yticks(range(3), ["", "", ""])
    ax2.axvline(0.05, color=INK2, lw=1, ls="--")
    ax2.axvline(0.10, color=INK2, lw=1, ls="--")
    ax2.set_xlim(0, 0.45)
    ax2.set_ylim(ax.get_ylim())
    top = ax.get_ylim()[1] - 0.12
    ax2.text(0.047, top, "5 %", color=INK2, fontsize=7.5, ha="right", va="top")
    ax2.text(0.104, top, "10 %", color=INK2, fontsize=7.5, ha="left", va="top")
    ax2.set_xlabel("false-alarm rate (same colours)")
    d = res["italy"]["real_full_gbm-real_lit_gbm"]
    save(
        fig,
        "fig7_real_data_lodo",
        "Real data: pair features help only in Italy; the threshold does not transfer to Japan",
        f"Tender-level, leave-one-country-out, run once. Italy gain over screens GBM "
        f"+{d['point']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}].",
    )
    write_csv(
        "fig7_real_data_lodo",
        ["country", "design", "auc", "lo", "hi", "fa", "fa_lo", "fa_hi"],
        rows,
    )


# ------------------------------------------------------------------ old environment
def fig_old_env():
    e1 = load(OLD / "new_detector_on_old_env" / "results.json")["rows"]
    e2 = load(OLD / "new_detector_recalibrated" / "results.json")["alarm_rates"]
    conds = (
        [f"cover_k{k}" for k in (2, 4, 8, 12, 16, 20)]
        + [f"target_k{k}" for k in (2, 4, 8, 12, 16, 20)]
        + ["tacit_all", "control_markup120"]
    )
    nice = (
        lambda c: c.replace("_k", " k=")
        .replace("control_markup120", "markup control")
        .replace("tacit_all", "tacit / focal")  # noqa: E731
    )
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 4.4), sharey=True)
    rows = []
    for i, c in enumerate(conds[::-1]):
        a = e1[f"{c}|vs|honest_mix"]
        ax.errorbar(
            a["auc"],
            i,
            xerr=[[a["auc"] - a["auc_ci"][0]], [a["auc_ci"][1] - a["auc"]]],
            fmt="o",
            color=BLUE,
            ms=5,
            lw=1.2,
        )
        t1 = e2["bidrig (E1)"][c]
        t2 = e2["env pooled (primary)"][c]
        ax2.plot(
            t1[0],
            i + 0.15,
            "s",
            color=ORANGE,
            ms=5,
            label="bidrig threshold (E1)" if i == 0 else None,
        )
        ax2.plot(
            t2[0],
            i - 0.15,
            "o",
            color=BLUE,
            ms=5,
            label="refitted on this env (E2)" if i == 0 else None,
        )
        ax2.hlines(i - 0.15, *t2[1], color=BLUE, lw=1.2)
        ax2.hlines(i + 0.15, *t1[1], color=ORANGE, lw=1.2)
        rows.append([c, a["auc"], *a["auc_ci"], t1[0], *t1[1], t2[0], *t2[1]])
    fa = e2["env pooled (primary)"]["honest_mix"][0]
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_yticks(range(len(conds)), [nice(c) for c in conds[::-1]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC vs honest episodes, mixed strategies")
    ax2.axvline(fa, color=INK2, lw=1, ls="--")
    ax2.text(
        fa + 0.02, len(conds) - 0.6, f"honest false alarms (E2) {fa:.3f}", color=INK2, fontsize=7.5
    )
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_xlabel("alarm rate (95 % CI)")
    ax2.legend(loc="center", bbox_to_anchor=(0.5, 0.42), fontsize=7.5)
    save(
        fig,
        "fig8_old_env_transfer",
        "The detector does not transfer to the old RL environment, even after recalibration",
        "Frozen full_gbm on scripted episodes (20 x 20). Most AUCs < 0.5; refitted threshold flags cartels no more than honest markets.",
    )
    write_csv(
        "fig8_old_env_transfer",
        ["condition", "auc_vs_mix", "lo", "hi", "alarm_e1", "lo", "hi", "alarm_e2", "lo", "hi"],
        rows,
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = load(RESULTS / "locked_1_full_gbm.json")["results"]
    fig_locked_mechanisms(res)
    fig_locked_baselines(res)
    fig_locked_false_alarms(res)
    fig_locked_precision(res)
    fig_validation_trace()
    fig_ablation()
    fig_real()
    fig_old_env()
    print("\n".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
