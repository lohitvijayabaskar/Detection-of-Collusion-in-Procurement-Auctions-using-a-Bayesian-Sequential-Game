"""Compare our detectors with re-implemented published methods on the VALIDATION split.

python scripts/run_literature_comparison.py --tag lit_cmp_v1 --seeds 0-9

Metrics (docs/literature_comparison_plan.md): macro leave-one-mechanism-out AUC with CIs,
worst-mechanism AUC, macro precision@10 %, macro true-positive rate and balanced accuracy at
the 5 % false-alarm threshold, highest false-alarm rate on any honest set, and the paired
macro-AUC difference to the reference. Writes results/literature_comparison_<tag>.{json,md}.
Never touches the test split.
"""

import argparse
import json
import time

import numpy as np
from _common import RESULTS, parse_seeds

from bidrig.collusion import IN_SCOPE
from bidrig.datasets import FEATURE_VERSION
from bidrig.detectors.published_methods import PUBLISHED_DETECTORS
from bidrig.protocol import evaluate, score_split
from bidrig.report import fmt

OURS = ("full_gbm", "previous", "stat_auc", "rl_screens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--reference", default="full_gbm")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    designs = list(OURS) + list(PUBLISHED_DETECTORS)
    t0 = time.time()
    groups, sets = score_split(designs, "validation", parse_seeds(args.seeds), args.scale)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)
    others = [d for d in designs if d != args.reference]
    res = evaluate(
        groups, sets, designs, baselines=(args.reference,), n_boot=args.n_boot, compare=others
    )
    honest = sorted(s for s in sets if s.startswith("honest@"))
    rows = {}
    for d in designs:
        r = lambda k: res[f"{d}::{k}"]  # noqa: E731
        tpr = float(np.mean([r(f"tpr[{m}]")["point"] for m in IN_SCOPE]))
        fa_mean = float(np.mean([r(f"fa[{h}]")["point"] for h in honest]))
        rows[d] = {
            "macro_auc": r("macro_auc"),
            "worst_mech_auc": min(r(f"auc[{m}]")["point"] for m in IN_SCOPE),
            "macro_prec10": r("macro_prec10"),
            "macro_tpr_at_5fa": tpr,
            "balanced_accuracy": (tpr + 1.0 - fa_mean) / 2.0,
            "max_fa": max(r(f"fa[{h}]")["point"] for h in honest),
            "paired_to_ref": (
                None if d == args.reference else res[f"{d}-{args.reference}::macro_auc"]
            ),
        }
    out = {
        "tag": args.tag,
        "split": "validation",
        "feature_version": FEATURE_VERSION,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "reference": args.reference,
        "rows": rows,
        "results": res,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"literature_comparison_{args.tag}.json").write_text(json.dumps(out, indent=1))
    md = [
        f"# Literature comparison (validation, seeds {args.seeds}, reference {args.reference})",
        "",
    ]
    md += [
        "| detector | macro LOMO AUC | worst scheme AUC | macro prec@10% | TPR@5%FA | "
        "balanced acc. | highest FA | paired AUC diff to ref |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(designs, key=lambda x: -rows[x]["macro_auc"]["point"]):
        w = rows[d]
        pair = "-" if w["paired_to_ref"] is None else fmt(w["paired_to_ref"])
        md.append(
            f"| {d} | {fmt(w['macro_auc'])} | {w['worst_mech_auc']:.3f} | "
            f"{fmt(w['macro_prec10'])} | {w['macro_tpr_at_5fa']:.3f} | "
            f"{w['balanced_accuracy']:.3f} | {w['max_fa']:.3f} | {pair} |"
        )
    md += ["", "Per-scheme AUC:", "", "| detector | " + " | ".join(IN_SCOPE) + " |"]
    md.append("|---|" + "---|" * len(IN_SCOPE))
    for d in designs:
        cells = [f"{res[f'{d}::auc[{m}]']['point']:.3f}" for m in IN_SCOPE]
        md.append(f"| {d} | " + " | ".join(cells) + " |")
    (RESULTS / f"literature_comparison_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
