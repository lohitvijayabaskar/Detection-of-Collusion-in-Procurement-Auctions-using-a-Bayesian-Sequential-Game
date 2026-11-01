"""Score one detector variant against the reference on the VALIDATION split (never test).

python scripts/run_variant_validation.py --variant previous --seeds 0-9 --tag previous_v1

Writes results/variant_<tag>.json and .md: macro AUC with CIs, per-scheme AUC, false alarms on
each honest validation set, and the paired difference variant - reference.
"""

import argparse
import json
import time

from _common import RESULTS, parse_seeds

from bidrig.collusion import IN_SCOPE
from bidrig.datasets import FEATURE_VERSION
from bidrig.protocol import evaluate, score_split
from bidrig.report import fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--reference", default="full_gbm")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    designs = [args.variant, args.reference]
    t0 = time.time()
    groups, sets = score_split(designs, "validation", parse_seeds(args.seeds), args.scale)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)
    res = evaluate(
        groups,
        sets,
        designs,
        baselines=(args.reference,),
        n_boot=args.n_boot,
        compare=[args.variant],
    )
    out = {
        "tag": args.tag,
        "variant": args.variant,
        "reference": args.reference,
        "split": "validation",
        "feature_version": FEATURE_VERSION,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "scale": args.scale,
        "results": res,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"variant_{args.tag}.json").write_text(json.dumps(out, indent=1))
    r = lambda d, k: res[f"{d}::{k}"]  # noqa: E731
    pair = res[f"{args.variant}-{args.reference}::macro_auc"]
    md = [f"# Variant {args.variant} vs {args.reference} (validation, seeds {args.seeds})", ""]
    md += ["| design | macro LOMO AUC | macro prec@10% |", "|---|---|---|"]
    for d in designs:
        md.append(f"| {d} | {fmt(r(d, 'macro_auc'))} | {fmt(r(d, 'macro_prec10'))} |")
    md += ["", f"paired macro AUC difference ({args.variant} - {args.reference}): {fmt(pair)}", ""]
    md += ["| scheme | " + " | ".join(designs) + " | paired diff |", "|---|---|---|---|"]
    for m in IN_SCOPE:
        cells = [fmt(r(d, f"auc[{m}]")) for d in designs]
        diff = fmt(res[f"{args.variant}-{args.reference}::auc[{m}]"])
        md.append(f"| {m} | " + " | ".join(cells) + f" | {diff} |")
    md += ["", "| honest set | " + " | ".join(designs) + " |", "|---|---|---|"]
    for h in sorted(s for s in sets if s.startswith("honest@")):
        md.append(f"| {h} | " + " | ".join(fmt(r(d, f"fa[{h}]")) for d in designs) + " |")
    (RESULTS / f"variant_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
