"""Choose the stat_auc configuration on the VALIDATION split (never the test split).

python scripts/select_stat_auc.py --tag stat_auc_select_v1 --seeds 0-9

Scores all 16 grid configurations with the existing protocol (leave-one-scheme-out, validation
split) and picks the one with the highest macro AUC; ties go to the first configuration in grid
order. Writes results/stat_auc_selection_<tag>.{json,md}. The choice is then frozen by hand in
bidrig/detectors/stat_auc.py (CHOSEN), so it enters the design hash.
"""

import argparse
import json
import time

from _common import RESULTS, parse_seeds

from bidrig.datasets import FEATURE_VERSION
from bidrig.detectors.stat_auc import GRID, grid_name
from bidrig.protocol import evaluate, score_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=20, help="point estimates only are used")
    args = ap.parse_args()
    names = [grid_name(s, c) for s, c in GRID]
    t0 = time.time()
    groups, sets = score_split(names, "validation", parse_seeds(args.seeds), args.scale)
    print(f"scored {len(names)} configurations in {time.time() - t0:.0f}s", flush=True)
    res = evaluate(groups, sets, names, baselines=(), n_boot=args.n_boot, compare=[])
    macro = {n: res[f"{n}::macro_auc"]["point"] for n in names}
    best = names[0]
    for n in names:
        if macro[n] > macro[best]:  # strict: ties keep the earlier configuration
            best = n
    subset, c = GRID[names.index(best)]
    out = {
        "tag": args.tag,
        "split": "validation",
        "feature_version": FEATURE_VERSION,
        "seeds": args.seeds,
        "scale": args.scale,
        "macro_auc": macro,
        "chosen": {"subset": subset, "C": c, "name": best, "macro_auc": macro[best]},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"stat_auc_selection_{args.tag}.json").write_text(json.dumps(out, indent=1))
    md = [
        f"# stat_auc selection (validation, seeds {args.seeds})",
        "",
        "| subset | C | macro AUC |",
        "|---|---|---|",
    ]
    for n, (s, cc) in zip(names, GRID):
        star = "  <- chosen" if n == best else ""
        md.append(f"| {s} | {cc:g} | {macro[n]:.4f}{star} |")
    md += ["", f'CHOSEN = ("{subset}", {c!r})  # freeze this in bidrig/detectors/stat_auc.py']
    (RESULTS / f"stat_auc_selection_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
