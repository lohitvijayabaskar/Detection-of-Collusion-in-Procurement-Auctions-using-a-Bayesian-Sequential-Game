"""Score designs on the validation split (never the test split).

python scripts/run_validation.py --tag v1 --designs full_gbm,dep_gbm --seeds 0-9
Writes results/validation_<tag>.json and .md.
"""

import argparse
import json
import time

from _common import RESULTS, parse_seeds

from bidrig.datasets import FEATURE_VERSION
from bidrig.detectors import BASELINES
from bidrig.detectors.gbm_groups import DESIGNS
from bidrig.protocol import criteria, evaluate, score_split
from bidrig.report import comparison_table, design_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    designs = args.designs.split(",")
    all_designs = designs + [b for b in BASELINES if b not in designs]
    t0 = time.time()
    groups, sets = score_split(all_designs, "validation", parse_seeds(args.seeds), args.scale)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)
    res = evaluate(groups, sets, all_designs, BASELINES, n_boot=args.n_boot, compare=designs)
    crit = {d: criteria(res, d, BASELINES, sets) for d in designs}
    out = {
        "tag": args.tag,
        "split": "validation",
        "feature_version": FEATURE_VERSION,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "results": res,
        "criteria": crit,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"validation_{args.tag}.json").write_text(json.dumps(out, indent=1))
    md = [f"# Validation {args.tag} (seeds {args.seeds}, feature v{FEATURE_VERSION})", ""]
    md.append(comparison_table(res, all_designs, BASELINES))
    for d in designs:
        flags = {k: v for k, v in crit[d].items() if k != "detail"}
        md += ["", f"## {d}", "", f"criteria (validation): {flags}", "", design_table(res, d, sets)]
    text = "\n".join(md)
    (RESULTS / f"validation_{args.tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
