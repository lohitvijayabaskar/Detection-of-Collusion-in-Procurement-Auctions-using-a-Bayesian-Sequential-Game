"""Diagnostic: which arm does the bandit pick on the full TUNING split, per seed?

python scripts/rl_screens_arms.py --seeds 0-9 --tag rl_screens_arms_v1

Reads the tuning split only (never validation or test) and writes
results/rl_screens_arms_<tag>.{json,md}: the chosen arm and the mean out-of-fold reward of every
arm, per seed.
"""

import argparse
import json

import numpy as np
from _common import RESULTS, parse_seeds

from bidrig.datasets import FEATURE_VERSION, load_table
from bidrig.detectors.rl_screens import ARMS, RLScreens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    rows = {}
    for seed in parse_seeds(args.seeds):
        tune = load_table(seed, "tuning", args.scale)
        d = RLScreens().fit(tune.X, tune.y)
        rows[seed] = {
            "chosen": d.arm,
            "reward": {a: float(np.mean(v)) for a, v in d.arm_rewards.items()},
            "evaluations": {a: len(v) for a, v in d.arm_rewards.items()},
        }
        print(f"seed {seed}: chosen {d.arm}", flush=True)
    out = {"tag": args.tag, "split": "tuning", "feature_version": FEATURE_VERSION, "seeds": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"rl_screens_arms_{args.tag}.json").write_text(json.dumps(out, indent=1))
    names = list(ARMS)
    md = [f"# rl_screens arm choice on the tuning split (seeds {args.seeds})", ""]
    md += ["| seed | chosen | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for seed, r in rows.items():
        cells = " | ".join(f"{r['reward'][a]:.4f}" for a in names)
        md.append(f"| {seed} | {r['chosen']} | {cells} |")
    counts = {a: sum(r["chosen"] == a for r in rows.values()) for a in names}
    md += ["", "chosen counts: " + ", ".join(f"{a} {n}" for a, n in counts.items() if n)]
    (RESULTS / f"rl_screens_arms_{args.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
