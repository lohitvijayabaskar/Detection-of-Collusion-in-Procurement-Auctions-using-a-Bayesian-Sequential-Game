"""Run the LOCKED test split once for a frozen design.

python scripts/run_locked_test.py --design full_gbm --note "frozen after validation v3"

The frozen design is identified by a hash of its name, feature groups, GBM parameters,
feature version and the bidrig source files (including bidrig/detectors/). A second run with
the same hash is refused,
so the test split cannot be used for tuning. Every run is appended to
results/locked_runs.jsonl and its tables are written to results/locked_<n>_<design>.md.
"""

import argparse
import datetime
import hashlib
import json
import time

from _common import RESULTS, ROOT, parse_seeds

from bidrig.datasets import FEATURE_VERSION
from bidrig.models import (
    BASELINES,
    DESIGNS,
    FUSED,
    GBM_PARAMS,
    MONOTONE,
    TAIL_DIRECTIONS,
    TAIL_MARGIN,
)
from bidrig.protocol import criteria, evaluate, score_split
from bidrig.report import comparison_table, design_table

LOG = RESULTS / "locked_runs.jsonl"


def design_hash(design: str) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            [
                design,
                DESIGNS.get(design),
                FUSED.get(design),
                MONOTONE.get(design),
                GBM_PARAMS,
                TAIL_DIRECTIONS,
                TAIL_MARGIN,
                FEATURE_VERSION,
            ]
        ).encode()
    )
    package = ROOT / "bidrig"
    for p in sorted(package.glob("*.py")) + sorted((package / "detectors").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def previous_runs() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    dh = design_hash(args.design)
    runs = previous_runs()
    if any(r["design_hash"] == dh for r in runs):
        raise SystemExit(f"design {args.design} ({dh}) already has a locked run; refusing")
    designs = [args.design] + [b for b in BASELINES if b != args.design]
    t0 = time.time()
    groups, sets = score_split(designs, "test", parse_seeds(args.seeds))
    res = evaluate(groups, sets, designs, BASELINES, n_boot=args.n_boot, compare=[args.design])
    crit = criteria(res, args.design, BASELINES, sets)
    n = len(runs) + 1
    entry = {
        "run": n,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "design": args.design,
        "design_hash": dh,
        "feature_version": FEATURE_VERSION,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "note": args.note,
        "criteria": {k: v for k, v in crit.items() if k != "detail"},
        "seconds": round(time.time() - t0),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"locked_{n}_{args.design}.json").write_text(
        json.dumps({**entry, "results": res, "criteria_detail": crit["detail"]}, indent=1)
    )
    md = [f"# LOCKED test run {n}: {args.design} ({dh})", "", f"note: {args.note}", ""]
    md += [f"criteria: {entry['criteria']}", "", comparison_table(res, designs, BASELINES), ""]
    md += [design_table(res, args.design, sets)]
    (RESULTS / f"locked_{n}_{args.design}.md").write_text("\n".join(md))
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
