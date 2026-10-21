"""Run the frozen tender-level leave-one-dataset-out evaluation on the real data, ONCE.

python scripts/run_real_lodo.py --note "approved by user on <date>"

The design hash covers bidrig/realeval.py, bidrig/realdata.py, bidrig/features.py,
bidrig/metrics.py and the GBM parameters. A second run with the same hash is refused.
Every run is appended to results/real_runs.jsonl; tables go to results/real_lodo_<n>.md.
"""

import argparse
import datetime
import hashlib
import json
import time

from _common import RESULTS, ROOT

from bidrig.models import GBM_PARAMS
from bidrig.realdata import load
from bidrig.realeval import DESIGNS, SINGLE, dataset_table, lodo

LOG = RESULTS / "real_runs.jsonl"
DATASETS_USED = ("brazil", "italy", "japan")
HASHED = ("realeval.py", "realdata.py", "features.py", "metrics.py")


def design_hash() -> str:
    h = hashlib.sha256(json.dumps([GBM_PARAMS, DATASETS_USED]).encode())
    for name in HASHED:
        h.update((ROOT / "bidrig" / name).read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    dh = design_hash()
    runs = (
        [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()] if LOG.exists() else []
    )
    if any(r["design_hash"] == dh for r in runs):
        raise SystemExit(f"real-data design {dh} already has a run; refusing")
    t0 = time.time()
    tables = {name: dataset_table(load(name)) for name in DATASETS_USED}
    print(f"features built in {time.time() - t0:.0f}s", flush=True)
    designs = list(DESIGNS) + [f"single:{s}" for s in SINGLE]
    res = lodo(tables, designs, n_boot=args.n_boot)
    n = len(runs) + 1
    entry = {
        "run": n,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "design_hash": dh,
        "note": args.note,
        "n_boot": args.n_boot,
        "tenders": {k: int(len(v["y"])) for k, v in tables.items()},
        "positives": {k: int(v["y"].sum()) for k, v in tables.items()},
        "seconds": round(time.time() - t0),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"real_lodo_{n}.json").write_text(json.dumps({**entry, "results": res}, indent=1))
    fmt = lambda v: f"{v['point']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"  # noqa: E731
    md = [f"# Real-data LODO run {n} ({dh})", "", f"note: {args.note}", ""]
    for held in sorted(res):
        md += [
            f"## held out: {held} ({entry['tenders'][held]} tenders, "
            f"{entry['positives'][held]} collusive)",
            "",
            "| design | AUC | prec@10% | TPR@5%FA | FA | base |",
            "|---|---|---|---|---|---|",
        ]
        for d in designs:
            r = res[held][d]
            md.append(
                f"| {d} | {fmt(r['auc'])} | {fmt(r['prec10'])} | {fmt(r['tpr'])} | "
                f"{fmt(r['fa'])} | {r['base_rate']['point']:.3f} |"
            )
        md += ["", "| paired AUC difference | value [95% CI] |", "|---|---|"]
        md += [f"| {k} | {fmt(v)} |" for k, v in res[held].items() if "-" in k]
        md.append("")
    (RESULTS / f"real_lodo_{n}.md").write_text("\n".join(md), encoding="utf-8")
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
