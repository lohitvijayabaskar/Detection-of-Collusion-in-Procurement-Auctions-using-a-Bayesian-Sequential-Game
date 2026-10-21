"""Generate markets and cache their features: python scripts/build_features.py --seeds 0-9"""

import argparse
import time

from _common import parse_seeds

from bidrig.datasets import cache_path, load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--splits", default="tuning,calib,validation")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    for seed in parse_seeds(args.seeds):
        for split in args.splits.split(","):
            t0 = time.time()
            existed = cache_path(seed, split, args.scale).exists()
            table = load_table(seed, split, args.scale, args.workers)
            state = "cached" if existed else f"built in {time.time() - t0:.0f}s"
            print(f"seed {seed} {split}: {len(table.y)} markets ({state})", flush=True)


if __name__ == "__main__":
    main()
