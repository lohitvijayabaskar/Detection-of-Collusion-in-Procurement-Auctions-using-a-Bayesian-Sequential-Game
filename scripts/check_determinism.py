"""Check that the same seed gives identical features in separate processes and worker counts.

python scripts/check_determinism.py
"""

import subprocess
import sys

from _common import ROOT

CODE = (
    "import hashlib,numpy as np;"
    "from bidrig.datasets import build_table;"
    "t=build_table({seed},'{split}',scale={scale},workers={w});"
    "h=hashlib.sha256(np.nan_to_num(t.X,nan=-9).tobytes());h.update(t.set_name.tobytes());"
    "print(h.hexdigest())"
)


def digest(seed, split, scale, workers):
    code = CODE.format(seed=seed, split=split, scale=scale, w=workers)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return out.stdout.strip()


def main():
    ok = True
    for seed, split in ((0, "tuning"), (7, "test")):
        a = digest(seed, split, 0.05, 1)
        b = digest(seed, split, 0.05, 4)
        c = digest(seed + 1, split, 0.05, 1)
        same = a == b
        ok &= same and a != c
        print(
            f"seed {seed} {split}: 1 worker {a[:12]} | 4 workers {b[:12]} | seed+1 {c[:12]} "
            f"-> {'identical' if same else 'DIFFERENT'}"
        )
    print("DETERMINISM OK" if ok else "DETERMINISM FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
