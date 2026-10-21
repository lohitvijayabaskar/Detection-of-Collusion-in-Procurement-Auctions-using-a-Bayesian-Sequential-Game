import subprocess
import sys
from pathlib import Path

import numpy as np

from bidrig.collusion import IN_SCOPE
from bidrig.datasets import build_table, split_sets
from bidrig.features import ALL_FEATURES
from bidrig.models import BASELINES, DESIGNS, GBM, SingleScreen, make_design
from bidrig.protocol import criteria, evaluate, metric_functions

ROOT = Path(__file__).resolve().parent.parent


def test_single_screen_picks_sign():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(ALL_FEATURES)))
    y = np.r_[np.ones(100), np.zeros(100)].astype(int)
    col = ALL_FEATURES.index("lit_cv_mean")
    X[:100, col] -= 3  # collusion lowers CV
    s = SingleScreen("lit_cv_mean").fit(X, y)
    assert s.sign == -1.0 and s.score(X)[:100].mean() > s.score(X)[100:].mean()


def test_gbm_learns_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(ALL_FEATURES)))
    y = (rng.random(300) < 0.5).astype(int)
    X[:, ALL_FEATURES.index("dep_pair_corr_pos")] += 3 * y
    g = GBM(("dep",)).fit(X, y)
    s = g.score(X)
    assert s[y == 1].mean() > s[y == 0].mean() + 0.5


def test_designs_and_baselines_construct():
    for name in list(DESIGNS) + list(BASELINES):
        assert make_design(name) is not None


def test_splits_are_disjoint_and_cover_mechanisms():
    test_sets = {s.name for s in split_sets("test")}
    for m in IN_SCOPE:
        assert {f"{m}@in", f"{m}@size", f"{m}@spread"} <= test_sets
    assert "uniform_markup@in" in test_sets
    tuning = {s.mechanism for s in split_sets("tuning")}
    assert "uniform_markup" not in tuning
    # test sizes never appear in tuning/validation ranges
    from bidrig.datasets import BASE, TEST_SIZES, VAL_SIZES

    used = set(range(BASE.n_firms[0][0], BASE.n_firms[0][1] + 1))
    used |= {v for a, b in VAL_SIZES for v in range(a, b + 1)}
    assert not used & {v for a, b in TEST_SIZES for v in range(a, b + 1)}


def test_protocol_smoke_end_to_end(tmp_path, monkeypatch):
    """Tiny tables -> scores -> metrics -> criteria, no caching."""
    import bidrig.protocol as protocol

    tables = {
        split: build_table(0, split, scale=0.06, workers=1)
        for split in ("tuning", "calib", "validation")
    }
    monkeypatch.setattr(protocol, "load_table", lambda seed, split, scale, w: tables[split])
    groups, sets = protocol.score_split(["dep_gbm", "single:lit_cv_mean"], "validation", [0, 1])
    fns = metric_functions("dep_gbm", sets)
    assert "macro_auc" in fns and f"auc[{IN_SCOPE[0]}@size]" in fns
    res = evaluate(
        groups, sets, ["dep_gbm", "single:lit_cv_mean"], ["single:lit_cv_mean"], n_boot=20
    )
    assert 0 <= res["dep_gbm::macro_auc"]["point"] <= 1
    crit = criteria(res, "dep_gbm", ["single:lit_cv_mean"], sets)
    assert set(crit) >= {"C1_lomo_auc>=0.85", "C2_fa<=0.10_every_honest_set"}


def test_same_seed_identical_across_processes():
    code = (
        "import hashlib,numpy as np;from bidrig.datasets import build_table;"
        "t=build_table(3,'calib',scale=0.02,workers=1);"
        "print(hashlib.sha256(np.nan_to_num(t.X,nan=-9).tobytes()).hexdigest())"
    )
    outs = [
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        for _ in range(2)
    ]
    assert outs[0].returncode == 0, outs[0].stderr
    assert outs[0].stdout == outs[1].stdout and len(outs[0].stdout.strip()) == 64
