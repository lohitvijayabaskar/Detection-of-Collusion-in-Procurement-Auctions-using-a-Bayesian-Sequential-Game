"""Benchmark splits, deterministic market generation and cached feature extraction.

Splits (each generated from its own seed stream, so no market appears in two splits):
  tuning      fit models. In-distribution specs, honest + all in-scope mechanisms.
  calib       honest markets from the tuning spec only; used to set the 5 % alarm threshold.
  validation  choose designs. In-distribution sets plus moderately shifted sizes/spreads.
  test        locked. In-distribution, unseen sizes (6-7, 25-30), unseen spreads, and
              honest stress sets (high markup, strong capacity effects, noisy estimates),
              plus the uniform-markup negative control.

The market rng is ``default_rng([seed, split_code, set_code, index])`` and the feature rng
appends a 1, so any market can be regenerated alone and results do not depend on worker
scheduling.
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bidrig.collusion import IN_SCOPE, MECHANISMS
from bidrig.features import ALL_FEATURES, market_features
from bidrig.simulator import MarketSpec, sample_config, simulate_market

FEATURE_VERSION = 3
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

BASE = MarketSpec()
VAL_SIZES = ((8, 9), (23, 24))
VAL_SPREADS = ((0.04, 0.06), (0.18, 0.22))
TEST_SIZES = ((6, 7), (25, 30))
TEST_SPREADS = ((0.02, 0.04), (0.22, 0.30))


@dataclass(frozen=True)
class SetSpec:
    name: str
    mechanism: str
    spec: MarketSpec
    count: int


def _mech_sets(prefix_spec: dict[str, MarketSpec], count: int) -> list[SetSpec]:
    return [
        SetSpec(f"{m}@{cond}", m, spec, count)
        for m in IN_SCOPE
        for cond, spec in prefix_spec.items()
    ]


def split_sets(split: str, scale: float = 1.0) -> list[SetSpec]:
    def c(n: int) -> int:
        return max(4, int(round(n * scale)))

    if split == "tuning":
        return [SetSpec("honest@in", "honest", BASE, c(400))] + _mech_sets({"in": BASE}, c(50))
    if split == "calib":
        return [SetSpec("honest@in", "honest", BASE, c(300))]
    if split == "validation":
        conds = {
            "in": BASE,
            "size": BASE.with_(n_firms=VAL_SIZES),
            "spread": BASE.with_(cost_spread=VAL_SPREADS),
        }
        honest = [SetSpec(f"honest@{k}", "honest", v, c(100)) for k, v in conds.items()]
        return honest + _mech_sets(conds, c(25))
    if split == "test":
        conds = {
            "in": BASE,
            "size": BASE.with_(n_firms=TEST_SIZES),
            "spread": BASE.with_(cost_spread=TEST_SPREADS),
        }
        stress = {
            "markup_high": BASE.with_(markup=((0.20, 0.25),)),
            "capacity": BASE.with_(capacity=((0.15, 0.30),)),
            "noisy": BASE.with_(estimate_noise=((0.20, 0.35),), idio_sd=((0.10, 0.15),)),
        }
        honest = [SetSpec(f"honest@{k}", "honest", v, c(100)) for k, v in conds.items()]
        honest += [SetSpec(f"honest@{k}", "honest", v, c(100)) for k, v in stress.items()]
        control = [SetSpec("uniform_markup@in", "uniform_markup", BASE, c(100))]
        return honest + _mech_sets(conds, c(40)) + control
    raise ValueError(split)


def code(name: str) -> int:
    return zlib.crc32(name.encode())


def market_rng(seed: int, split: str, set_name: str, i: int, stream: int = 0):
    return np.random.default_rng([seed, code(split), code(set_name), i, stream])


def generate_market(seed: int, split: str, s: SetSpec, i: int):
    rng = market_rng(seed, split, s.name, i)
    cfg = sample_config(s.spec, rng)
    return simulate_market(cfg, MECHANISMS[s.mechanism], rng)


def _features_for(args) -> tuple[list[float], dict]:
    seed, split, s, i = args
    m = generate_market(seed, split, s, i)
    f = market_features(m.bids, m.estimate, market_rng(seed, split, s.name, i, stream=1))
    meta = {
        "n_firms": m.config.n_firms,
        "n_tenders": m.config.n_tenders,
        "cartel_size": len(m.members),
    }
    return [f[k] for k in ALL_FEATURES], meta


@dataclass
class FeatureTable:
    X: np.ndarray
    y: np.ndarray
    set_name: np.ndarray
    mechanism: np.ndarray
    n_firms: np.ndarray
    cartel_size: np.ndarray

    def subset(self, keep: np.ndarray) -> "FeatureTable":
        return FeatureTable(*(getattr(self, k)[keep] for k in self.__dataclass_fields__))


def build_table(seed: int, split: str, scale: float = 1.0, workers: int | None = None):
    sets = split_sets(split, scale)
    jobs = [(seed, split, s, i) for s in sets for i in range(s.count)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        results = [_features_for(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_features_for, jobs, chunksize=16))
    X = np.array([r[0] for r in results], dtype=float)
    names = np.array([j[2].name for j in jobs])
    mech = np.array([j[2].mechanism for j in jobs])
    y = np.array([MECHANISMS[m].label for m in mech])
    return FeatureTable(
        X=X,
        y=y,
        set_name=names,
        mechanism=mech,
        n_firms=np.array([r[1]["n_firms"] for r in results]),
        cartel_size=np.array([r[1]["cartel_size"] for r in results]),
    )


def cache_path(seed: int, split: str, scale: float) -> Path:
    return CACHE_DIR / f"feat_v{FEATURE_VERSION}_s{seed}_{split}_x{scale:g}.npz"


def load_table(seed: int, split: str, scale: float = 1.0, workers: int | None = None):
    path = cache_path(seed, split, scale)
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return FeatureTable(**{k: d[k] for k in FeatureTable.__dataclass_fields__})
    table = build_table(seed, split, scale, workers)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: getattr(table, k) for k in FeatureTable.__dataclass_fields__})
    os.replace(tmp, path)
    return table
