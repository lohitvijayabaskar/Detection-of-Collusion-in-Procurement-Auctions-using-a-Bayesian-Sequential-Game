import numpy as np

import bidrig.detectors as detectors
from bidrig.detectors import rl_screens
from bidrig.detectors.base import make
from bidrig.features import ALL_FEATURES


def synthetic(seed=0, n=240):
    """Signal only in a dependence feature; every literature column is pure noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(ALL_FEATURES)))
    y = (np.arange(n) % 4 == 0).astype(int)
    X[y == 1, ALL_FEATURES.index("dep_pair_corr_pos")] += 3.0
    return X, y


def test_registered_and_arms_are_distinct_nonempty_subsets():
    assert "rl_screens" in detectors.REGISTRY
    sets = [frozenset(rl_screens.arm_features(a)) for a in rl_screens.ARMS]
    assert len(set(sets)) == len(rl_screens.ARMS) and all(sets)


def test_arm_features_exist_in_the_table():
    for a in rl_screens.ARMS:
        assert set(rl_screens.arm_features(a)) <= set(ALL_FEATURES)


def test_selects_a_signal_bearing_arm_and_scores_with_it():
    X, y = synthetic()
    d = make("rl_screens").fit(X, y)
    assert d.arm != "lit"  # the lit arm carries no signal here
    assert "dep_pair_corr_pos" in d.features
    s = d.score(X)
    assert s.shape == (len(y),) and np.isfinite(s).all()
    assert s[y == 1].mean() > s[y == 0].mean()
    assert set(d.arm_rewards) == set(rl_screens.ARMS)


def test_successive_halving_evaluates_survivors_more_often():
    X, y = synthetic(seed=1)
    d = make("rl_screens")
    d.select_arm(X, y)
    counts = sorted(len(v) for v in d.arm_rewards.values())
    assert counts[0] == 1 and counts[-1] == rl_screens.ROUNDS
    assert sum(c == rl_screens.ROUNDS for c in counts) == len(rl_screens.ARMS) // 2


def test_fit_is_deterministic():
    X, y = synthetic(seed=2)
    a = make("rl_screens").fit(X, y)
    b = make("rl_screens").fit(X, y)
    assert a.arm == b.arm and np.array_equal(a.score(X), b.score(X))


def test_fit_takes_only_the_training_data():
    """No validation or test data can reach the arm choice: ``fit`` accepts only (X, y)."""
    import inspect

    params = list(inspect.signature(rl_screens.RLScreens.fit).parameters)
    assert params == ["self", "X", "y"]
    params = list(inspect.signature(rl_screens.RLScreens.select_arm).parameters)
    assert params == ["self", "X", "y"]
