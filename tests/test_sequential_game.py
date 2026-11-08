"""Tests for Bayesian sequential game dynamics and time-varying metrics."""

import numpy as np
import pytest

from bidrig.sequential_game import (
    ALL_SEQ_GAME_COMBINATIONS,
    BayesianSequentialFilter,
    OnlineGradientLearner,
    SEQ_GAME_COMBINATIONS_BY_TIER,
    SEQ_GAME_PILLARS,
    TemporalVolatilityTracker,
)


def test_seq_game_tier_definitions():
    assert len(ALL_SEQ_GAME_COMBINATIONS) == 29
    assert len(SEQ_GAME_COMBINATIONS_BY_TIER["all_metrics"]) == 1
    assert len(SEQ_GAME_COMBINATIONS_BY_TIER["single_metric"]) == 5
    assert len(SEQ_GAME_COMBINATIONS_BY_TIER["pair_metrics"]) == 10
    assert len(SEQ_GAME_COMBINATIONS_BY_TIER["triplet_metrics"]) == 8
    assert len(SEQ_GAME_COMBINATIONS_BY_TIER["quad_metrics"]) == 5


def test_online_gradient_learner():
    learner = OnlineGradientLearner(n_features=3, learning_rate=0.1)
    x = np.array([0.2, 0.05, 1.2])

    # First update
    res1 = learner.update(x, baseline_prob=0.15)
    assert "grad_norm" in res1
    assert "grad_drift" in res1
    assert "grad_velocity" in res1
    assert "grad_online_loss" in res1
    assert res1["grad_norm"] >= 0.0
    assert res1["grad_drift"] >= 0.0

    # Repeat updates to verify drift accumulation
    for _ in range(5):
        learner.update(x, baseline_prob=0.15)
    assert learner.t == 6


def test_bayesian_sequential_filter():
    f = BayesianSequentialFilter(prior=0.1)
    # Repeated suspicious updates should increase posterior and CUSUM
    for _ in range(5):
        res = f.update(0.9)
    assert res["seq_bayes_post"] > 0.1
    assert res["seq_cusum"] > 0.0
    assert res["seq_shiryaev_roberts"] > 0.0

    # Competitive updates should cool down
    for _ in range(10):
        res_cool = f.update(0.05)
    assert res_cool["seq_bayes_post"] < res["seq_bayes_post"]


def test_temporal_volatility_tracker():
    tracker = TemporalVolatilityTracker()
    for t in range(10):
        res = tracker.update(cv=0.1 + 0.01 * t, gap=0.05 + 0.02 * t, susp=0.5)
    assert "temp_roll_cv_sd" in res
    assert "temp_gap_trend" in res
    assert "temp_macd_suspicion" in res
    assert res["temp_gap_trend"] > 0.0

