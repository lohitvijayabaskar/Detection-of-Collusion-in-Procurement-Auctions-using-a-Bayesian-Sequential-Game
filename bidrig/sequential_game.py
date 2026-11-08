"""Bayesian sequential game dynamics and time-varying metrics for procurement collusion detection.

Implements the time-varying and sequential mechanics required for the Bayesian Sequential Game:
1. Online Gradient Learner (instantaneous loss gradient norm, directional drift, online loss tracking)
2. Bayesian Sequential Belief Filter (HMM-style posterior, Page's CUSUM, Shiryaev-Roberts statistic)
3. Temporal Volatility & Trend Dynamics (rolling volatility, gap velocity, momentum MACD)

Organizes features into 5 foundational sequential game pillars:
- static       : classical price dispersion, shape, and margin screens
- identity     : cross-firm pair correlation, co-bidding, and relational cooperation
- bayes_seq    : dynamic Bayesian belief updating and change-point statistics
- grad_learn   : online gradient descent learner tracking deviation from competitive manifold
- temporal     : time-varying volatility, trend velocity, and moving-average divergence

Defines 29 systematic combinations across 5 tiers (Single, Pairwise, Triplet, Quad, All).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit


class OnlineGradientLearner:
    """Sequential online gradient learner tracking structural drift from competitive bidding.
    
    At each tender t in time order, updates a regularized online model on observed screens:
      y_hat_t = sigmoid(w_t^T x_t + b_t)
      gradient g_t = (y_hat_t - y_baseline) * x_t
      
    Extracts time-varying metrics:
      - grad_norm: instantaneous L2 magnitude of the gradient
      - grad_drift: cumulative gradient drift ||sum g_tau|| / sqrt(t)
      - grad_velocity: rolling rate of change of gradient norm
      - grad_online_loss: cumulative binary cross-entropy loss
    """

    def __init__(self, n_features: int = 5, learning_rate: float = 0.05, l2_reg: float = 0.01):
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cumulative_gradient = np.zeros(n_features)
        self.t = 0
        self.history_norms: List[float] = []

    def update(self, x: np.ndarray, baseline_prob: float = 0.15) -> Dict[str, float]:
        """Update learner with tender x, return instantaneous time-varying metrics."""
        self.t += 1
        x_clean = np.nan_to_num(x, nan=0.0)

        # Forward pass
        logit = float(np.dot(self.weights, x_clean) + self.bias)
        y_hat = float(expit(np.clip(logit, -10.0, 10.0)))

        # Gradient of loss with respect to baseline (deviation from competitive baseline)
        err = y_hat - baseline_prob
        grad_w = err * x_clean + self.l2_reg * self.weights
        grad_b = err

        norm_g = float(np.linalg.norm(grad_w))
        self.history_norms.append(norm_g)
        self.cumulative_gradient += grad_w

        # Cumulative drift
        drift = float(np.linalg.norm(self.cumulative_gradient) / np.sqrt(self.t))

        # Velocity (trend over last 5 tenders)
        if len(self.history_norms) >= 5:
            vel = float(self.history_norms[-1] - np.mean(self.history_norms[-5:-1]))
        else:
            vel = 0.0

        # Online cross-entropy loss
        eps = 1e-6
        loss = -float(baseline_prob * np.log(y_hat + eps) + (1.0 - baseline_prob) * np.log(1.0 - y_hat + eps))

        # SGD step
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b

        return {
            "grad_norm": norm_g,
            "grad_drift": drift,
            "grad_velocity": vel,
            "grad_online_loss": loss,
        }


class BayesianSequentialFilter:
    """Online Bayesian belief filter, Page's CUSUM, and Shiryaev-Roberts sequential tracking.
    
    Maintains:
      - bayes_post: P(Cartel active at t | history_{1:t})
      - cusum: S_t = max(0, S_{t-1} + llr_t)
      - shiryaev_roberts: R_t = (1 + R_{t-1}) * exp(llr_t)
    """

    def __init__(self, prior: float = 0.1, stay_prob: float = 0.98, llr_scale: float = 1.5):
        self.prior = prior
        self.stay_prob = stay_prob
        self.llr_scale = llr_scale
        self.posterior = prior
        self.cusum = 0.0
        self.shiryaev_roberts = 0.0

    def update(self, suspicion_score: float) -> Dict[str, float]:
        """Update filter with tender suspicion score in [0, 1]."""
        # HMM time update
        p_pred = self.stay_prob * self.posterior + (1.0 - self.stay_prob) * (1.0 - self.posterior)

        # Likelihood ratio based on suspicion
        susp = np.clip(suspicion_score, 0.01, 0.99)
        llr = self.llr_scale * float(np.log(susp / (1.0 - susp)))
        llr = np.clip(llr, -4.0, 4.0)

        # Measurement update
        lr = np.exp(llr)
        odds = (p_pred / (1.0 - p_pred + 1e-12)) * lr
        self.posterior = float(odds / (1.0 + odds))

        # CUSUM update
        self.cusum = float(max(0.0, self.cusum + llr - 0.2))

        # Shiryaev-Roberts update
        self.shiryaev_roberts = float(np.clip((1.0 + self.shiryaev_roberts) * lr, 0.0, 1e6))

        return {
            "seq_bayes_post": self.posterior,
            "seq_cusum": self.cusum,
            "seq_shiryaev_roberts": float(np.log1p(self.shiryaev_roberts)),
        }


class TemporalVolatilityTracker:
    """Tracks time-varying volatility, trend velocities, and moving-average divergence."""

    def __init__(self, window_short: int = 5, window_long: int = 20):
        self.w_short = window_short
        self.w_long = window_long
        self.cv_history: List[float] = []
        self.gap_history: List[float] = []
        self.susp_history: List[float] = []

    def update(self, cv: float, gap: float, susp: float) -> Dict[str, float]:
        self.cv_history.append(cv if np.isfinite(cv) else 0.1)
        self.gap_history.append(gap if np.isfinite(gap) else 0.05)
        self.susp_history.append(susp if np.isfinite(susp) else 0.5)

        # Rolling CV standard deviation
        if len(self.cv_history) >= 5:
            roll_cv_sd = float(np.std(self.cv_history[-min(10, len(self.cv_history)):]))
        else:
            roll_cv_sd = 0.0

        # Gap trend / velocity
        if len(self.gap_history) >= 5:
            recent_gap = self.gap_history[-min(5, len(self.gap_history)):]
            gap_trend = float(recent_gap[-1] - recent_gap[0])
        else:
            gap_trend = 0.0

        # MACD of suspicion: fast EMA - slow EMA
        if len(self.susp_history) >= self.w_long:
            ema_fast = float(np.mean(self.susp_history[-self.w_short:]))
            ema_slow = float(np.mean(self.susp_history[-self.w_long:]))
            macd = ema_fast - ema_slow
        else:
            macd = 0.0

        return {
            "temp_roll_cv_sd": roll_cv_sd,
            "temp_gap_trend": gap_trend,
            "temp_macd_suspicion": macd,
        }


# ==============================================================================
# The 5 Foundational Sequential Game Feature Pillars
# ==============================================================================

SEQ_GAME_PILLARS: Dict[str, Tuple[str, ...]] = {
    "static": ("scr_cv", "scr_spread", "scr_log_nbids", "scr_skew", "scr_kurt", "scr_rd", "scr_gap"),
    "identity": ("pair_close_max", "pair_close_top3", "pair_cobid_max", "pair_corr_max"),
    "bayes_seq": ("seq_bayes_post", "seq_cusum", "seq_shiryaev_roberts"),
    "grad_learn": ("grad_norm", "grad_drift", "grad_velocity", "grad_online_loss"),
    "temporal": ("temp_roll_cv_sd", "temp_gap_trend", "temp_macd_suspicion"),
}

# 29 Systematic Combinations across 5 Tiers
SEQ_GAME_COMBINATIONS_BY_TIER: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "all_metrics": {
        "all_seq_game_gbm": ("static", "identity", "bayes_seq", "grad_learn", "temporal"),
    },
    "single_metric": {
        "single_static_gbm": ("static",),
        "single_identity_gbm": ("identity",),
        "single_bayes_seq_gbm": ("bayes_seq",),
        "single_grad_learn_gbm": ("grad_learn",),
        "single_temporal_gbm": ("temporal",),
    },
    "pair_metrics": {
        "pair_static_identity_gbm": ("static", "identity"),
        "pair_static_bayes_gbm": ("static", "bayes_seq"),
        "pair_static_grad_gbm": ("static", "grad_learn"),
        "pair_static_temporal_gbm": ("static", "temporal"),
        "pair_identity_bayes_gbm": ("identity", "bayes_seq"),
        "pair_identity_grad_gbm": ("identity", "grad_learn"),
        "pair_identity_temporal_gbm": ("identity", "temporal"),
        "pair_bayes_grad_gbm": ("bayes_seq", "grad_learn"),
        "pair_bayes_temporal_gbm": ("bayes_seq", "temporal"),
        "pair_grad_temporal_gbm": ("grad_learn", "temporal"),
    },
    "triplet_metrics": {
        "triplet_static_identity_bayes_gbm": ("static", "identity", "bayes_seq"),
        "triplet_static_identity_grad_gbm": ("static", "identity", "grad_learn"),
        "triplet_static_bayes_grad_gbm": ("static", "bayes_seq", "grad_learn"),
        "triplet_static_grad_temporal_gbm": ("static", "grad_learn", "temporal"),
        "triplet_identity_bayes_grad_gbm": ("identity", "bayes_seq", "grad_learn"),
        "triplet_identity_grad_temporal_gbm": ("identity", "grad_learn", "temporal"),
        "triplet_bayes_grad_temporal_gbm": ("bayes_seq", "grad_learn", "temporal"),
        "triplet_static_identity_temporal_gbm": ("static", "identity", "temporal"),
    },
    "quad_metrics": {
        "quad_minus_static_gbm": ("identity", "bayes_seq", "grad_learn", "temporal"),
        "quad_minus_identity_gbm": ("static", "bayes_seq", "grad_learn", "temporal"),
        "quad_minus_bayes_gbm": ("static", "identity", "grad_learn", "temporal"),
        "quad_minus_grad_gbm": ("static", "identity", "bayes_seq", "temporal"),
        "quad_minus_temporal_gbm": ("static", "identity", "bayes_seq", "grad_learn"),
    },
}

ALL_SEQ_GAME_COMBINATIONS: Dict[str, Tuple[str, ...]] = {
    name: groups
    for tier in SEQ_GAME_COMBINATIONS_BY_TIER.values()
    for name, groups in tier.items()
}


def extract_sequential_game_features(
    bids_by_order: np.ndarray,
    estimates_by_order: np.ndarray,
    base_screens: Dict[str, np.ndarray],
    pair_features_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Extract time-varying sequential game features over a chronological series of tenders."""
    t_count = bids_by_order.shape[0]

    # Initialize online processors
    learner = OnlineGradientLearner(n_features=5, learning_rate=0.05)
    bayes_filter = BayesianSequentialFilter(prior=0.1)
    temp_tracker = TemporalVolatilityTracker(window_short=5, window_long=20)

    # Output feature dictionaries
    out_features: Dict[str, np.ndarray] = {
        # Static screens
        **base_screens,
        # Relational / Identity features
        **pair_features_dict,
        # Bayesian sequential features
        "seq_bayes_post": np.zeros(t_count),
        "seq_cusum": np.zeros(t_count),
        "seq_shiryaev_roberts": np.zeros(t_count),
        # Gradient learner features
        "grad_norm": np.zeros(t_count),
        "grad_drift": np.zeros(t_count),
        "grad_velocity": np.zeros(t_count),
        "grad_online_loss": np.zeros(t_count),
        # Temporal dynamics features
        "temp_roll_cv_sd": np.zeros(t_count),
        "temp_gap_trend": np.zeros(t_count),
        "temp_macd_suspicion": np.zeros(t_count),
    }

    for t in range(t_count):
        # Current tender screens
        cv_val = base_screens["scr_cv"][t]
        gap_val = base_screens["scr_gap"][t]
        skew_val = base_screens["scr_skew"][t]
        rd_val = base_screens["scr_rd"][t]
        spr_val = base_screens["scr_spread"][t]

        # Instantaneous heuristic suspicion score in [0, 1]
        cv_susp = 1.0 / (1.0 + np.exp(10.0 * (cv_val - 0.15))) if np.isfinite(cv_val) else 0.5
        gap_susp = 1.0 / (1.0 + np.exp(-10.0 * (gap_val - 0.05))) if np.isfinite(gap_val) else 0.5
        t_susp = float(np.clip(0.5 * cv_susp + 0.5 * gap_susp, 0.01, 0.99))

        # 1. Update Bayesian Sequential Filter
        bayes_res = bayes_filter.update(t_susp)
        out_features["seq_bayes_post"][t] = bayes_res["seq_bayes_post"]
        out_features["seq_cusum"][t] = bayes_res["seq_cusum"]
        out_features["seq_shiryaev_roberts"][t] = bayes_res["seq_shiryaev_roberts"]

        # 2. Update Online Gradient Learner
        feat_vector = np.array([
            cv_val if np.isfinite(cv_val) else 0.15,
            gap_val if np.isfinite(gap_val) else 0.05,
            skew_val if np.isfinite(skew_val) else 0.0,
            rd_val if np.isfinite(rd_val) else 0.5,
            spr_val if np.isfinite(spr_val) else 0.2,
        ])
        grad_res = learner.update(feat_vector, baseline_prob=0.15)
        out_features["grad_norm"][t] = grad_res["grad_norm"]
        out_features["grad_drift"][t] = grad_res["grad_drift"]
        out_features["grad_velocity"][t] = grad_res["grad_velocity"]
        out_features["grad_online_loss"][t] = grad_res["grad_online_loss"]

        # 3. Update Temporal Dynamics Tracker
        temp_res = temp_tracker.update(cv_val, gap_val, t_susp)
        out_features["temp_roll_cv_sd"][t] = temp_res["temp_roll_cv_sd"]
        out_features["temp_gap_trend"][t] = temp_res["temp_gap_trend"]
        out_features["temp_macd_suspicion"][t] = temp_res["temp_macd_suspicion"]

    return out_features

