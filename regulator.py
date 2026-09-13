"""
regulator.py
====================================================================
Bayesian Sequential Regulator for Collusion Detection in Procurement
Auctions.

This module implements the "regulator" side of the game described in
the paper "Detecting Collusion in Procurement Auctions using a
Bayesian Sequential Game". It is deliberately decoupled from RLlib /
PettingZoo: it only ever consumes a dict of {agent_id: bid_value}
per round, i.e. exactly what a real-world regulator would observe
(public bids), never private costs, roles, or cartel identity. This
preserves the grey-box information asymmetry that is the whole point
of the model.

Pipeline
--------
1. compute_screens(bids)         -> sufficient statistics per round
2. calibrate(...)                -> fit P(screens | H0) and
                                     P(screens | H1) from simulation
3. BayesianSequentialRegulator   -> online, per-round:
     - Bayesian belief filter (HMM-style) over "cartel active"
     - CUSUM (Page's test) change-point statistic
     - Shiryaev-Roberts statistic (asymptotically optimal detection
       delay for a given false-alarm rate)
4. calibrate_alarm_threshold(...) -> set alarm thresholds from a
     Monte-Carlo null (H0-only) run so that the false-positive rate /
     average run length (ARL0) is controlled, rather than picking a
     magic number.

Math summary
------------
Let r_t = (r_{1,t}, ..., r_{N,t}) be the (latent, unobserved-by-
regulator) role vector at round t, and let H_t in {0,1} denote
"no cartel active" / "at least one cartel active". The environment
already implements a Markov chain over r_t (firms migrate in/out of
cartels based on trust and loss streaks), so H_t is itself
(approximately) Markov with some transition matrix Q.

The regulator only observes b_t (bids), from which it computes a
low-dimensional screen vector phi_t = (CV_t, NormDiff_t, Skew_t, ...).
Rather than assume a parametric form for P(phi_t | H_t) analytically
(hard, since bids are produced by a black-box neural policy), we
estimate the two conditional densities L0 = P(phi | H=0) and
L1 = P(phi | H=1) empirically via Monte-Carlo simulation under
forced regimes, using per-feature Gaussian KDE (a naive-Bayes style
factorization across screens for robustness with limited samples).

Given L0, L1, and a per-round log-likelihood ratio

    llr_t = log L1(phi_t) - log L0(phi_t)

the regulator maintains three complementary sequential statistics:

(a) HMM belief filter (if Q is known/assumed):
        pi_{t|t-1} = Q^T pi_{t-1}
        pi_t propto pi_{t|t-1} * [1 - p, p] .* exp([0, llr_t])  (informal)
    implemented below as an odds update, which is exact for the
    binary-state case.

(b) CUSUM / Page's test (model-free w.r.t. the transition dynamics,
    good when Q is unknown or non-stationary):
        S_t = max(0, S_{t-1} + llr_t),   alarm when S_t > h

(c) Shiryaev-Roberts statistic (minimizes expected detection delay
    for a given false-alarm rate, under a diffuse prior on the
    unknown change-point):
        R_t = (1 + R_{t-1}) * exp(llr_t),  alarm when R_t > A

Thresholds h and A are NOT hand-picked; calibrate_alarm_threshold()
runs the null (competitive-only) regime many times and sets the
threshold at the desired quantile of max_t S_t (equivalently R_t),
which is the standard way to control the average run length to a
false alarm (ARL0) in the change-point-detection literature.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from scipy import stats
from scipy.special import expit, logsumexp

# A single anomalous round (e.g. a screen far in the tail of both KDEs)
# should nudge the sequential statistics, not blow them up numerically.
# All per-feature log-likelihood contributions are clipped to this range.
LLR_CLIP = 8.0


# ====================================================================
# 1. Screens: the only thing the regulator ever sees
# ====================================================================

def compute_screens(bids: Dict[str, float], reserve_price: Optional[float] = None) -> Dict[str, float]:
    """
    Map a round's public bid vector to a small set of sufficient
    statistics ("screens") drawn from the empirical-IO collusion-
    screening literature (Porter & Zona 1993; Bajari & Ye 2003;
    Abrantes-Metz et al. 2006). Only bids are used -- no costs, no
    roles, no cartel identity.

    cv / norm_diff / skew are all SHAPE screens: each is invariant to
    multiplying every bid by the same positive constant. That means a
    cartel that inflates every firm's bid by the same proportional
    markup over its own (private) cost is, by construction, statistically
    indistinguishable from competitive bidding under these three alone
    -- no amount of calibration data fixes that, it's a property of the
    statistics. `level` closes that gap: bid-to-reserve-price is the
    standard "bid / engineer's estimate" anchor from the empirical
    screening literature, using the reserve price (which is announced
    to bidders, i.e. legitimately public) as the estimate. It is NOT
    scale-invariant, so a uniform markup inflation now shows up
    directly as a rightward shift in this feature even when it is
    invisible to the other three.
    """
    values = np.sort(np.array(list(bids.values()), dtype=float))
    n = len(values)
    mean = values.mean()
    std = values.std()

    cv = float(std / mean) if mean > 0 else 0.0

    diffs = np.diff(values)
    diff_1_2 = float(diffs[0]) if len(diffs) > 0 else 0.0
    mean_diff = float(diffs.mean()) if len(diffs) > 0 else 0.0
    norm_diff = diff_1_2 / mean_diff if mean_diff > 0 else 0.0

    try:
        skew = float(stats.skew(values)) if n > 2 else 0.0
        if np.isnan(skew):
            skew = 0.0
    except Exception:
        skew = 0.0

    screens = {"cv": cv, "norm_diff": norm_diff, "skew": skew}
    if reserve_price is not None and reserve_price > 0:
        screens["level"] = float(mean / reserve_price)
    return screens


def suspect_scores(bids: Dict[str, float]) -> Dict[str, float]:
    """
    Lightweight, market-level-detection-independent heuristic for
    *who* looks like a cover bidder: firms bidding suspiciously close
    together but comfortably above the lowest bid. Returns a z-score
    per firm (higher = more "clustered-above-winner", i.e. classic
    cover-bid signature). This is NOT part of the formal detection
    statistic -- treat it as a candidate feature for a future
    firm-level attribution model / equilibrium analysis.
    """
    items = sorted(bids.items(), key=lambda kv: kv[1])
    values = np.array([v for _, v in items])
    if len(values) < 3:
        return {k: 0.0 for k, _ in items}
    losers = values[1:]
    mu, sigma = losers.mean(), losers.std() if losers.std() > 0 else 1.0
    scores = {}
    for name, v in items[1:]:
        # negative z-score => tightly clustered just above the pack mean
        scores[name] = float(-(abs(v - mu) / sigma))
    scores[items[0][0]] = 0.0  # winner excluded
    return scores


# ====================================================================
# 2. Calibration: turn simulation into likelihoods
# ====================================================================

class KDELikelihood:
    """Per-feature Gaussian KDE likelihood, factorized (naive-Bayes
    style) across screens. Falls back to a tiny-variance Gaussian if
    a feature is (near-)degenerate under a regime."""

    def __init__(self, samples: Dict[str, np.ndarray]):
        self._kdes = {}
        for key, arr in samples.items():
            arr = np.asarray(arr, dtype=float)
            if arr.std() < 1e-6:
                arr = arr + np.random.normal(0, 1e-4, size=arr.shape)
            self._kdes[key] = stats.gaussian_kde(arr)

    def feature_logpdf(self, key: str, val: float) -> float:
        density = max(float(self._kdes[key].evaluate([val])[0]), 1e-300)
        return float(np.log(density))

    def __call__(self, screens: Dict[str, float]) -> float:
        density = 1.0
        for key, val in screens.items():
            density *= float(self._kdes[key].evaluate([val])[0])
        return max(density, 1e-300)


def force_regime(env, cartel_active: bool) -> None:
    """
    Mutate a freshly-reset env in place into a pure regime, for
    calibration purposes only. cartel_active=False collapses every
    firm to 'honest' (H0, competitive benchmark); cartel_active=True
    leaves whatever cartel assignment env.reset() produced (H1).
    """
    if not cartel_active:
        for agent in env.agents:
            env.roles[agent] = "honest"
        env.cartels = {k: [] for k in env.cartels}


def calibrate(env_factory: Callable[[], object],
              policy_fn: Callable[[np.ndarray], np.ndarray],
              n_episodes: int = 400,
              rounds_per_episode: int = 5,
              seed0: int = 0) -> Tuple[KDELikelihood, KDELikelihood]:
    """
    Monte-Carlo calibration of P(screens | H0) and P(screens | H1) by
    forcing the environment into each regime and rolling out the
    (already-trained, or mock) bidding policy.

    env_factory: zero-arg callable returning a fresh ProcurementHybridEnv
    policy_fn:   maps a single agent's observation array -> action array
    """
    samples = {0: {"cv": [], "norm_diff": [], "skew": []},
               1: {"cv": [], "norm_diff": [], "skew": []}}

    for hyp in (0, 1):
        for ep in range(n_episodes):
            env = env_factory()
            obs, _ = env.reset(seed=seed0 + ep)
            force_regime(env, cartel_active=bool(hyp))
            obs = env._get_obs()  # regenerate obs consistent with forced roles

            for _ in range(rounds_per_episode):
                actions = {a: policy_fn(obs[a]) for a in env.agents}
                obs, _, terms, _, infos = env.step(actions)

                bids = _extract_bids(infos, actions, env)
                screens = compute_screens(bids)
                for k, v in screens.items():
                    samples[hyp][k].append(v)

                if all(terms.values()):
                    break

    l0 = KDELikelihood({k: np.array(v) for k, v in samples[0].items()})
    l1 = KDELikelihood({k: np.array(v) for k, v in samples[1].items()})
    return l0, l1


def _extract_bids(infos, actions, env) -> Dict[str, float]:
    """Recover the round's bid vector regardless of which env variant
    is in use: the mock env exposes bids via infos[__global__ or per
    agent]; otherwise recompute from actions + env.private_costs using
    the same formula both env implementations use."""
    if "__global__" in infos:
        # hybrid_simulation_mock style: per-agent info has 'bid'
        return {a: infos[a]["bid"] for a in env.possible_agents if a in infos}
    bids = {}
    for a, act in actions.items():
        cost = env.private_costs[a]
        raw = cost * float(act[0])
        bids[a] = round(max(cost * 1.01, min(raw, env.reserve_price)), 2)
    return bids


# ====================================================================
# 3. Online sequential regulator
# ====================================================================

@dataclass
class RegulatorConfig:
    prior_h1: float = 0.10          # prior P(cartel active) at t=0
    transition: Optional[np.ndarray] = None  # 2x2 Markov kernel over H, or None
    cusum_threshold: float = 8.0    # overwritten by calibrate_alarm_threshold
    sr_threshold: float = 50.0      # overwritten by calibrate_alarm_threshold
    reserve_price: Optional[float] = None  # enables the level (bid/reserve) screen


@dataclass
class BayesianSequentialRegulator:
    likelihood_h0: KDELikelihood
    likelihood_h1: KDELikelihood
    config: RegulatorConfig = field(default_factory=RegulatorConfig)

    posterior: float = field(init=False)
    cusum: float = field(init=False, default=0.0)
    sr_stat: float = field(init=False, default=0.0)
    history: List[dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.posterior = self.config.prior_h1
        self._log_sr = -np.inf  # log(0); logaddexp(0, -inf) = 0 on first update

    def log_likelihood_ratio(self, screens: Dict[str, float]) -> float:
        """Sum of per-feature clipped log-likelihood-ratio contributions
        (naive-Bayes factorization). Clipping each feature independently
        keeps a single outlier screen from producing a numerically
        unstable, unbounded swing in the sequential statistics."""
        total = 0.0
        for key, val in screens.items():
            contrib = self.likelihood_h1.feature_logpdf(key, val) - \
                      self.likelihood_h0.feature_logpdf(key, val)
            total += float(np.clip(contrib, -LLR_CLIP, LLR_CLIP))
        return total

    def update(self, bids: Dict[str, float]) -> dict:
        screens = compute_screens(bids, reserve_price=self.config.reserve_price)
        llr = self.log_likelihood_ratio(screens)

        # All three statistics are propagated in log-space (log-odds /
        # log-R) and only mapped back with expit / exp at the end, so
        # they stay numerically stable even after long alarm runs.

        # (a) HMM-style Bayesian belief filter over H in {0,1}
        prior_odds = self.posterior / (1 - self.posterior + 1e-12)
        log_prior_odds = np.log(prior_odds + 1e-300)
        if self.config.transition is not None:
            q = self.config.transition
            pred = q[0, 1] * (1 - self.posterior) + q[1, 1] * self.posterior
            pred = min(max(pred, 1e-9), 1 - 1e-9)
            log_prior_odds = np.log(pred / (1 - pred))
        log_post_odds = log_prior_odds + llr
        self.posterior = float(expit(log_post_odds))

        # (b) CUSUM / Page's test -- robust to unknown transition dynamics
        self.cusum = max(0.0, self.cusum + llr)

        # (c) Shiryaev-Roberts, tracked as log(R_t) via log-sum-exp so it
        # never overflows even under a long run of positive evidence:
        #     R_t = (1 + R_{t-1}) * exp(llr_t)
        #  => log R_t = logaddexp(0, log R_{t-1}) + llr_t
        self._log_sr = logsumexp([0.0, self._log_sr]) + llr
        self.sr_stat = float(np.exp(min(self._log_sr, 700.0)))

        result = {
            "screens": screens,
            "llr": llr,
            "posterior_collusion": self.posterior,
            "cusum": self.cusum,
            "shiryaev_roberts": self.sr_stat,
            "log_shiryaev_roberts": float(self._log_sr),
            "alarm_cusum": self.cusum > self.config.cusum_threshold,
            "alarm_sr": self._log_sr > np.log(max(self.config.sr_threshold, 1e-300)),
            "suspects": suspect_scores(bids),
        }
        self.history.append(result)
        return result


def calibrate_alarm_threshold(env_factory, policy_fn, likelihood_h0, likelihood_h1,
                               n_episodes: int = 300, rounds_per_episode: int = 5,
                               false_alarm_rate: float = 0.05,
                               seed0: int = 10_000) -> Tuple[float, float]:
    """
    Run the *null* regime (H0: no cartel) many times, track the running
    max of CUSUM and Shiryaev-Roberts, and set thresholds at the
    (1 - false_alarm_rate) quantile of those maxima. This controls the
    probability of a false alarm within an episode at approximately
    false_alarm_rate, i.e. calibrated ARL0, instead of a magic number.
    """
    max_cusum, max_sr = [], []
    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        force_regime(env, cartel_active=False)
        obs = env._get_obs()

        reg = BayesianSequentialRegulator(likelihood_h0, likelihood_h1,
                                           RegulatorConfig(cusum_threshold=np.inf,
                                                           sr_threshold=np.inf))
        for _ in range(rounds_per_episode):
            actions = {a: policy_fn(obs[a]) for a in env.agents}
            obs, _, terms, _, infos = env.step(actions)
            bids = _extract_bids(infos, actions, env)
            reg.update(bids)
            if all(terms.values()):
                break

        max_cusum.append(max(h["cusum"] for h in reg.history))
        max_sr.append(max(h["log_shiryaev_roberts"] for h in reg.history))

    cusum_h = float(np.quantile(max_cusum, 1 - false_alarm_rate))
    # keep a strictly-positive floor: an all-zero null CUSUM quantile
    # would trigger alarms on any positive evidence at all.
    cusum_h = max(cusum_h, 1e-3)
    log_sr_h = float(np.quantile(max_sr, 1 - false_alarm_rate))
    sr_h = float(np.exp(min(log_sr_h, 700.0)))
    return cusum_h, sr_h


def calibrate_from_labeled_data(df,
                                 tender_id_col: str = "tender_id",
                                 bid_col: str = "bid",
                                 label_col: str = "collusive",
                                 min_bidders: int = 3) -> Tuple[KDELikelihood, KDELikelihood]:
    """
    Calibrate P(screens | H0) / P(screens | H1) directly from real,
    labeled procurement data, instead of from the RL simulator. This is
    the standard identification strategy in the empirical screening
    literature: label tenders as H1 (collusive) during a period a
    cartel is known/suspected to have operated, and H0 (competitive)
    from a benchmark period -- typically *after* the same cartel was
    broken up in the same market (Porter & Zona 1993; Bajari & Ye 2003;
    Huber & Imhof 2019). Publicly available labeled data covering 68-73
    prosecuted European cartels (2004-2021) is released by Silveira et
    al. (2022): https://zenodo.org/records/7111547 -- a natural
    external-validity benchmark for this exact pipeline.

    `df` should have one row per (tender, bidder): a tender id, a bid
    amount, and a tender-level binary label. If tenders vary a lot in
    scale, normalize `bid` first (e.g. bid / engineer's estimate) so
    that norm_diff/skew are comparable across contracts -- cv is
    already scale-invariant.
    """
    samples = {0: {"cv": [], "norm_diff": [], "skew": []},
               1: {"cv": [], "norm_diff": [], "skew": []}}

    for _, grp in df.groupby(tender_id_col):
        if len(grp) < min_bidders:
            continue  # screens are not meaningful with too few bids
        label = int(grp[label_col].iloc[0])
        bids = {f"b{i}": v for i, v in enumerate(grp[bid_col].values)}
        screens = compute_screens(bids)
        for k, v in screens.items():
            samples[label][k].append(v)

    for label in (0, 1):
        n = len(samples[label]["cv"])
        if n < 10:
            raise ValueError(
                f"Only {n} tenders with label={label}; need >=10 per class "
                "for a stable KDE. Real labeled cartel data is scarce -- "
                "consider pooling tenders across similar markets/years, or "
                "using calibrate_from_labeled_data alongside the simulator "
                "as a prior (see notes in the module docstring)."
            )

    l0 = KDELikelihood({k: np.array(v) for k, v in samples[0].items()})
    l1 = KDELikelihood({k: np.array(v) for k, v in samples[1].items()})
    return l0, l1


# ====================================================================
# 5. Demo: calibrate + run on a mixed episode, using the lightweight
#    mock env (no ray/torch needed) so this file is runnable standalone.
# ====================================================================

if __name__ == "__main__":
    from hybrid_simulation_mock import ProcurementHybridEnv, mock_trained_rl_policy

    print("Calibrating H0 / H1 likelihoods from simulation...")
    l0, l1 = calibrate(ProcurementHybridEnv, mock_trained_rl_policy,
                        n_episodes=150, rounds_per_episode=5)

    print("Calibrating alarm thresholds against the null regime "
          "(target false-alarm rate = 5%)...")
    cusum_h, sr_h = calibrate_alarm_threshold(ProcurementHybridEnv, mock_trained_rl_policy,
                                               l0, l1, n_episodes=150,
                                               false_alarm_rate=0.05)
    print(f"  -> CUSUM threshold h = {cusum_h:.3f}")
    print(f"  -> Shiryaev-Roberts threshold A = {sr_h:.3f}")

    print("\nRunning regulator on a fresh (natural, mixed-role) episode...\n")
    env = ProcurementHybridEnv()
    obs, infos = env.reset(seed=123)
    reg = BayesianSequentialRegulator(
        l0, l1,
        RegulatorConfig(prior_h1=0.1, cusum_threshold=cusum_h, sr_threshold=sr_h),
    )

    for r in range(1, 6):
        actions = {a: mock_trained_rl_policy(obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, step_infos = env.step(actions)
        bids = _extract_bids(step_infos, actions, env)
        result = reg.update(bids)

        n_colluding = sum(1 for a in env.possible_agents
                           if env.roles.get(a, "honest") != "honest")
        print(f"round {r}: true #firms-in-cartel={n_colluding:2d} | "
              f"P(cartel)={result['posterior_collusion']:.3f} | "
              f"CUSUM={result['cusum']:6.2f} (alarm={result['alarm_cusum']}) | "
              f"SR={result['shiryaev_roberts']:8.2f} (alarm={result['alarm_sr']})")
