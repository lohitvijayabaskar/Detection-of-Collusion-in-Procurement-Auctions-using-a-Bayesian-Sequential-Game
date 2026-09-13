"""
emergent_procurement_env.py
====================================================================
Closed-loop, emergent-collusion procurement environment.

This supersedes the scripted-cartel version (hybrid_simulation_mock.py /
train_hybrid_rl.py) to address the novelty gap: instead of assigning
firms a ground-truth role ('honest' / 'Cartel A' / 'Cartel B') and
hand-writing a profit-sharing + grim-trigger reward to *teach* them to
collude, this environment gives every firm an IDENTICAL, purely
individualistic reward (its own profit). If collusive bidding patterns
appear, they are an emergent property of repeated-game reward
maximization under a shared self-play policy -- the standard mechanism
behind tacit algorithmic collusion results (Calvano, Calzolari, Denicolo
& Pastorello 2020; Klein 2021; Banchio & Skrzypacz 2022) -- not a
scripted regime.

The second change is structural: the Bayesian-sequential regulator from
regulator.py is no longer a passive, offline, post-hoc observer that
scores an already-frozen policy. It is instantiated *inside* the
environment and updated every round from the public bid vector. Its
running posterior / CUSUM / Shiryaev-Roberts statistics are:
  1. fed back into every firm's observation (so firms can react to how
     suspicious they currently look), and
  2. used to levy an in-game fine on firms whose bids match the
     collusive "cover-bid" signature whenever an alarm fires.

That makes detection part of the environment's transition dynamics
rather than an external evaluation metric, turning this into a genuine
two-sided, closed-loop multiagent problem: firms adapt to evade
detection, and any resulting equilibrium (do firms still collude? at a
lower markup, just under the detection boundary? not at all?) is an
empirical question answered by the trained policies, not baked in.

Calibrating the regulator's H0/H1 likelihoods still requires *some*
benchmark notion of "competitive" vs. "collusive" bidding -- exactly as
a real regulator calibrates screens against historical cartel /
non-cartel tenders. We do this once, offline, with two simple
synthetic reference bidders (see `calibrate_emergent_likelihoods`)
that are NEVER used inside the training env itself and never define
firm identity -- they only seed the two densities the regulator's
log-likelihood ratio is computed against. This preserves the grey-box
property (the regulator's calibration data is independent of, and
cruder than, whatever the trained firms actually learn to do).
"""

from __future__ import annotations

import random
from typing import Dict as TypingDict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from regulator import (
    BayesianSequentialRegulator,
    RegulatorConfig,
    KDELikelihood,
    compute_screens,
    suspect_scores,
)


# =====================================================================
# 1. Emergent-collusion environment
# =====================================================================
class ProcurementEmergentEnv(ParallelEnv):
    """
    Repeated first-price-style procurement auction with:
      - No scripted roles / cartel identity. Every firm has the same
        reward function (its own profit, minus any regulator fine).
      - A longer horizon per episode than the scripted version, since
        tacit collusion via reciprocal/punishment strategies needs
        enough repeated rounds to be learnable at all.
      - An embedded Bayesian-sequential regulator, updated every round,
        whose belief state is (a) observed by firms and (b) used to
        fine suspected cover bidders when it alarms -- i.e. the
        regulator is part of the environment's dynamics, not a
        downstream evaluation script.
    """
    metadata = {"render_modes": ["human"], "name": "procurement_emergent_v0"}

    def __init__(
        self,
        reserve_price: float = 100.0,
        total_firms: int = 20,
        rounds_per_episode: int = 20,
        likelihood_h0: Optional[KDELikelihood] = None,
        likelihood_h1: Optional[KDELikelihood] = None,
        prior_h1: float = 0.10,
        cusum_threshold: float = 8.0,
        sr_threshold: float = 50.0,
        fine_rate: float = 1.5,
    ):
        super().__init__()
        self.reserve_price = reserve_price
        self.total_firms = total_firms
        self.rounds_per_episode = rounds_per_episode
        self.fine_rate = fine_rate

        self.possible_agents = [f"firm{i:02d}" for i in range(1, self.total_firms + 1)]
        self.agents = self.possible_agents[:]

        self._action_spaces = {
            agent: Box(low=1.0, high=2.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }
        # Obs: [private_cost, prev_winning_bid, regulator_posterior,
        #       regulator_alarm, round_progress]
        # NOTE: no is_in_cartel / is_designated -- there is no scripted
        # identity left to observe. The regulator features replace them.
        self._observation_spaces = {
            agent: Box(low=0.0, high=float(self.reserve_price), shape=(5,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.likelihood_h0 = likelihood_h0
        self.likelihood_h1 = likelihood_h1
        self.regulator_config = RegulatorConfig(
            prior_h1=prior_h1,
            cusum_threshold=cusum_threshold,
            sr_threshold=sr_threshold,
            reserve_price=reserve_price,
        )

        self.private_costs = {}
        self.prev_winning_bid = self.reserve_price
        self.round_num = 0
        self.regulator: Optional[BayesianSequentialRegulator] = None
        self._last_posterior = prior_h1
        self._last_alarm = 0.0

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def set_likelihoods(self, l0: KDELikelihood, l1: KDELikelihood) -> None:
        """Allows the training script to inject freshly-calibrated
        likelihoods after construction (e.g. from env_creator config)."""
        self.likelihood_h0 = l0
        self.likelihood_h1 = l1

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.round_num = 0
        self.prev_winning_bid = self.reserve_price
        self.private_costs = {
            agent: round(random.uniform(50.0, 60.0), 2) for agent in self.agents
        }

        if self.likelihood_h0 is not None and self.likelihood_h1 is not None:
            self.regulator = BayesianSequentialRegulator(
                self.likelihood_h0, self.likelihood_h1, self.regulator_config
            )
        else:
            self.regulator = None  # regulator disabled (e.g. ablation baseline)

        self._last_posterior = self.regulator_config.prior_h1
        self._last_alarm = 0.0

        return self._get_obs(), {a: {} for a in self.agents}

    def _get_obs(self):
        # NOTE: round_num is incremented at the top of step() before
        # _get_obs() is called, so by the time reset()/step() hands this
        # back it already reflects "rounds completed so far" -- do NOT
        # add +1 here, that overshoots to >1.0 (and >reserve_price once
        # scaled) on the final round and violates the declared Box space.
        round_progress = self.round_num / self.rounds_per_episode
        obs = {}
        for agent in self.agents:
            obs[agent] = np.array(
                [
                    self.private_costs[agent],
                    self.prev_winning_bid,
                    self._last_posterior * self.reserve_price,  # scale into obs range
                    self._last_alarm * self.reserve_price,
                    round_progress * self.reserve_price,
                ],
                dtype=np.float32,
            )
        return obs

    def step(self, actions: TypingDict[str, np.ndarray]):
        self.round_num += 1
        bids = {}
        for agent, action_array in actions.items():
            markup = float(action_array[0])
            cost = self.private_costs[agent]
            raw_bid = cost * markup
            bids[agent] = round(max(cost * 1.00, min(raw_bid, self.reserve_price)), 2)

        valid_bids = {k: v for k, v in bids.items() if v <= self.reserve_price}
        winner = min(valid_bids, key=valid_bids.get)
        winning_bid = valid_bids[winner]
        self.prev_winning_bid = winning_bid

        # ---- Base reward: pure individual profit, identical for every
        # firm. No role, no shared cartel pool, no scripted penalty. ----
        rewards = {agent: 0.0 for agent in self.agents}
        for agent in self.agents:
            if agent == winner:
                rewards[agent] = winning_bid - self.private_costs[agent]

        # ---- Regulator: updates its Bayesian-sequential belief from
        # the public bid vector, then (endogenously) fines firms whose
        # bids match the cover-bid signature if it currently alarms. ----
        posterior, alarm, susp = self._last_posterior, 0.0, {}
        if self.regulator is not None:
            result = self.regulator.update(bids)
            posterior = result["posterior_collusion"]
            alarm = 1.0 if (result["alarm_cusum"] or result["alarm_sr"]) else 0.0
            susp = result["suspects"]  # suspect_scores(): <=0, ~0 = most suspicious

            if alarm:
                for agent in self.agents:
                    suspicion = float(np.exp(susp.get(agent, -50.0)))  # in (0, 1]
                    fine = self.fine_rate * suspicion * self.reserve_price * 0.01
                    rewards[agent] -= fine

        self._last_posterior = posterior
        self._last_alarm = alarm

        infos = {}
        for agent in self.agents:
            infos[agent] = {
                "cost": self.private_costs[agent],
                "bid": bids[agent],
                "markup": bids[agent] / self.private_costs[agent],
                "reward": rewards[agent],
                "suspicion": susp.get(agent, 0.0),
            }
        infos["__common__"] = {
            "winner": winner,
            "winning_bid": winning_bid,
            "posterior_collusion": posterior,
            "alarm": alarm,
        }

        terminations = {a: (self.round_num >= self.rounds_per_episode) for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = self._get_obs()

        if self.round_num >= self.rounds_per_episode:
            self.agents = []

        return obs, rewards, terminations, truncations, infos


# =====================================================================
# 2. Calibration reference bidders (used ONLY to seed the regulator's
#    H0/H1 densities -- never used as firm behaviour during training)
# =====================================================================
def _competitive_calibration_bidder(cost: float, reserve_price: float) -> float:
    """Reference H0 behaviour: bid close to cost, i.e. what firms with
    no ability to coordinate are forced into by Bertrand-style
    undercutting pressure."""
    markup = random.uniform(1.00, 1.02)
    return round(min(cost * markup, reserve_price), 2)


def _collusive_calibration_bidder(cost: float, reserve_price: float) -> float:
    """Reference H1 behaviour: bid close to the reserve price, i.e.
    what a fully successful cartel with no competitive pressure would
    do. This is a deliberately crude, purely synthetic benchmark -- it
    defines the *statistical signature* the regulator screens for, not
    the strategy any trained firm is scripted to follow.

    On its own this ONE archetype (near-identical target price) leaves
    the H1 density with almost no support anywhere else in collusion
    space -- e.g. a uniform proportional markup inflation lands outside
    both H0 and H1's support and the detector can't score it at all.
    Use `_mixed_collusive_calibration_bidder` below for actual
    calibration; this function is kept as one archetype in that mix.
    """
    markup = random.uniform(0.90, 0.98) * (reserve_price / cost)
    return round(min(cost * markup, reserve_price), 2)


def _uniform_markup_calibration_bidder(cost: float, reserve_price: float) -> float:
    """H1 archetype: every firm applies (approximately) the same
    proportional markup over its OWN cost. This is the pattern that is
    invisible to scale-invariant screens (cv/norm_diff/skew) but shows
    up in `level`. Severity is randomized per-episode by the caller so
    the resulting KDE has support across mild-to-severe uniform markup,
    not just one fixed level."""
    markup = _UNIFORM_MARKUP_SEVERITY[0]
    return round(min(cost * markup, reserve_price), 2)


_UNIFORM_MARKUP_SEVERITY = [1.2]  # mutated per-episode during calibration


def _complementary_calibration_bidder(cost: float, reserve_price: float, is_designated: bool) -> float:
    """H1 archetype: one designated low bidder near cost, the rest bid
    a common elevated markup (classic cover-bidding)."""
    if is_designated:
        markup = random.uniform(1.00, 1.06)
    else:
        markup = _UNIFORM_MARKUP_SEVERITY[0]
    return round(min(cost * markup, reserve_price), 2)


def calibrate_emergent_likelihoods(
    reserve_price: float = 100.0,
    total_firms: int = 20,
    n_episodes: int = 300,
    rounds_per_episode: int = 20,
    seed0: int = 0,
) -> Tuple[KDELikelihood, KDELikelihood]:
    """
    Seed P(screens | H0) and P(screens | H1) from the two synthetic
    reference bidders above, applied to cost draws from the *same*
    cost distribution the real env uses. This is analogous to a real
    regulator calibrating screens against a stylized competitive
    benchmark and a stylized cartel benchmark before ever looking at
    the market under investigation.
    """
    samples = {0: {"cv": [], "norm_diff": [], "skew": [], "level": []},
               1: {"cv": [], "norm_diff": [], "skew": [], "level": []}}

    # ---- H0: competitive reference, one archetype (bid near own cost) ----
    rng0 = random.Random(seed0)
    for ep in range(n_episodes):
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng0.uniform(50.0, 60.0) for i in range(total_firms)}
            bids = {k: _competitive_calibration_bidder(c, reserve_price) for k, c in costs.items()}
            screens = compute_screens(bids, reserve_price=reserve_price)
            for k, v in screens.items():
                samples[0][k].append(v)

    # ---- H1: MIXTURE of collusive archetypes, each with randomized
    # severity. A single archetype leaves huge gaps in the H1 density's
    # support (e.g. a uniform-markup cartel lands outside both KDEs and
    # can't be scored at all); mixing archetypes + severities gives the
    # detector real coverage over "what collusion can look like". ----
    rng1 = random.Random(seed0 + 1)
    for ep in range(n_episodes):
        archetype = rng1.choice(["target_price", "uniform_markup", "complementary"])
        _UNIFORM_MARKUP_SEVERITY[0] = rng1.uniform(1.10, 1.60)
        designated = f"f0"
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng1.uniform(50.0, 60.0) for i in range(total_firms)}
            if archetype == "target_price":
                bids = {k: _collusive_calibration_bidder(c, reserve_price) for k, c in costs.items()}
            elif archetype == "uniform_markup":
                bids = {k: _uniform_markup_calibration_bidder(c, reserve_price) for k, c in costs.items()}
            else:
                bids = {k: _complementary_calibration_bidder(c, reserve_price, k == designated)
                         for k, c in costs.items()}
            screens = compute_screens(bids, reserve_price=reserve_price)
            for k, v in screens.items():
                samples[1][k].append(v)

    l0 = KDELikelihood({k: np.array(v) for k, v in samples[0].items()})
    l1 = KDELikelihood({k: np.array(v) for k, v in samples[1].items()})
    return l0, l1


def calibrate_emergent_thresholds(
    l0: KDELikelihood,
    l1: KDELikelihood,
    reserve_price: float = 100.0,
    total_firms: int = 20,
    n_episodes: int = 200,
    rounds_per_episode: int = 20,
    false_alarm_rate: float = 0.05,
    seed0: int = 10_000,
) -> Tuple[float, float]:
    """Calibrate CUSUM / Shiryaev-Roberts thresholds against the
    competitive reference bidder. Same idea as calibrate_alarm_threshold
    in regulator.py (run the null regime many times, set thresholds at
    a quantile of the running max), but implemented directly against
    bids rather than against a role-based env, since this env has no
    roles to force into a regime -- there is nothing to force, the
    "null regime" here IS the competitive reference bidder."""
    max_cusum, max_log_sr = [], []
    for ep in range(n_episodes):
        rng = random.Random(seed0 + ep)
        reg = BayesianSequentialRegulator(
            l0, l1, RegulatorConfig(cusum_threshold=np.inf, sr_threshold=np.inf)
        )
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng.uniform(50.0, 60.0) for i in range(total_firms)}
            bids = {k: _competitive_calibration_bidder(c, reserve_price) for k, c in costs.items()}
            reg.update(bids)
        max_cusum.append(max(h["cusum"] for h in reg.history))
        max_log_sr.append(max(h["log_shiryaev_roberts"] for h in reg.history))

    cusum_h = float(np.quantile(max_cusum, 1 - false_alarm_rate))
    cusum_h = max(cusum_h, 1e-3)
    log_sr_h = float(np.quantile(max_log_sr, 1 - false_alarm_rate))
    sr_h = float(np.exp(min(log_sr_h, 700.0)))
    return cusum_h, sr_h


if __name__ == "__main__":
    print("Calibrating regulator H0/H1 likelihoods from reference bidders...")
    l0, l1 = calibrate_emergent_likelihoods(n_episodes=150, rounds_per_episode=20)

    print("Calibrating alarm thresholds against the competitive reference...")
    cusum_h, sr_h = calibrate_emergent_thresholds(l0, l1, n_episodes=100, rounds_per_episode=20)
    print(f"  -> CUSUM threshold h = {cusum_h:.3f}")
    print(f"  -> Shiryaev-Roberts threshold A = {sr_h:.3f}")

    env = ProcurementEmergentEnv(
        likelihood_h0=l0, likelihood_h1=l1,
        cusum_threshold=cusum_h, sr_threshold=sr_h,
        rounds_per_episode=20,
    )
    obs, _ = env.reset(seed=42)

    print("\nRunning a random-policy smoke test episode...\n")
    for r in range(1, 21):
        actions = {a: np.array([random.uniform(1.0, 1.3)], dtype=np.float32) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        g = infos["__common__"]
        print(f"round {r:2d}: winner={g['winner']:8} bid=${g['winning_bid']:6.2f} "
              f"P(cartel)={g['posterior_collusion']:.3f} alarm={bool(g['alarm'])}")
        if all(terms.values()):
            break