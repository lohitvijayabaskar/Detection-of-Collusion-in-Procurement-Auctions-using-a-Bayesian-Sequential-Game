"""
emergent_procurement_env.py
====================================================================
Closed-loop, emergent-collusion procurement environment with discrete strategies.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict as TypingDict
from typing import List, Optional, Set, Tuple

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv

from regulator import (BayesianSequentialRegulator, KDELikelihood,
                       RegulatorConfig, compute_screens, suspect_scores)

# =====================================================================
# 1. Cartel and Firm Classes
# =====================================================================


class Cartel:
    """
    Superclass (conceptually) that manages the shared cartel mechanics.
    Firms interact with this to form their decisions and execute punishments/admissions.
    """

    def __init__(self):
        self.members: Set[str] = set()
        self.banned_until: TypingDict[str, int] = {}

        # Policies:
        # Admission: 0 = Open (Admit all applicants), 1 = Selective (Cost < Avg)
        self.admission_policy = 0
        # Punishment: 0 = Grim Trigger (Ban forever), 1 = Tit-for-Tat (Ban 1 round)
        self.punishment_policy = 0

        self.designated_winner: Optional[str] = None

    def process_votes(self, votes: TypingDict[str, Tuple[int, int]]):
        """
        votes: dict of agent_id -> (admission_vote, punishment_vote)
        Only current members get to vote.
        """
        if not self.members:
            # If no members, the collective votes of everyone might form the first cartel
            active_voters = list(votes.keys())
        else:
            active_voters = list(self.members)

        if active_voters:
            ad_votes = [votes[a][0] for a in active_voters if a in votes]
            pun_votes = [votes[a][1] for a in active_voters if a in votes]

            if ad_votes:
                self.admission_policy = Counter(ad_votes).most_common(1)[0][0]
            if pun_votes:
                self.punishment_policy = Counter(pun_votes).most_common(1)[0][0]

    def process_admissions(
        self, applicants: List[str], costs: TypingDict[str, float], current_round: int
    ):
        for applicant in applicants:
            if (
                applicant in self.banned_until
                and self.banned_until[applicant] >= current_round
            ):
                continue  # Still banned

            if self.admission_policy == 0:
                self.members.add(applicant)
            elif self.admission_policy == 1:
                # Selective: Only admit if cost is lower than average market cost
                avg_cost = sum(costs.values()) / len(costs) if costs else 0
                if costs.get(applicant, float("inf")) < avg_cost:
                    self.members.add(applicant)

    def select_designated_winner(self, costs: TypingDict[str, float]):
        if not self.members:
            self.designated_winner = None
            self.rotation_queue = []
            return None

        valid_members = sorted([m for m in self.members if m in costs])
        if not valid_members:
            self.designated_winner = None
            return None

        if not hasattr(self, "rotation_queue"):
            self.rotation_queue = []

        # Rebuild queue if members changed
        if set(self.rotation_queue) != set(valid_members):
            self.rotation_queue = valid_members[:]

        # Round-robin selection
        self.designated_winner = self.rotation_queue.pop(0)
        self.rotation_queue.append(self.designated_winner)

        return self.designated_winner

    def check_defectors_and_punish(
        self, bids: TypingDict[str, float], current_round: int
    ):
        if not self.designated_winner or self.designated_winner not in bids:
            return

        designated_bid = bids[self.designated_winner]
        defectors = []

        # A defector is any member who bid lower than the designated winner,
        # intentionally undercutting them.
        for member in list(self.members):
            if member != self.designated_winner and member in bids:
                if bids[member] < designated_bid:
                    defectors.append(member)

        for defector in defectors:
            self.members.remove(defector)
            if self.punishment_policy == 0:
                # Grim Trigger: Ban forever
                self.banned_until[defector] = float("inf")
            elif self.punishment_policy == 1:
                # Tit-for-Tat: Ban for 1 round
                self.banned_until[defector] = current_round + 1


# =====================================================================
# 2. Emergent-collusion environment
# =====================================================================
class ProcurementEmergentEnv(ParallelEnv):
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

        # Actions: [Bidding Strategy (0-6), Admission Vote (0-1), Punishment Vote (0-1)]
        # Overhauled Dynamic Economic Bidding Strategies:
        # 0: Honest Bertrand (Max price initially, then competitive markdown from previous clearing price)
        # 1: Adaptive Best-Response (Empirical surplus maximization based on winning history)
        # 2: Dynamic Margin Tracking (Market-trend adaptive margin expansion/compression)
        # 3: Strategic Adaptive Undercutting (Dynamic price shading proportional to cost gap)
        # 4: Cartel Evasive Target Pricing (Market-anchored collusive rent seeking)
        # 5: Cartel Camouflaged Cover Bidding (Anti-trust evasive cover bids trailing designated winner)
        # 6: Tacit Collusion / Focal Point Bidding (Running median price anchoring)
        self._action_spaces = {
            agent: MultiDiscrete([7, 2, 2]) for agent in self.possible_agents
        }

        # Obs: [private_cost, cartel_member_flag, prev_winning_bid, regulator_posterior,
        #       regulator_alarm, round_progress]
        self._observation_spaces = {
            agent: Box(
                low=0.0, high=float(self.reserve_price), shape=(6,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.likelihood_h0 = likelihood_h0
        self.likelihood_h1 = likelihood_h1
        self.regulator_config = RegulatorConfig(
            prior_h1=prior_h1,
            cusum_threshold=cusum_threshold,
            sr_threshold=sr_threshold,
            eng_estimate_default=reserve_price,
        )

        self.private_costs = {}
        self.prev_winning_bid = self.reserve_price
        self.winning_bid_history: List[float] = []
        self.round_num = 0
        self.regulator: Optional[BayesianSequentialRegulator] = None
        self._last_posterior = prior_h1
        self._last_alarm = 0.0

        self.cartel = Cartel()

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def set_likelihoods(self, l0: KDELikelihood, l1: KDELikelihood) -> None:
        self.likelihood_h0 = l0
        self.likelihood_h1 = l1

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.round_num = 0
        self.prev_winning_bid = self.reserve_price
        self.winning_bid_history = []
        self.private_costs = {
            agent: round(random.uniform(50.0, 60.0), 2) for agent in self.agents
        }

        if self.likelihood_h0 is not None and self.likelihood_h1 is not None:
            self.regulator = BayesianSequentialRegulator(
                self.likelihood_h0, self.likelihood_h1, self.regulator_config
            )
        else:
            self.regulator = None

        self._last_posterior = self.regulator_config.prior_h1
        self._last_alarm = 0.0

        self.cartel = Cartel()

        return self._get_obs(), {a: {} for a in self.agents}

    def _get_obs(self):
        round_progress = self.round_num / self.rounds_per_episode
        obs = {}
        for agent in self.agents:
            is_member = 1.0 if agent in self.cartel.members else 0.0
            obs[agent] = np.array(
                [
                    self.private_costs[agent],
                    is_member * self.reserve_price,
                    self.prev_winning_bid,
                    self._last_posterior * self.reserve_price,
                    self._last_alarm * self.reserve_price,
                    round_progress * self.reserve_price,
                ],
                dtype=np.float32,
            )
        return obs

    def step(self, actions: TypingDict[str, np.ndarray]):
        self.round_num += 1

        # 1. Cartel phase: Process votes
        votes = {agent: (act[1], act[2]) for agent, act in actions.items()}
        self.cartel.process_votes(votes)

        # 2. Cartel phase: Admissions
        # We consider any firm that chose a Dishonest strategy (4 or 5) as an applicant
        applicants = [
            a
            for a in self.agents
            if actions[a][0] in [4, 5] and a not in self.cartel.members
        ]
        self.cartel.process_admissions(applicants, self.private_costs, self.round_num)

        # 3. Cartel phase: Identify designated winner
        designated_winner = self.cartel.select_designated_winner(self.private_costs)

        # Calculate history summary metrics for dynamic strategy calculations
        if self.winning_bid_history:
            recent_winning_bids = self.winning_bid_history[-5:]
            avg_winning_bid = float(np.mean(recent_winning_bids))
            std_winning_bid = max(
                1.0,
                (
                    float(np.std(recent_winning_bids))
                    if len(recent_winning_bids) > 1
                    else 2.5
                ),
            )
        else:
            avg_winning_bid = self.reserve_price * 0.95  # ~95.0 initial baseline
            std_winning_bid = 2.5

        # Pre-calculate designated winner's cartel target bid if cartel exists
        designated_target_bid = None
        if designated_winner and designated_winner in actions:
            des_cost = self.private_costs[designated_winner]
            collusive_target = min(
                self.reserve_price * 0.90, max(avg_winning_bid * 1.03, des_cost * 1.25)
            )
            designated_target_bid = round(
                max(
                    des_cost * 1.02,
                    min(
                        collusive_target + random.uniform(-0.5, 0.5), self.reserve_price
                    ),
                ),
                2,
            )

        # 4. Bidding phase
        bids = {}
        for agent, act in actions.items():
            strategy = act[0]
            cost = self.private_costs[agent]

            if strategy == 0:
                # Honest Bertrand (Max price in Round 1, then competitive markdown from previous winning bid)
                if not self.winning_bid_history:
                    raw_bid = self.reserve_price
                else:
                    markdown_pct = random.uniform(0.01, 0.03)
                    target_bid = self.prev_winning_bid * (1.0 - markdown_pct)
                    raw_bid = max(cost * 1.01, target_bid)
            elif strategy == 1:
                # Adaptive Best-Response (Empirical Surplus Optimization)
                if avg_winning_bid > cost:
                    lambda_param = random.uniform(0.40, 0.60)
                    target_surplus = lambda_param * (avg_winning_bid - cost)
                    raw_bid = cost + target_surplus + random.uniform(-0.5, 0.5)
                else:
                    raw_bid = cost * random.uniform(1.02, 1.05)
            elif strategy == 2:
                # Dynamic Margin Tracking (Market Trend Shading)
                price_trend = avg_winning_bid / self.reserve_price
                markup = 1.0 + (0.15 * price_trend) + random.uniform(-0.01, 0.01)
                raw_bid = cost * max(1.015, markup)
            elif strategy == 3:
                # Strategic Adaptive Undercutting (Dynamic Price Shading)
                gap = self.prev_winning_bid - cost
                if gap > 1.0:
                    undercut_pct = 0.01 + 0.02 * min(
                        1.0, gap / max(1.0, self.reserve_price - cost)
                    )
                    target_bid = self.prev_winning_bid * (1.0 - undercut_pct)
                    raw_bid = max(cost * 1.01, target_bid) + random.uniform(-0.2, 0.2)
                else:
                    raw_bid = cost * random.uniform(1.01, 1.03)
            elif strategy == 4:
                # Cartel Evasive Target Pricing (Market-Anchored Monopoly Rent)
                collusive_target = min(
                    self.reserve_price * 0.90, max(avg_winning_bid * 1.03, cost * 1.25)
                )
                raw_bid = collusive_target + random.uniform(-0.75, 0.75)
            elif strategy == 5:
                # Cartel Camouflaged Cover Bidding (Anti-Trust Evasive Cover Bids)
                if agent == designated_winner:
                    if designated_target_bid is not None:
                        raw_bid = designated_target_bid
                    else:
                        collusive_target = min(
                            self.reserve_price * 0.90,
                            max(avg_winning_bid * 1.03, cost * 1.25),
                        )
                        raw_bid = collusive_target + random.uniform(-0.5, 0.5)
                else:
                    # Cover bidder: bid slightly above designated winner's bid with realistic noise
                    # so that it trails the winner closely without triggering obvious level/variance alarms
                    base_target = (
                        designated_target_bid
                        if designated_target_bid is not None
                        else avg_winning_bid
                    )
                    cover_offset = random.uniform(1.5, 4.0) + max(
                        0.0, random.gauss(0, std_winning_bid * 0.5)
                    )
                    raw_bid = base_target + cover_offset
            elif strategy == 6:
                # Tacit Collusion / Focal Point Bidding (Running Median Anchoring)
                focal_price = avg_winning_bid
                firm_offset = random.uniform(0.0, 0.04)
                raw_bid = max(
                    cost * 1.02,
                    focal_price * (1.0 + firm_offset) + random.uniform(-0.3, 0.3),
                )
            else:
                raw_bid = cost * 1.02

            bids[agent] = round(max(cost * 1.01, min(raw_bid, self.reserve_price)), 2)

        valid_bids = {k: v for k, v in bids.items() if v <= self.reserve_price}
        winner = min(valid_bids, key=valid_bids.get)
        winning_bid = valid_bids[winner]
        self.winning_bid_history.append(winning_bid)
        self.prev_winning_bid = winning_bid

        # 5. Defector Check & Punishment
        self.cartel.check_defectors_and_punish(bids, self.round_num)

        # 6. Rewards
        rewards = {agent: 0.0 for agent in self.agents}
        for agent in self.agents:
            if agent == winner:
                rewards[agent] = winning_bid - self.private_costs[agent]

        # 7. Regulator
        posterior, alarm, susp = self._last_posterior, 0.0, {}
        firm_posteriors = {}
        alarm_rotation = False
        entropy = 0.0

        # Government engineering estimate: true lowest cost + up to 5% noise
        true_cost_min = min(self.private_costs.values())
        engineering_estimate = round(true_cost_min * random.uniform(0.95, 1.05), 2)

        if self.regulator is not None:
            result = self.regulator.update(bids, eng_estimate=engineering_estimate)
            posterior = result["posterior_collusion"]
            alarm = 1.0 if (result["alarm_cusum"] or result["alarm_sr"]) else 0.0
            alarm_rotation = result.get("alarm_rotation", False)
            entropy = result.get("entropy", 0.0)
            susp = result["suspects"]
            firm_posteriors = result.get("firm_posteriors", {})

            # Surgical fines based on individual firm posterior tracking (Cover Bidding)
            for agent in self.agents:
                firm_post = firm_posteriors.get(agent, self.regulator_config.prior_h1)
                if firm_post > 0.80:
                    fine = self.fine_rate * firm_post * self.reserve_price * 0.05
                    rewards[agent] -= fine

            # Conspiracy fines for Bid Rotation Cartels
            if alarm_rotation:
                for agent in set(self.regulator.winner_history):
                    if agent in rewards:
                        rewards[agent] -= self.fine_rate * self.reserve_price * 0.05

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
                "firm_posterior": firm_posteriors.get(agent, 0.10),
                "strategy": actions[agent][0],
            }
        infos["__common__"] = {
            "winner": winner,
            "winning_bid": winning_bid,
            "posterior_collusion": posterior,
            "alarm": alarm,
            "alarm_rotation": alarm_rotation,
            "entropy": entropy,
            "cartel_size": len(self.cartel.members),
            "engineering_estimate": engineering_estimate,
        }

        terminations = {
            a: (self.round_num >= self.rounds_per_episode) for a in self.agents
        }
        truncations = {a: False for a in self.agents}
        obs = self._get_obs()

        if self.round_num >= self.rounds_per_episode:
            self.agents = []

        return obs, rewards, terminations, truncations, infos


# =====================================================================
# 3. Calibration reference bidders
# =====================================================================
def _competitive_calibration_bidder(
    cost: float, reserve_price: float, prev_winning_bid: float = 95.0
) -> float:
    # Simulates competitive reference bidders using history-aware strategies 0, 1, 2, 3
    strat = random.choices([0, 1, 2, 3], weights=[0.50, 0.25, 0.15, 0.10])[0]
    avg_w = prev_winning_bid
    if strat == 0:
        if avg_w >= reserve_price or avg_w >= 95.0:
            raw = reserve_price
        else:
            markdown_pct = random.uniform(0.01, 0.03)
            raw = max(cost * 1.01, avg_w * (1.0 - markdown_pct))
    elif strat == 1:
        if avg_w > cost:
            raw = (
                cost
                + random.uniform(0.40, 0.60) * (avg_w - cost)
                + random.uniform(-0.5, 0.5)
            )
        else:
            raw = cost * random.uniform(1.02, 1.05)
    elif strat == 2:
        price_trend = avg_w / reserve_price
        raw = cost * max(
            1.015, 1.0 + (0.15 * price_trend) + random.uniform(-0.01, 0.01)
        )
    else:
        gap = avg_w - cost
        if gap > 1.0:
            target = avg_w * (
                1.0 - (0.01 + 0.02 * min(1.0, gap / max(1.0, reserve_price - cost)))
            )
            raw = max(cost * 1.01, target) + random.uniform(-0.2, 0.2)
        else:
            raw = cost * random.uniform(1.01, 1.03)
    return round(max(cost * 1.01, min(raw, reserve_price)), 2)


def _collusive_calibration_bidder(
    cost: float, reserve_price: float, prev_winning_bid: float = 70.0
) -> float:
    collusive_target = min(
        reserve_price * 0.90, max(prev_winning_bid * 1.03, cost * 1.25)
    )
    raw = collusive_target + random.uniform(-0.75, 0.75)
    return round(max(cost * 1.01, min(raw, reserve_price)), 2)


def _uniform_markup_calibration_bidder(cost: float, reserve_price: float) -> float:
    markup = _UNIFORM_MARKUP_SEVERITY[0]
    return round(min(cost * markup, reserve_price), 2)


_UNIFORM_MARKUP_SEVERITY = [1.2]


def _complementary_calibration_bidder(
    cost: float, reserve_price: float, is_designated: bool, des_bid: float = 75.0
) -> float:
    if is_designated:
        collusive_target = min(reserve_price * 0.90, max(des_bid * 1.03, cost * 1.25))
        raw = collusive_target + random.uniform(-0.5, 0.5)
    else:
        cover_offset = random.uniform(1.5, 4.0) + max(0.0, random.gauss(0, 1.5))
        raw = des_bid + cover_offset
    return round(max(cost * 1.01, min(raw, reserve_price)), 2)


def calibrate_emergent_likelihoods(
    reserve_price: float = 100.0,
    total_firms: int = 20,
    n_episodes: int = 300,
    rounds_per_episode: int = 20,
    seed0: int = 0,
) -> Tuple[KDELikelihood, KDELikelihood]:
    samples = {
        0: {"cv": [], "norm_diff": [], "skew": [], "level": []},
        1: {"cv": [], "norm_diff": [], "skew": [], "level": []},
    }

    rng0 = random.Random(seed0)
    for ep in range(n_episodes):
        prev_win = 60.0
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng0.uniform(50.0, 60.0) for i in range(total_firms)}
            bids = {
                k: _competitive_calibration_bidder(c, reserve_price, prev_win)
                for k, c in costs.items()
            }
            prev_win = min(bids.values())

            true_cost_min = min(costs.values())
            eng_estimate = round(true_cost_min * rng0.uniform(0.95, 1.05), 2)

            screens = compute_screens(bids, eng_estimate=eng_estimate)
            for k, v in screens.items():
                samples[0][k].append(v)

    rng1 = random.Random(seed0 + 1)
    for ep in range(n_episodes):
        archetype = rng1.choice(["target_price", "uniform_markup", "complementary"])
        _UNIFORM_MARKUP_SEVERITY[0] = rng1.uniform(1.10, 1.60)
        designated = "f0"
        prev_win = 75.0
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng1.uniform(50.0, 60.0) for i in range(total_firms)}
            if archetype == "target_price":
                bids = {
                    k: _collusive_calibration_bidder(c, reserve_price, prev_win)
                    for k, c in costs.items()
                }
            elif archetype == "uniform_markup":
                bids = {
                    k: _uniform_markup_calibration_bidder(c, reserve_price)
                    for k, c in costs.items()
                }
            else:
                des_cost = costs[designated]
                des_bid = _collusive_calibration_bidder(
                    des_cost, reserve_price, prev_win
                )
                bids = {
                    k: _complementary_calibration_bidder(
                        c, reserve_price, k == designated, des_bid
                    )
                    for k, c in costs.items()
                }
            prev_win = min(bids.values())

            true_cost_min = min(costs.values())
            eng_estimate = round(true_cost_min * rng1.uniform(0.95, 1.05), 2)

            screens = compute_screens(bids, eng_estimate=eng_estimate)
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
    max_cusum, max_log_sr = [], []
    for ep in range(n_episodes):
        rng = random.Random(seed0 + ep)
        reg = BayesianSequentialRegulator(
            l0, l1, RegulatorConfig(cusum_threshold=np.inf, sr_threshold=np.inf)
        )
        prev_win = 60.0
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng.uniform(50.0, 60.0) for i in range(total_firms)}
            bids = {
                k: _competitive_calibration_bidder(c, reserve_price, prev_win)
                for k, c in costs.items()
            }
            prev_win = min(bids.values())

            true_cost_min = min(costs.values())
            eng_estimate = round(true_cost_min * rng.uniform(0.95, 1.05), 2)

            reg.update(bids, eng_estimate=eng_estimate)
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
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0, l1, n_episodes=100, rounds_per_episode=20
    )
    print(f"  -> CUSUM threshold h = {cusum_h:.3f}")
    print(f"  -> Shiryaev-Roberts threshold A = {sr_h:.3f}")

    env = ProcurementEmergentEnv(
        likelihood_h0=l0,
        likelihood_h1=l1,
        cusum_threshold=cusum_h,
        sr_threshold=sr_h,
        rounds_per_episode=20,
    )
    obs, _ = env.reset(seed=42)

    print("\nRunning a random-policy smoke test episode...\n")
    for r in range(1, 21):
        actions = {
            a: np.array(
                [random.randint(0, 5), random.randint(0, 1), random.randint(0, 1)]
            )
            for a in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        g = infos["__common__"]
        print(
            f"round {r:2d}: winner={g['winner']:8} bid=${g['winning_bid']:6.2f} "
            f"P(cartel)={g['posterior_collusion']:.3f} alarm={bool(g['alarm'])} cartel_size={g['cartel_size']}"
        )
        if all(terms.values()):
            break
