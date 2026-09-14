"""
emergent_procurement_env.py
====================================================================
Closed-loop, emergent-collusion procurement environment with discrete strategies.
"""

from __future__ import annotations

import random
from typing import Dict as TypingDict, Optional, Tuple, Set, List

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv
from collections import Counter

from regulator import (
    BayesianSequentialRegulator,
    RegulatorConfig,
    KDELikelihood,
    compute_screens,
    suspect_scores,
)

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

    def process_admissions(self, applicants: List[str], costs: TypingDict[str, float], current_round: int):
        for applicant in applicants:
            if applicant in self.banned_until and self.banned_until[applicant] >= current_round:
                continue # Still banned
                
            if self.admission_policy == 0:
                self.members.add(applicant)
            elif self.admission_policy == 1:
                # Selective: Only admit if cost is lower than average market cost
                avg_cost = sum(costs.values()) / len(costs) if costs else 0
                if costs.get(applicant, float('inf')) < avg_cost:
                    self.members.add(applicant)

    def select_designated_winner(self, costs: TypingDict[str, float]):
        if not self.members:
            self.designated_winner = None
            return None
        # Designated winner is the member with the lowest cost
        valid_members = [m for m in self.members if m in costs]
        if valid_members:
            self.designated_winner = min(valid_members, key=lambda m: costs[m])
        else:
            self.designated_winner = None
        return self.designated_winner

    def check_defectors_and_punish(self, bids: TypingDict[str, float], current_round: int):
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
                self.banned_until[defector] = float('inf')
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

        # Actions: [Bidding Strategy (0-5), Admission Vote (0-1), Punishment Vote (0-1)]
        # Bidding Strategies:
        # 0: Honest Bertrand (cost * ~1.02)
        # 1: Honest Cournot Low (cost * ~1.10)
        # 2: Honest Cournot High (cost * ~1.20)
        # 3: Honest Random (cost * ~1.25, noisy)
        # 4: Dishonest Target Price (bid near reserve)
        # 5: Dishonest Cover Bid (yield to designated winner, else bid high)
        self._action_spaces = {
            agent: MultiDiscrete([6, 2, 2])
            for agent in self.possible_agents
        }
        
        # Obs: [private_cost, cartel_member_flag, prev_winning_bid, regulator_posterior,
        #       regulator_alarm, round_progress]
        self._observation_spaces = {
            agent: Box(low=0.0, high=float(self.reserve_price), shape=(6,), dtype=np.float32)
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
        applicants = [a for a in self.agents if actions[a][0] in [4, 5] and a not in self.cartel.members]
        self.cartel.process_admissions(applicants, self.private_costs, self.round_num)
        
        # 3. Cartel phase: Identify designated winner
        designated_winner = self.cartel.select_designated_winner(self.private_costs)
        
        # 4. Bidding phase
        bids = {}
        for agent, act in actions.items():
            strategy = act[0]
            cost = self.private_costs[agent]
            
            if strategy == 0:
                # Honest Bertrand
                markup = random.uniform(1.0, 1.05)
            elif strategy == 1:
                # Honest Cournot Low
                markup = random.uniform(1.05, 1.15)
            elif strategy == 2:
                # Honest Cournot High
                markup = random.uniform(1.15, 1.25)
            elif strategy == 3:
                # Honest Random
                markup = random.uniform(1.0, 1.5)
            elif strategy == 4:
                # Dishonest Target Price
                markup = (self.reserve_price / cost) * random.uniform(0.9, 1.0)
            elif strategy == 5:
                # Dishonest Cover Bid
                if agent == designated_winner:
                    # Win near cost or slightly inflated
                    markup = random.uniform(1.0, 1.1)
                else:
                    # Cover bid near reserve
                    markup = (self.reserve_price / cost) * random.uniform(0.95, 1.0)
            else:
                markup = 1.0
                
            raw_bid = cost * markup
            bids[agent] = round(max(cost * 1.00, min(raw_bid, self.reserve_price)), 2)

        valid_bids = {k: v for k, v in bids.items() if v <= self.reserve_price}
        winner = min(valid_bids, key=valid_bids.get)
        winning_bid = valid_bids[winner]
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
        if self.regulator is not None:
            result = self.regulator.update(bids)
            posterior = result["posterior_collusion"]
            alarm = 1.0 if (result["alarm_cusum"] or result["alarm_sr"]) else 0.0
            susp = result["suspects"]

            if alarm:
                for agent in self.agents:
                    suspicion = float(np.exp(susp.get(agent, -50.0)))
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
                "strategy": actions[agent][0],
            }
        infos["__common__"] = {
            "winner": winner,
            "winning_bid": winning_bid,
            "posterior_collusion": posterior,
            "alarm": alarm,
            "cartel_size": len(self.cartel.members),
        }

        terminations = {a: (self.round_num >= self.rounds_per_episode) for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = self._get_obs()

        if self.round_num >= self.rounds_per_episode:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

# =====================================================================
# 3. Calibration reference bidders
# =====================================================================
def _competitive_calibration_bidder(cost: float, reserve_price: float) -> float:
    markup = random.uniform(1.00, 1.02)
    return round(min(cost * markup, reserve_price), 2)

def _collusive_calibration_bidder(cost: float, reserve_price: float) -> float:
    markup = random.uniform(0.90, 0.98) * (reserve_price / cost)
    return round(min(cost * markup, reserve_price), 2)

def _uniform_markup_calibration_bidder(cost: float, reserve_price: float) -> float:
    markup = _UNIFORM_MARKUP_SEVERITY[0]
    return round(min(cost * markup, reserve_price), 2)

_UNIFORM_MARKUP_SEVERITY = [1.2]

def _complementary_calibration_bidder(cost: float, reserve_price: float, is_designated: bool) -> float:
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
    samples = {0: {"cv": [], "norm_diff": [], "skew": [], "level": []},
               1: {"cv": [], "norm_diff": [], "skew": [], "level": []}}

    rng0 = random.Random(seed0)
    for ep in range(n_episodes):
        for _ in range(rounds_per_episode):
            costs = {f"f{i}": rng0.uniform(50.0, 60.0) for i in range(total_firms)}
            bids = {k: _competitive_calibration_bidder(c, reserve_price) for k, c in costs.items()}
            screens = compute_screens(bids, reserve_price=reserve_price)
            for k, v in screens.items():
                samples[0][k].append(v)

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
        actions = {a: np.array([random.randint(0, 5), random.randint(0, 1), random.randint(0, 1)]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        g = infos["__common__"]
        print(f"round {r:2d}: winner={g['winner']:8} bid=${g['winning_bid']:6.2f} "
              f"P(cartel)={g['posterior_collusion']:.3f} alarm={bool(g['alarm'])} cartel_size={g['cartel_size']}")
        if all(terms.values()):
            break