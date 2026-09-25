"""
train_emergent_rl.py
====================================================================
Trains the shared PPO policy on ProcurementEmergentEnv and, critically
for the reframed contribution, logs the CO-ADAPTATION trajectory: at
regular checkpoints during training we freeze the current policy,
roll out a handful of evaluation episodes with exploration off, and
record both sides of the game:

  - collusion_index: mean markup firms charge over cost
  - strategy_dist: distribution of strategies chosen
  - regulator_alarm_rate / mean_posterior: how often, and how
    confidently, the *regulator* is catching behaviour

Plotting collusion_index against alarm_rate across training iterations
is the equilibrium-analysis artifact the AAMAS reviewer feedback asked
for: it shows whether firms (a) never learn to collude because
individual profit maximization plus detection risk isn't worth it,
(b) learn to collude and get suppressed by the regulator (equilibrium
markup tracks just under the detection boundary), or (c) learn to
collude faster than the regulator's fixed calibration can adapt.
Option (c) would itself be a finding -- it would motivate making the
regulator's thresholds adaptive/RL-trained too, which is a natural
follow-up but out of scope for this pass.
"""

import csv
import os

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from src.emergent_procurement_env import (ProcurementEmergentEnv,
                                          calibrate_emergent_likelihoods,
                                          calibrate_emergent_thresholds)

ROUNDS_PER_EPISODE = 20
TOTAL_FIRMS = 20
RESERVE_PRICE = 100.0
NUM_ITERATIONS = 100
CHECKPOINT_DIR = os.path.abspath("./procurement_model_checkpoint")
COADAPTATION_LOG = os.path.abspath("./results/coadaptation_log.csv")


def env_creator(config):
    env = ProcurementEmergentEnv(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        likelihood_h0=config.get("likelihood_h0"),
        likelihood_h1=config.get("likelihood_h1"),
        cusum_threshold=config.get("cusum_threshold", 8.0),
        sr_threshold=config.get("sr_threshold", 50.0),
        fine_rate=config.get("fine_rate", 1.5),
    )
    return ParallelPettingZooEnv(env)


def evaluate_coadaptation(algo, l0, l1, cusum_h, sr_h, n_episodes: int = 5) -> dict:
    """Freeze the current policy and roll out a few episodes to measure
    where the firm-side / regulator-side equilibrium currently sits."""
    env = ProcurementEmergentEnv(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        likelihood_h0=l0,
        likelihood_h1=l1,
        cusum_threshold=cusum_h,
        sr_threshold=sr_h,
    )

    markups, posteriors, alarms, profits, strategies = [], [], [], [], []
    cartel_sizes, buyer_costs, allocative_efficiencies = [], [], []
    tp, fp, tn, fn = 0, 0, 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            actions = {
                a: algo.compute_single_action(
                    observation=obs[a], policy_id="shared_policy", explore=False
                )
                for a in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)

            # Track agent-level metrics
            costs = {}
            for a, info in infos.items():
                if a == "__common__":
                    continue
                markups.append(info["markup"])
                strategies.append(info["strategy"])
                costs[a] = info["cost"]

            # Track global/economic metrics
            g = infos["__common__"]
            posteriors.append(g["posterior_collusion"])
            alarms.append(g["alarm"])
            profits.append(sum(rewards.values()))
            cartel_sizes.append(g["cartel_size"])
            buyer_costs.append(g["winning_bid"])

            # Allocative Efficiency: Did the firm with the absolute lowest cost win?
            min_cost = min(costs.values())
            winner_cost = costs[g["winner"]]
            allocative_efficiencies.append(1 if winner_cost == min_cost else 0)

            # Confusion matrix tracking
            is_cartel_active = g["cartel_size"] > 1
            if g["alarm"] and is_cartel_active:
                tp += 1
            elif g["alarm"] and not is_cartel_active:
                fp += 1
            elif not g["alarm"] and not is_cartel_active:
                tn += 1
            elif not g["alarm"] and is_cartel_active:
                fn += 1

            done = all(terms.values())

    strategy_counts = {i: 0 for i in range(7)}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    total_actions = len(strategies) if strategies else 1
    strategy_dist = {
        f"strat_{k}_pct": (v / total_actions) * 100 for k, v in strategy_counts.items()
    }

    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    res = {
        "collusion_index": float(np.mean(markups)),
        "mean_posterior": float(np.mean(posteriors)),
        "alarm_rate": float(np.mean(alarms)),
        "mean_round_profit": float(np.mean(profits)),
        "avg_cartel_size": float(np.mean(cartel_sizes)),
        "avg_buyer_cost": float(np.mean(buyer_costs)),
        "allocative_efficiency": float(np.mean(allocative_efficiencies)),
        "tpr": tpr,
        "fpr": fpr,
    }
    res.update(strategy_dist)
    return res


def main():
    ray.init(ignore_reinit_error=True)

    print("Calibrating regulator H0/H1 likelihoods from reference bidders...")
    l0, l1 = calibrate_emergent_likelihoods(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=150,
        rounds_per_episode=ROUNDS_PER_EPISODE,
    )
    print("Calibrating alarm thresholds against the competitive reference...")
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0,
        l1,
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=100,
        rounds_per_episode=ROUNDS_PER_EPISODE,
    )
    print(f"  -> CUSUM threshold h = {cusum_h:.3f}, SR threshold A = {sr_h:.3f}")

    env_name = "emergent_procurement_env"
    env_config = {
        "likelihood_h0": l0,
        "likelihood_h1": l1,
        "cusum_threshold": cusum_h,
        "sr_threshold": sr_h,
    }
    register_env(env_name, env_creator)

    temp_env = ProcurementEmergentEnv(likelihood_h0=l0, likelihood_h1=l1)
    obs_space = temp_env.observation_space("firm01")
    act_space = temp_env.action_space("firm01")

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name, env_config=env_config)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(num_env_runners=1, rollout_fragment_length=100)
        .training(train_batch_size=1000)
        .resources(num_gpus=0)
    )

    algo = config.build_algo()

    if os.path.exists(CHECKPOINT_DIR):
        print(f"\n---> Found existing checkpoint at {CHECKPOINT_DIR}. Restoring...")
        try:
            algo.restore(CHECKPOINT_DIR)
        except Exception:
            print("Checkpoint incompatible. Starting fresh...")
    else:
        print("\n---> No existing checkpoint found. Training from scratch...")

    print("\n" + "=" * 60)
    print(f" STARTING TRAINING LOOP ({NUM_ITERATIONS} ITERATIONS, {TOTAL_FIRMS} FIRMS)")
    print("=" * 60)

    with open(COADAPTATION_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "reward_mean",
                "collusion_index",
                "mean_posterior",
                "alarm_rate",
                "tpr",
                "fpr",
                "mean_round_profit",
                "avg_cartel_size",
                "avg_buyer_cost",
                "allocative_efficiency",
                "strat_0_pct",
                "strat_1_pct",
                "strat_2_pct",
                "strat_3_pct",
                "strat_4_pct",
                "strat_5_pct",
                "strat_6_pct",
            ]
        )

        for i in range(1, NUM_ITERATIONS + 1):
            result = algo.train()

            if (
                "env_runners" in result
                and "policy_reward_mean" in result["env_runners"]
            ):
                reward = result["env_runners"]["policy_reward_mean"].get(
                    "shared_policy", 0.0
                )
            elif "policy_reward_mean" in result:
                reward = result["policy_reward_mean"].get("shared_policy", 0.0)
            else:
                reward = 0.0

            if i % 10 == 0 or i == 1:
                co = evaluate_coadaptation(algo, l0, l1, cusum_h, sr_h, n_episodes=5)
                print(
                    f"Iteration {i:03d} | RewardMean={reward:7.2f} | "
                    f"BuyerCost=${co['avg_buyer_cost']:.2f} | AllocEff={co['allocative_efficiency']:.2f} | "
                    f"Alarm={co['alarm_rate']:.3f} (TPR:{co['tpr']:.2f}, FPR:{co['fpr']:.2f}) | "
                    f"Strats: 0:{co['strat_0_pct']:.0f}% 4:{co['strat_4_pct']:.0f}% 5:{co['strat_5_pct']:.0f}% 6:{co['strat_6_pct']:.0f}%"
                )
                writer.writerow(
                    [
                        i,
                        reward,
                        co["collusion_index"],
                        co["mean_posterior"],
                        co["alarm_rate"],
                        co["tpr"],
                        co["fpr"],
                        co["mean_round_profit"],
                        co["avg_cartel_size"],
                        co["avg_buyer_cost"],
                        co["allocative_efficiency"],
                        co["strat_0_pct"],
                        co["strat_1_pct"],
                        co["strat_2_pct"],
                        co["strat_3_pct"],
                        co["strat_4_pct"],
                        co["strat_5_pct"],
                        co["strat_6_pct"],
                    ]
                )
                f.flush()

    print(f"\n---> Saving checkpoint to {CHECKPOINT_DIR}...")
    algo.save(CHECKPOINT_DIR)
    print(f"---> Co-adaptation trajectory saved to {COADAPTATION_LOG}")

    ray.shutdown()
    print("Training complete and safely shut down.")


if __name__ == "__main__":
    main()
