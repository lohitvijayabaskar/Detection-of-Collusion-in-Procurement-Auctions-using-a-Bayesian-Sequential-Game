import os
import sys

# Add the project root to the python path so it can find the src/ package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from train_emergent_rl import (CHECKPOINT_DIR, RESERVE_PRICE,
                                       ROUNDS_PER_EPISODE, TOTAL_FIRMS,
                                       env_creator)
from src.emergent_procurement_env import (ProcurementEmergentEnv,
                                          calibrate_emergent_likelihoods,
                                          calibrate_emergent_thresholds)

RESULTS_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "results", "real_evaluation_metrics.csv"
    )
)
OUTPUT_PLOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "real_metrics_plot.png")
)


def evaluate_and_plot_real_metrics(num_episodes=5):
    ray.init(ignore_reinit_error=True)

    env_name = "emergent_procurement_env"
    register_env(env_name, env_creator)

    print("Re-calibrating regulator (same procedure as training, independent seed)...")
    l0, l1 = calibrate_emergent_likelihoods(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=150,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        seed0=99,
    )
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0,
        l1,
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=100,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        seed0=20_099,
    )

    env_config = {
        "likelihood_h0": l0,
        "likelihood_h1": l1,
        "cusum_threshold": cusum_h,
        "sr_threshold": sr_h,
    }

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
        .env_runners(num_env_runners=1, explore=True)
        .resources(num_gpus=0)
    )
    algo = config.build_algo()

    print(f"\n---> Loading trained model from {CHECKPOINT_DIR}...\n")
    try:
        algo.restore(CHECKPOINT_DIR)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure you have trained the model first!")
        sys.exit(1)

    env = ProcurementEmergentEnv(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        likelihood_h0=l0,
        likelihood_h1=l1,
        cusum_threshold=cusum_h,
        sr_threshold=sr_h,
    )

    all_data = []

    print("=" * 90)
    print(" GATHERING AUTHENTIC NON-DOCTORED METRICS FROM PPO AGENTS")
    print("=" * 90)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        for r in range(1, ROUNDS_PER_EPISODE + 1):
            actions = {
                a: algo.compute_single_action(
                    observation=obs[a], policy_id="shared_policy", explore=True
                )
                for a in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)
            g = infos.pop("__common__")

            winner = g["winner"]
            winner_info = infos[winner]

            # Calculate metrics
            winner_markup = winner_info["markup"]
            winner_cost = winner_info["bid"] / winner_markup if winner_markup > 0 else 0
            raw_profit = winner_info["bid"] - winner_cost
            alarm_triggered = 1 if g["alarm"] or g.get("alarm_rotation", False) else 0

            # Calculate total fines levied this round
            # Reward = Raw Profit - Fine. So Fine = Raw Profit - Reward (for winner)
            # For losers, Reward = -Fine. So Fine = -Reward.
            total_fines = 0
            for agent, info in infos.items():
                if agent == winner:
                    firm_cost = info["bid"] / info["markup"]
                    firm_raw_profit = info["bid"] - firm_cost
                    firm_fine = firm_raw_profit - info["reward"]
                else:
                    firm_fine = -info["reward"]
                total_fines += firm_fine

            all_data.append(
                {
                    "episode": ep + 1,
                    "round": r,
                    "global_round": ep * ROUNDS_PER_EPISODE + r,
                    "winner_markup": winner_markup,
                    "alarm_triggered": alarm_triggered,
                    "winner_raw_profit": raw_profit,
                    "total_fines": total_fines,
                }
            )

            if all(terms.values()):
                break

    ray.shutdown()

    # Save CSV
    df = pd.DataFrame(all_data)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved raw authentic metrics to {RESULTS_CSV}")

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: RL Overcharge Rate (Markup)
    axs[0].plot(
        df["global_round"],
        df["winner_markup"],
        color="purple",
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    axs[0].axhline(
        y=1.01, color="r", linestyle="--", label="Competitive Baseline (1.01x)"
    )
    axs[0].set_title(
        "Real RL Overcharge Rate (Winning Markup Over Time)", fontsize=14, weight="bold"
    )
    axs[0].set_ylabel("Winning Markup Multiplier")
    axs[0].set_xlabel("Auction Round")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Regulator Detection Rate
    rolling_alarm = df["alarm_triggered"].rolling(window=5, min_periods=1).mean() * 100
    axs[1].plot(
        df["global_round"],
        rolling_alarm,
        color="red",
        linewidth=2,
        label="Detection Rate (%)",
    )
    axs[1].fill_between(df["global_round"], 0, rolling_alarm, color="red", alpha=0.1)
    axs[1].set_title(
        "Real Regulator Detection Rate (Rolling Avg)", fontsize=14, weight="bold"
    )
    axs[1].set_ylabel("Detection Rate (%)")
    axs[1].set_xlabel("Auction Round")
    axs[1].set_ylim(-5, 105)
    axs[1].grid(True, alpha=0.3)

    # Plot 3: True Economic Impact (Profit vs Fines)
    axs[2].plot(
        df["global_round"],
        df["winner_raw_profit"],
        color="green",
        label="Cartel Winner Raw Profit",
        linewidth=2,
    )
    axs[2].plot(
        df["global_round"],
        df["total_fines"],
        color="red",
        label="Total Fines Levied by Regulator",
        linewidth=2,
        linestyle="--",
    )
    axs[2].set_title(
        "True Economic Impact (Cartel Profit vs. Regulatory Fines)",
        fontsize=14,
        weight="bold",
    )
    axs[2].set_ylabel("Dollar Amount ($)")
    axs[2].set_xlabel("Auction Round")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved real metrics plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    evaluate_and_plot_real_metrics()
