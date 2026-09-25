import os
import sys

# Add the project root to the python path so it can find the src/ package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_menu():
    while True:
        print("\n" + "=" * 50)
        print(" PROCUREMENT COLLUSION SIMULATION MENU")
        print("=" * 50)
        print("1. Run Emergent RL Training (Auto-learning firms)")
        print("2. Evaluate Pre-Trained RL Model (No Training)")
        print("3. Run Custom Simulation (Manual Firm Allocation)")
        print("4. Generate Graphs from previous Training Run")
        print("5. Exit")
        choice = input("Select an option (1-5): ")

        if choice == "1":
            import subprocess

            print("\nStarting RL Training (this will take 3-4 minutes)...")
            subprocess.run([sys.executable, "train_emergent_rl.py"])
            print("\nGenerating final metrics graph...")
            subprocess.run([sys.executable, "plot_metrics.py"])
        elif choice == "2":
            import subprocess

            print("\nLoading Pre-Trained RL Model for Evaluation...")
            subprocess.run([sys.executable, "eval_emergent.py"])
        elif choice == "3":
            run_custom_simulation()
        elif choice == "4":
            import subprocess

            subprocess.run([sys.executable, "plot_metrics.py"])
        elif choice == "5":
            sys.exit(0)
        else:
            print("Invalid choice.")


def run_custom_simulation():
    from train_emergent_rl import RESERVE_PRICE, TOTAL_FIRMS

    try:
        rounds = int(input("\nEnter number of rounds to simulate (e.g., 20): "))
        cartel_count = int(input(f"Enter number of cartel firms (0-{TOTAL_FIRMS}): "))
    except ValueError:
        print("Invalid input.")
        return

    honest_count = TOTAL_FIRMS - cartel_count

    print(
        f"\nInitializing simulation with {rounds} rounds, {cartel_count} cartel firms, and {honest_count} honest firms..."
    )

    from src.emergent_procurement_env import (ProcurementEmergentEnv,
                                              calibrate_emergent_likelihoods,
                                              calibrate_emergent_thresholds)

    print("Calibrating regulator (Fast Mode)...")
    l0, l1 = calibrate_emergent_likelihoods(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=50,
        rounds_per_episode=20,
        seed0=99,
    )
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0,
        l1,
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        n_episodes=50,
        rounds_per_episode=20,
        seed0=20_099,
    )

    env = ProcurementEmergentEnv(
        reserve_price=RESERVE_PRICE,
        total_firms=TOTAL_FIRMS,
        rounds_per_episode=rounds,
        likelihood_h0=l0,
        likelihood_h1=l1,
        cusum_threshold=cusum_h,
        sr_threshold=sr_h,
    )
    obs, _ = env.reset(seed=42)

    # Assign roles
    all_agents = env.agents[:]
    cartel_firms = all_agents[:cartel_count]
    honest_firms = all_agents[cartel_count:]

    print("\n" + "=" * 80)
    print(f" CUSTOM EVALUATION: {cartel_count} Cartel vs {honest_count} Honest")
    print("=" * 80)

    # Metrics Tracking for the custom run
    metrics_log = []
    all_bids_log = []

    for r in range(1, rounds + 1):
        actions = {}
        for a in env.agents:
            if a in cartel_firms:
                act = np.array([5, 1, 1])  # Strategy 5: Camouflaged Cover Bidding
            else:
                act = np.array([3, 0, 0])  # Strategy 3: Strategic Adaptive Undercutting
            actions[a] = act

        obs, rewards, terms, truncs, infos = env.step(actions)
        g = infos.pop("__common__")

        # Log all 20 bids for the scatter plot
        for a, info in infos.items():
            all_bids_log.append(
                {
                    "Round": r,
                    "Firm": a,
                    "Bid": info["bid"],
                    "Role": "Cartel" if a in cartel_firms else "Honest",
                }
            )

        sorted_agents = sorted(infos.items(), key=lambda kv: kv[1]["bid"])

        # Allocative Efficiency Check
        costs = {a: i["cost"] for a, i in infos.items()}
        min_cost = min(costs.values())
        alloc_eff = 1 if costs[g["winner"]] == min_cost else 0

        rotation_status = "YES" if g.get("alarm_rotation", False) else "no"
        print(
            f"\n--- ROUND {r:2d} --- P(cartel)={g['posterior_collusion']:.3f} "
            f"alarm={'YES' if g['alarm'] else 'no'} | RotAlarm={rotation_status} (Ent={g.get('entropy', 0.0):.2f}) | "
            f"Eng Estimate=${g['engineering_estimate']:.2f} | Buyer Cost=${g['winning_bid']:.2f}"
        )

        for idx, (agent, info) in enumerate(sorted_agents[:5]):
            flag = " <== WINNER" if agent == g["winner"] else ""
            role = "(CARTEL)" if agent in cartel_firms else "(HONEST)"
            print(
                f"    {idx+1:02d}. {agent} {role:8} | Suspicion={info['firm_posterior']:.2f} | markup={info['markup']:.3f}x | "
                f"bid=${info['bid']:.2f} | reward=${info['reward']:.2f}{flag}"
            )

        winner_role = "Cartel" if g["winner"] in cartel_firms else "Honest"

        metrics_log.append(
            {
                "Round": r,
                "P_Cartel": g["posterior_collusion"],
                "Alarm": g["alarm"],
                "RotAlarm": g.get("alarm_rotation", False),
                "EngEstimate": g["engineering_estimate"],
                "BuyerCost": g["winning_bid"],
                "AllocEfficiency": alloc_eff,
                "WinnerType": winner_role,
            }
        )

        if all(terms.values()):
            break

    # Print a matrix/table of metrics
    print("\n" + "=" * 60)
    print(" SIMULATION METRICS MATRIX")
    print("=" * 60)
    df = pd.DataFrame(metrics_log)
    print(df.to_string(index=False))

    print("\nAverage Buyer Cost: $", round(df["BuyerCost"].mean(), 2))
    print("Overall Allocative Efficiency: ", round(df["AllocEfficiency"].mean(), 2))

    # Generate a plot for the custom run
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Custom Simulation Results ({cartel_count} Cartel vs {honest_count} Honest)",
        fontsize=14,
    )

    # 1. Left Plot: Buyer Cost & Alarm over time
    ax1 = axes[0]
    ax1.plot(
        df["Round"], df["BuyerCost"], marker="o", color="tab:green", label="Buyer Cost"
    )
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Buyer Cost ($)", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.bar(
        df["Round"], df["Alarm"], color="red", alpha=0.3, label="Alarm Triggered"
    )
    ax1_twin.set_ylabel("Alarm Status (1=YES)", color="red")
    ax1_twin.set_ylim(0, 1.2)
    ax1.set_title("Buyer Cost & Regulator Alarm")

    # 2. Middle Plot: Winner Distribution (Cartel vs Honest)
    ax2 = axes[1]
    winner_counts = df["WinnerType"].value_counts()
    ax2.pie(
        winner_counts,
        labels=winner_counts.index,
        autopct="%1.1f%%",
        colors=["coral", "skyblue"],
    )
    ax2.set_title("Who Won the Auctions?")

    # 3. Right Plot: Bid Distribution Scatter Plot
    ax3 = axes[2]
    bid_df = pd.DataFrame(all_bids_log)
    cartel_bids = bid_df[bid_df["Role"] == "Cartel"]
    honest_bids = bid_df[bid_df["Role"] == "Honest"]

    ax3.scatter(
        honest_bids["Round"],
        honest_bids["Bid"],
        color="skyblue",
        alpha=0.6,
        label="Honest Bids",
    )
    ax3.scatter(
        cartel_bids["Round"],
        cartel_bids["Bid"],
        color="coral",
        alpha=0.6,
        label="Cartel Cover Bids",
        marker="x",
    )

    ax3.set_xlabel("Round")
    ax3.set_ylabel("Bid Amount ($)")
    ax3.set_title("Distribution of All Bids per Round")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plot_path = (
        os.path.join(os.path.dirname(__file__), "..", "results")
        + "/custom_simulation_plot.png"
    )
    plt.savefig(plot_path)
    print(f"\nSaved simulation graph to {plot_path}")
    # plt.show() # Optional: displays window if running in GUI


if __name__ == "__main__":
    run_menu()
