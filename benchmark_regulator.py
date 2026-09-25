import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.emergent_procurement_env import (calibrate_emergent_likelihoods,
                                          calibrate_emergent_thresholds)
from src.regulator import (BayesianSequentialRegulator, RegulatorConfig,
                           compute_screens)


# ==========================================
# 1. Simulate Bidding Data
# ==========================================
def generate_benchmark_data(num_rounds=50, num_honest=5, num_cartel=5):
    """
    Generates a synthetic sequence of bids where the Cartel is using
    a highly advanced Bid-Rotation + Camouflage Cover Bidding strategy.
    """
    history = []
    rotation_queue = [f"cartel_{i}" for i in range(num_cartel)]
    honest_firms = [f"honest_{i}" for i in range(num_honest)]

    for r in range(num_rounds):
        # Base true costs matching the calibration scale (~100 reserve price)
        costs = {f: random.uniform(80.0, 95.0) for f in honest_firms + rotation_queue}
        eng_estimate = round(min(costs.values()) * random.uniform(0.95, 1.05), 2)

        bids = {}
        # Honest firms bid competitively (markup ~ 1.05x)
        for f in honest_firms:
            bids[f] = costs[f] * random.uniform(1.02, 1.08)

        # Cartel logic (Rotation)
        designated = rotation_queue.pop(0)
        rotation_queue.append(designated)

        # Designated winner bids just below the lowest honest bid to guarantee win
        lowest_honest = min(bids[h] for h in honest_firms)
        bids[designated] = lowest_honest - random.uniform(0.1, 0.5)

        # Other cartel members submit highly suspicious clustered cover bids
        for f in rotation_queue[:-1]:  # exclude designated
            bids[f] = lowest_honest + random.uniform(0.5, 1.5)

        history.append(
            {
                "round": r,
                "eng_estimate": eng_estimate,
                "bids": bids,
                "cartel_firms": set(rotation_queue),
                "honest_firms": set(honest_firms),
                "designated": designated,
            }
        )
    return history


# ==========================================
# 2. Run Benchmarks
# ==========================================
def run_benchmarks():
    print("Calibrating the Bayesian Regulators (this takes a few seconds)...")
    l0, l1 = calibrate_emergent_likelihoods(
        n_episodes=20, rounds_per_episode=20, seed0=42
    )
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0, l1, n_episodes=20, rounds_per_episode=20, seed0=142
    )

    data = generate_benchmark_data(num_rounds=100)

    # Init Regulators
    reg_before = BayesianSequentialRegulator(
        l0,
        l1,
        RegulatorConfig(
            cusum_threshold=cusum_h, sr_threshold=sr_h, eng_estimate_default=100.0
        ),
    )

    reg_after = BayesianSequentialRegulator(
        l0,
        l1,
        RegulatorConfig(
            cusum_threshold=cusum_h,
            sr_threshold=sr_h,
            eng_estimate_default=100.0,
            rotation_window=15,
            entropy_threshold=1.7,
        ),
    )

    results_before = []
    results_after = []

    print("\nRunning 'Before' and 'After' simulations...")
    for rd in data:
        bids = rd["bids"]

        # --- 'BEFORE' LOGIC ---
        # Used a static reserve price (100.0) instead of eng_estimate
        res_b = reg_before.update(bids, eng_estimate=100.0)
        global_alarm_b = res_b["alarm_cusum"] or res_b["alarm_sr"]

        fined_before = []
        if global_alarm_b:
            fined_before = list(bids.keys())  # Global alarm fines EVERYONE

        results_before.append(
            {
                "round": rd["round"],
                "fined_honest": len([f for f in fined_before if "honest" in f]),
                "fined_cartel": len([f for f in fined_before if "cartel" in f]),
                "detected": global_alarm_b,
            }
        )

        # --- 'AFTER' LOGIC ---
        # Uses true eng_estimate, individual suspicion, and entropy rotation tracking
        res_a = reg_after.update(bids, eng_estimate=rd["eng_estimate"])

        fined_after = []
        # Surgical fines (Individual Firm Suspicion > 80%)
        for firm, prob in res_a.get("firm_posteriors", {}).items():
            if prob > 0.80:
                fined_after.append(firm)

        # Conspiracy Fines (Rotation Alarm)
        if res_a.get("alarm_rotation", False):
            for firm in set(reg_after.winner_history):
                if firm not in fined_after:
                    fined_after.append(firm)

        results_after.append(
            {
                "round": rd["round"],
                "fined_honest": len([f for f in fined_after if "honest" in f]),
                "fined_cartel": len([f for f in fined_after if "cartel" in f]),
                "detected": len(fined_after) > 0,
            }
        )

    # ==========================================
    # 3. Process & Plot Metrics
    # ==========================================
    df_b = pd.DataFrame(results_before)
    df_a = pd.DataFrame(results_after)

    # 1. Total Innocent Firms Fined (False Positives)
    fp_b = df_b["fined_honest"].sum()
    fp_a = df_a["fined_honest"].sum()

    # 2. Time to First Detection
    ttd_b = (
        df_b[df_b["detected"] == True]["round"].min() if df_b["detected"].any() else 100
    )
    ttd_a = (
        df_a[df_a["detected"] == True]["round"].min() if df_a["detected"].any() else 100
    )

    print("\n" + "=" * 50)
    print(" BENCHMARK RESULTS: BEFORE vs AFTER")
    print("=" * 50)
    print(f"False Positives (Honest Firms Fined):")
    print(f"  Before: {fp_b}")
    print(f"  After : {fp_a}")
    print(f"Time to Detection (Rounds):")
    print(f"  Before: {ttd_b} rounds (Often evaded indefinitely)")
    print(f"  After : {ttd_a} rounds")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar Chart: False Positives
    ax1.bar(
        ["Legacy Regulator", "Upgraded Regulator"],
        [fp_b, fp_a],
        color=["#ef5350", "#66bb6a"],
    )
    ax1.set_title(
        "Total Fines Levied Against Innocent Firms\n(Lower is Better)",
        fontsize=14,
        weight="bold",
    )
    ax1.set_ylabel("Number of False Positives")

    # Line Chart: Cartel Fines Over Time
    ax2.plot(
        df_b["round"],
        df_b["fined_cartel"].cumsum(),
        color="#ef5350",
        label="Legacy Regulator",
        linewidth=2,
    )
    ax2.plot(
        df_a["round"],
        df_a["fined_cartel"].cumsum(),
        color="#66bb6a",
        label="Upgraded Regulator",
        linewidth=2,
    )
    ax2.set_title(
        "Cumulative Fines Levied Against Cartel Firms\n(Higher & Faster is Better)",
        fontsize=14,
        weight="bold",
    )
    ax2.set_xlabel("Simulation Round")
    ax2.set_ylabel("Total Cartel Fines")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/regulator_benchmark.png", dpi=300)
    print("\nSaved benchmark graph to results/regulator_benchmark.png")


if __name__ == "__main__":
    run_benchmarks()
