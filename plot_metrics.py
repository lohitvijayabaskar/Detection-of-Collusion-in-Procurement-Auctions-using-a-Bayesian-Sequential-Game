import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "coadaptation_log.csv"
OUTPUT_FILE = "metrics_plot.png"

def plot_metrics():
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} not found.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("Log file is empty.")
        return

    # Create a figure with 3 subplots (stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Emergent Collusion & Regulator Performance', fontsize=16)

    # 1. Regulator Performance (TPR, FPR, Alarm Rate)
    ax1 = axes[0]
    ax1.plot(df['iteration'], df['alarm_rate'], label='Overall Alarm Rate', linestyle='--', color='gray')
    ax1.plot(df['iteration'], df['tpr'], label='TPR (True Positives)', color='red', marker='o')
    ax1.plot(df['iteration'], df['fpr'], label='FPR (False Positives)', color='blue', marker='x')
    ax1.set_title('Regulator Detection Accuracy')
    ax1.set_ylabel('Rate (0.0 to 1.0)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Economic Impact (Buyer Cost vs Allocative Efficiency)
    ax2 = axes[1]
    color1 = 'tab:green'
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Buyer Cost ($)', color=color1)
    ax2.plot(df['iteration'], df['avg_buyer_cost'], color=color1, marker='s', label='Buyer Cost')
    ax2.tick_params(axis='y', labelcolor=color1)
    
    ax2_twin = ax2.twinx()
    color2 = 'tab:purple'
    ax2_twin.set_ylabel('Allocative Efficiency', color=color2)
    ax2_twin.plot(df['iteration'], df['allocative_efficiency'], color=color2, marker='^', linestyle=':', label='Efficiency')
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2.set_title('Economic Impact: Cost vs Efficiency')
    ax2.grid(True, alpha=0.3)

    # 3. Strategy Distribution
    ax3 = axes[2]
    strats = ['strat_0_pct', 'strat_1_pct', 'strat_2_pct', 'strat_3_pct', 'strat_4_pct', 'strat_5_pct', 'strat_6_pct']
    labels = ['0: Honest Bertrand', '1: Cournot Low', '2: Cournot High', '3: Random', '4: Target Price', '5: Cover Bid', '6: Adaptive Undercut']
    
    # Filter only columns that exist
    available_strats = [s for s in strats if s in df.columns]
    available_labels = [labels[strats.index(s)] for s in available_strats]
    
    ax3.stackplot(df['iteration'], *[df[s] for s in available_strats], labels=available_labels, alpha=0.8)
    ax3.set_title('Firm Strategy Distribution Over Time')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Percentage (%)')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_metrics()
