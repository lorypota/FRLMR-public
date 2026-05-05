"""
Failure Rate Plot
========================================

This script generates a figure showing how failure rates in remote and central
areas vary with the fairness parameter beta.

The plot demonstrates the beta mapping problem:
- There's no direct way to specify "failure rate <= X%" for a category
- You must search through beta values experimentally to achieve a target
- This motivates the CMDP formulation which allows direct constraints

Output (saved to plots/ folder):
    plots/failure_rates_by_beta.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
REMOTE_FILE = os.path.join(
    SCRIPT_DIR, "results", "failure_rate_remote_2_cat_3seeds.npy"
)
CENTRAL_FILE = os.path.join(
    SCRIPT_DIR, "results", "failure_rate_central_2_cat_3seeds.npy"
)
COSTS_NO_PENALTY_FILE = os.path.join(
    SCRIPT_DIR, "results", "costs_no_penalty_2_cat_3seeds.npy"
)
# COSTS_NO_PENALTY_FILE = os.path.join(SCRIPT_DIR, 'results', 'bike_costs_2_cat_3seeds.npy') # same as n_bikes_2_cat_3seeds / 100

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

# Beta values
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================


def main():
    # Load data
    # Shape: (11 betas, 3 seeds)
    failure_rate_remote = np.load(REMOTE_FILE)
    failure_rate_central = np.load(CENTRAL_FILE)

    # Load costs data if available
    if os.path.exists(COSTS_NO_PENALTY_FILE):
        costs_data = np.load(COSTS_NO_PENALTY_FILE)
        costs_mean = np.array([np.mean(costs_data[i]) for i in range(len(BETAS))])
        costs_std = np.array([np.std(costs_data[i]) for i in range(len(BETAS))])
        print(f"Loaded costs: {costs_data.shape}")
    else:
        costs_mean = None
        costs_std = None
        print(f"Costs file not found: {COSTS_NO_PENALTY_FILE}")

    print(f"Loaded remote failure rates: {failure_rate_remote.shape}")
    print(f"Loaded central failure rates: {failure_rate_central.shape}")

    # Compute statistics across seeds for each beta
    remote_mean = np.array([np.mean(failure_rate_remote[i]) for i in range(len(BETAS))])
    remote_std = np.array([np.std(failure_rate_remote[i]) for i in range(len(BETAS))])

    central_mean = np.array(
        [np.mean(failure_rate_central[i]) for i in range(len(BETAS))]
    )
    central_std = np.array([np.std(failure_rate_central[i]) for i in range(len(BETAS))])

    # Set up plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Plot remote areas (category 0)
    ax.plot(
        BETAS,
        remote_mean,
        "o-",
        color="green",
        linewidth=2,
        markersize=8,
        label="Remote areas (category 0)",
    )
    ax.fill_between(
        BETAS,
        remote_mean - remote_std,
        remote_mean + remote_std,
        color="green",
        alpha=0.2,
    )

    # Plot central areas (category 4)
    ax.plot(
        BETAS,
        central_mean,
        "s-",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        label="Central areas (category 4)",
    )
    ax.fill_between(
        BETAS,
        central_mean - central_std,
        central_mean + central_std,
        color="#1f77b4",
        alpha=0.2,
    )

    # Create secondary y-axis for costs
    ax2 = ax.twinx()

    # Plot costs on secondary axis (dotted line)
    if costs_mean is not None:
        ax2.plot(
            BETAS,
            costs_mean,
            "o--",
            color="#d62728",
            linewidth=2,
            markersize=6,
            label="Costs",
        )
        ax2.fill_between(
            BETAS,
            costs_mean - costs_std,
            costs_mean + costs_std,
            color="#d62728",
            alpha=0.1,
        )
        ax2.set_ylabel("Costs without penalty β", fontsize=16, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=14)
        # Disable grid lines for secondary axis
        ax2.grid(False)

    # Labels and formatting
    ax.set_xlabel(r"Fairness parameter $\beta$", fontsize=16)
    ax.set_ylabel("Failure rate (%)", fontsize=16)
    ax.tick_params(labelsize=14)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    bbox_to_anchor = (0.15, 1)
    if costs_mean is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=14,
            loc="upper left",
            bbox_to_anchor=bbox_to_anchor,
            framealpha=0.9,
        )
    else:
        ax.legend(
            fontsize=14, loc="upper left", bbox_to_anchor=bbox_to_anchor, framealpha=0.9
        )

    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "failure_rates_by_beta.png")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    print(f"\nSaved: {png_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table:")
    print("-" * 80)
    print(
        f"{'Beta':<6} {'Remote FR (%)':<20} {'Central FR (%)':<20} {'Ratio (R/C)':<15}"
    )
    print("-" * 80)

    for i, beta in enumerate(BETAS):
        ratio = (
            remote_mean[i] / central_mean[i] if central_mean[i] > 0 else float("inf")
        )
        print(
            f"{beta:<6.1f} "
            f"{remote_mean[i]:>6.1f} ± {remote_std[i]:<6.1f}     "
            f"{central_mean[i]:>6.1f} ± {central_std[i]:<6.1f}     "
            f"{ratio:>6.2f}"
        )

    print("-" * 80)

    plt.show()


if __name__ == "__main__":
    main()
