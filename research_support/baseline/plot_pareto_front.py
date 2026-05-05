"""
Baseline Pareto Plot
============================================================

This script generates the Pareto front figure showing the trade-off between
operational cost and fairness (Gini index) for different beta values.

The Pareto front demonstrates that:
- Lower beta (0.0) = lower cost but higher Gini (less fair)
- Higher beta (1.0) = higher cost but lower Gini (more fair)

Output (saved to local plots/ folder):
    plots/pareto_front.pdf
    plots/pareto_front.png
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
GINI_FILE = os.path.join(SCRIPT_DIR, "results", "gini_2_cat_3seeds.npy")
COST_FILE = os.path.join(SCRIPT_DIR, "results", "cost_2_cat_3seeds.npy")

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

# Beta labels for annotation
BETA_LABELS = [
    r"$\beta$=0.0",
    r"$\beta$=0.1",
    r"$\beta$=0.2",
    r"$\beta$=0.3",
    r"$\beta$=0.4",
    r"$\beta$=0.5",
    r"$\beta$=0.6",
    r"$\beta$=0.7",
    r"$\beta$=0.8",
    r"$\beta$=0.9",
    r"$\beta$=1.0",
]

# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================


def main():
    # Load data
    # Shape: (num_seeds, num_betas) after transpose
    gini = np.load(GINI_FILE).transpose()
    cost = np.load(COST_FILE).transpose()

    # Compute averages across seeds for each beta
    avg_ginis = [np.mean(gini[:, i]) for i in range(11)]
    avg_costs = [np.mean(cost[:, i]) for i in range(11)]

    # Print summary statistics
    print("\nResults summary:")
    print("-" * 40)
    print(f"{'Beta':<8} {'Avg Cost':<12} {'Avg Gini':<12}")
    print("-" * 40)
    for i, _beta in enumerate(BETA_LABELS):
        print(f"{i / 10:<8.1f} {avg_costs[i]:<12.3f} {avg_ginis[i]:<12.3f}")
    print("-" * 40)

    # Set up plot style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Identify Pareto-efficient points
    # A point is Pareto-efficient if no other point has both lower cost AND lower Gini
    pareto_efficient = []
    non_pareto = []

    for i in range(11):
        is_dominated = False
        for j in range(11):
            # Check if point j dominates point i
            if (
                i != j
                and avg_costs[j] <= avg_costs[i]
                and avg_ginis[j] <= avg_ginis[i]
                and (avg_costs[j] < avg_costs[i] or avg_ginis[j] < avg_ginis[i])
            ):
                is_dominated = True
                break
        if is_dominated:
            non_pareto.append(i)
        else:
            pareto_efficient.append(i)

    print(f"\nPareto-efficient beta values: {[i / 10 for i in pareto_efficient]}")
    print(f"Dominated beta values: {[i / 10 for i in non_pareto]}")

    # Plot Pareto-efficient points
    pareto_costs = [avg_costs[i] for i in pareto_efficient]
    pareto_ginis = [avg_ginis[i] for i in pareto_efficient]
    ax.scatter(pareto_costs, pareto_ginis, s=40, color="blue", marker="s", zorder=3)

    # Plot non-Pareto points (if any)
    if non_pareto:
        non_pareto_costs = [avg_costs[i] for i in non_pareto]
        non_pareto_ginis = [avg_ginis[i] for i in non_pareto]
        ax.scatter(
            non_pareto_costs, non_pareto_ginis, s=100, color="red", marker="+", zorder=3
        )

    # Draw step-wise Pareto frontier connecting efficient points
    # Sort by cost for proper connection
    sorted_indices = sorted(pareto_efficient, key=lambda i: avg_costs[i])
    for k in range(len(sorted_indices) - 1):
        i = sorted_indices[k]
        j = sorted_indices[k + 1]
        # Horizontal line
        ax.plot(
            [avg_costs[i], avg_costs[j]],
            [avg_ginis[i], avg_ginis[i]],
            color="blue",
            linewidth=1,
            zorder=2,
        )
        # Vertical line
        ax.plot(
            [avg_costs[j], avg_costs[j]],
            [avg_ginis[i], avg_ginis[j]],
            color="blue",
            linewidth=1,
            zorder=2,
        )

    # Add beta labels to each point
    # Offset positions to avoid overlapping with points
    for i in range(11):
        offset_x = -0.5 if i == 0 else -1.2
        offset_y = -0.02
        if i in non_pareto:
            offset_y = 0.015  # Place above for dominated points

        if i == 2:
            offset_y -= 0.01
            offset_x += 0.4

        if i == 4:
            offset_y += 0.01

        if i == 9:
            offset_x = -1
            offset_y = -0.015

        if i == 10:
            offset_x = -0.2
            offset_y = 0.025

        ax.text(
            avg_costs[i] + offset_x,
            avg_ginis[i] + offset_y,
            BETA_LABELS[i],
            fontsize=13,
        )

    # Create legend
    handles = [mpatches.Patch(color="blue", label="Pareto efficient solutions")]
    if non_pareto:
        handles.append(mpatches.Patch(color="red", label="Non-Pareto solutions"))
    ax.legend(handles=handles, fontsize=18, loc="lower left", framealpha=0.4)

    # Labels and formatting
    ax.set_xlabel("Global service cost", fontsize=20)
    ax.set_ylabel("Gini index", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)

    plt.tight_layout()

    # Save figures
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "pareto_front.png")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    print(f"Saved: {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
