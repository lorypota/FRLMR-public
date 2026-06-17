"""
Failure Rates of Learned Policies by r_max

Morning periods are solid lines. Evening periods are dotted lines.

Usage:
    uv run cmdp_den_haag_case/plots/failure_rates_by_rmax.py --failure-cost-coef 0.0 --save
    uv run cmdp_den_haag_case/plots/failure_rates_by_rmax.py --demand-scales 0.005 0.01 --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import (
    DEMAND_SCALES,
    R_MAX_VALUES,
    build_den_haag_scenario,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORY_NAMES = {
    0: "Cat 0 (remote)",
    1: "Cat 1",
    2: "Cat 2",
    3: "Cat 3",
    4: "Cat 4 (central)",
}
CATEGORY_COLORS = {
    0: "#2ca02c",
    1: "#8c564b",
    2: "#ff7f0e",
    3: "#9467bd",
    4: "#1f77b4",
}


def plot_demand_scale(demand_scale, num_seeds, bf_token, save):
    scenario = build_den_haag_scenario(demand_scale=demand_scale)
    active_cats = scenario["active_cats"]
    demand_params = scenario["demand_params"]
    cat_period_departures = {
        cat: {
            0: 12 * demand_params[cat_idx][0][1],
            1: 12 * demand_params[cat_idx][1][1],
        }
        for cat_idx, cat in enumerate(active_cats)
    }

    scale_token = f"scale{fmt_token(demand_scale)}"
    results_dir = os.path.join(PLOT_DIR, "..", "results", "cat5", scale_token, "eval")

    # =============================================================================
    # LOAD EVALUATION ARRAYS
    # =============================================================================

    data = np.load(
        os.path.join(
            results_dir,
            f"failure_rates_per_cat_period_{num_seeds}seeds_{bf_token}.npy",
        )
    )
    reb_costs = np.load(
        os.path.join(results_dir, f"cost_reb_{num_seeds}seeds_{bf_token}.npy")
    )

    # =============================================================================
    # CONVERT FAILURE COUNTS TO FAILURE RATES
    # =============================================================================

    rates = np.zeros_like(data)
    for cat_idx, cat in enumerate(active_cats):
        for period in (0, 1):
            rates[:, :, cat_idx, period] = (
                data[:, :, cat_idx, period] / cat_period_departures[cat][period] * 100
            )

    # Reverse so x goes from loose (right) to tight (left), as in the cmdp plots.
    rates = rates[::-1]
    reb_costs = np.asarray(reb_costs)[::-1]
    x = np.arange(len(R_MAX_VALUES))

    # =============================================================================
    # PLOT FAILURE RATES AND REBALANCING COSTS
    # =============================================================================

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for cat_idx, cat in enumerate(active_cats):
        morning = rates[:, :, cat_idx, 0]
        morning_mean = np.mean(morning, axis=1)
        morning_std = np.std(morning, axis=1)

        evening = rates[:, :, cat_idx, 1]
        evening_mean = np.mean(evening, axis=1)
        evening_std = np.std(evening, axis=1)

        # Morning: solid line (with label)
        ax.plot(
            x,
            morning_mean,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            marker="o",
            markersize=5,
            label=CATEGORY_NAMES[cat],
        )
        ax.fill_between(
            x,
            morning_mean - morning_std,
            morning_mean + morning_std,
            color=CATEGORY_COLORS[cat],
            alpha=0.15,
        )

        # Evening: dotted line (no label to avoid legend duplication)
        ax.plot(
            x,
            evening_mean,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            linestyle=":",
            marker="s",
            markersize=4,
        )
        ax.fill_between(
            x,
            evening_mean - evening_std,
            evening_mean + evening_std,
            color=CATEGORY_COLORS[cat],
            alpha=0.1,
        )

    # Secondary y-axis for rebalancing costs
    ax2 = ax.twinx()
    reb_mean = np.mean(reb_costs, axis=1)
    reb_std = np.std(reb_costs, axis=1)
    ax2.plot(
        x,
        reb_mean,
        color="#d62728",
        linewidth=1.5,
        linestyle="-.",
        marker="D",
        markersize=5,
        alpha=0.8,
        label="Reb. costs",
    )
    ax2.fill_between(
        x,
        reb_mean - reb_std,
        reb_mean + reb_std,
        color="#d62728",
        alpha=0.1,
    )
    ax2.set_ylabel("Rebalancing costs", fontsize=20, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=16)
    ax2.grid(False)

    ax.set_xlabel(r"$r_{max}$ (%)", fontsize=20)
    ax.set_ylabel("Failure rate (%)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(x)
    xlabels = [f"{value * 100:g}" for value in reversed(R_MAX_VALUES)]
    if xlabels and xlabels[0] == "100":
        xlabels[0] = "100\n(no constr.)"
    ax.set_xticklabels(xlabels)

    # Combined legend with morning/evening note
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = (
        lines1
        + lines2
        + [
            Line2D(
                [0],
                [0],
                color="grey",
                linewidth=1.5,
                linestyle="-",
                label="Morning (0-12h)",
            ),
            Line2D(
                [0],
                [0],
                color="grey",
                linewidth=1.5,
                linestyle=":",
                label="Evening (12-24h)",
            ),
        ]
    )
    legend_labels = labels1 + labels2 + ["Morning (0-12h)", "Evening (12-24h)"]
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=11,
        loc="upper right",
        bbox_to_anchor=(0.65, 1.0),
        framealpha=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)
    # ax.set_title(rf"Failure rates by $r_{{max}}$ (scale {demand_scale})", fontsize=22)
    plt.tight_layout()

    if save:
        path = os.path.join(
            PLOT_DIR, f"failure_rates_by_rmax_{scale_token}_{bf_token}.png"
        )
        fig.savefig(path, format="png", bbox_inches="tight", dpi=600)
        print(f"Saved: {path}")
    plt.show()


def main():
    # =============================================================================
    # ARGUMENTS
    # =============================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-scales", nargs="+", type=float, default=DEMAND_SCALES)
    parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
    parser.add_argument("--failure-cost-coef", type=float, default=0.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    num_seeds = args.seeds[1] - args.seeds[0]
    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

    for demand_scale in args.demand_scales:
        plot_demand_scale(demand_scale, num_seeds, bf_token, args.save)


if __name__ == "__main__":
    main()
