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

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import (
    DEMAND_SCALES,
    R_MAX_VALUES,
    build_den_haag_scenario,
)

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
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

    x = np.arange(len(R_MAX_VALUES))

    # =============================================================================
    # PLOT FAILURE RATES AND REBALANCING COSTS
    # =============================================================================

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for cat_idx, cat in enumerate(active_cats):
        morning = rates[:, :, cat_idx, 0]
        evening = rates[:, :, cat_idx, 1]
        ax.plot(
            x,
            np.mean(morning, axis=1),
            color=CATEGORY_COLORS[cat],
            marker="o",
            label=f"Cat {cat}",
        )
        ax.plot(
            x,
            np.mean(evening, axis=1),
            color=CATEGORY_COLORS[cat],
            marker="s",
            linestyle=":",
        )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        np.mean(reb_costs, axis=1),
        color="#d62728",
        marker="D",
        linestyle="-.",
        label="Reb. costs",
    )
    ax.set_xlabel(r"$r_{max}$")
    ax.set_ylabel("Failure rate [%]")
    ax2.set_ylabel("Rebalancing costs")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in R_MAX_VALUES])
    ax.invert_xaxis()
    ax.set_title(f"Den Haag failure rates by r_max, scale {demand_scale}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        path = os.path.join(
            PLOT_DIR, f"failure_rates_by_rmax_{scale_token}_{bf_token}.png"
        )
        fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
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
