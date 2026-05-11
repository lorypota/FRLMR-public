"""
Den Haag CMDP Pareto-style cost vs max-failure-rate plots.

Generates separate figures for morning (period 0) and evening (period 1).

Usage:
    uv run cmdp_den_haag_case/plots/paretoplots.py --failure-cost-coef 0.0 --save
    uv run cmdp_den_haag_case/plots/paretoplots.py --demand-scales 0.005 0.01 --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import DEMAND_SCALES, R_MAX_VALUES

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt(value):
    if value == int(value):
        return str(int(value))
    text = f"{value:g}"
    return text[1:] if text.startswith("0.") else text


def pareto_indices(costs, failures):
    indices = []
    for i, (cost_i, fail_i) in enumerate(zip(costs, failures, strict=True)):
        dominated = False
        for j, (cost_j, fail_j) in enumerate(zip(costs, failures, strict=True)):
            if i == j:
                continue
            if (
                cost_j <= cost_i
                and fail_j <= fail_i
                and (cost_j < cost_i or fail_j < fail_i)
            ):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return indices


def plot_demand_scale(demand_scale, num_seeds, bf_token, save):
    scale_token = f"scale{fmt_token(demand_scale)}"
    results_dir = os.path.join(PLOT_DIR, "..", "results", "cat5", scale_token, "eval")

    # =============================================================================
    # LOAD EVALUATION ARRAYS
    # =============================================================================

    max_fr = np.load(
        os.path.join(
            results_dir,
            f"max_failure_rate_per_period_{num_seeds}seeds_{bf_token}.npy",
        )
    )
    costs = np.load(os.path.join(results_dir, f"cost_{num_seeds}seeds_{bf_token}.npy"))
    avg_costs = np.mean(costs, axis=1)
    period_data = [
        ("morning", np.mean(max_fr[:, :, 0], axis=1)),
        ("evening", np.mean(max_fr[:, :, 1], axis=1)),
    ]

    # =============================================================================
    # PARETO PLOTS
    # =============================================================================

    sns.set_theme(style="whitegrid")
    for period_name, avg_failures in period_data:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        pareto = pareto_indices(avg_costs, avg_failures)
        for i, r_max in enumerate(R_MAX_VALUES):
            marker = "s" if i in pareto else "o"
            size = 120 if i in pareto else 50
            ax.scatter(avg_costs[i], avg_failures[i], s=size, marker=marker)
            ax.annotate(
                _fmt(r_max),
                (avg_costs[i], avg_failures[i]),
                textcoords="offset points",
                xytext=(8, 8),
            )
        ax.set_xlabel("Global service cost")
        ax.set_ylabel("Max failure rate [%]")
        ax.set_title(f"Den Haag CMDP Pareto plot ({period_name}), scale {demand_scale}")
        plt.tight_layout()
        if save:
            path = os.path.join(
                PLOT_DIR,
                f"pareto_costs_maxfr_{period_name}_{scale_token}_{bf_token}.png",
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
