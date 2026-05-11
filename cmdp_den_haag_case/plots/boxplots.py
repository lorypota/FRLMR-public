"""
Den Haag CMDP boxplots from evaluation outputs.

Usage:
    uv run cmdp_den_haag_case/plots/boxplots.py --failure-cost-coef 0.0 --save
    uv run cmdp_den_haag_case/plots/boxplots.py --demand-scales 0.005 0.01 --save
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


def load_array(results_dir, name, num_seeds, bf_token):
    path = os.path.join(results_dir, f"{name}_{num_seeds}seeds_{bf_token}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing evaluation output: {path}")
    return np.load(path)


def save_or_show(fig, path, save):
    plt.tight_layout()
    if save:
        fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_box(data, r_values, ylabel, title, output_path, save):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    box = ax.boxplot(data.transpose(), patch_artist=True, widths=0.6)
    for patch in box["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)
    for median in box["medians"]:
        median.set(color="black", linewidth=1.5)
    ax.set_xlabel(r"$r_{max}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, len(r_values) + 1))
    ax.set_xticklabels([_fmt(value) for value in r_values])
    ax.invert_xaxis()
    save_or_show(fig, output_path, save)


def plot_demand_scale(demand_scale, num_seeds, bf_token, save):
    scale_token = f"scale{fmt_token(demand_scale)}"
    results_dir = os.path.join(PLOT_DIR, "..", "results", "cat5", scale_token, "eval")

    # =============================================================================
    # LOAD EVALUATION ARRAYS
    # =============================================================================

    gini = load_array(results_dir, "gini", num_seeds, bf_token)
    cost_reb = load_array(results_dir, "cost_reb", num_seeds, bf_token)
    cost_fail = load_array(results_dir, "cost_fail", num_seeds, bf_token)
    cost_bikes = load_array(results_dir, "cost_bikes", num_seeds, bf_token)
    max_fail = load_array(
        results_dir, "max_failure_rate_per_period", num_seeds, bf_token
    )

    if len(gini) != len(R_MAX_VALUES):
        raise ValueError(
            f"Expected {len(R_MAX_VALUES)} r_max points, found {len(gini)}"
        )

    plots = [
        ("gini", gini, "Gini index"),
        ("costs_reb", cost_reb, "Weighted rebalancing operations"),
        ("costs_fails", cost_fail, "Failure rate [%]"),
        ("costs_bikes", cost_bikes, "Number of vehicles"),
        ("max_failure_rate_morning", max_fail[:, :, 0], "Max fail. [%] morning"),
        ("max_failure_rate_evening", max_fail[:, :, 1], "Max fail. [%] evening"),
    ]

    # =============================================================================
    # PLOT METRICS
    # =============================================================================

    for name, data, ylabel in plots:
        plot_box(
            data,
            R_MAX_VALUES,
            ylabel,
            f"Den Haag CMDP {ylabel}, demand scale {demand_scale}",
            os.path.join(PLOT_DIR, f"boxplot_{name}_{scale_token}_{bf_token}.png"),
            save,
        )


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
