"""
Den Haag CMDP lambda convergence plots from saved lambda histories.

This mirrors the purpose of `cmdp/plots/lambda_convergence.py`, but reads
scale-specific Den Haag training histories. Morning lambdas are solid lines and
evening lambdas are dashed lines, averaged over the constrained categories.

Usage:
    uv run cmdp_den_haag_case/plots/lambda_convergence.py --failure-cost-coef 0.0 --save
    uv run cmdp_den_haag_case/plots/lambda_convergence.py --demand-scales 0.005 0.01 --save
"""

import argparse
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D  # line style legend entries

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import DEMAND_SCALES

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_R_MAX_VALUES = [0.005, 0.01, 0.02, 0.04, 0.05]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
STYLE_HANDLES = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Morning"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Evening"),
]


def _fmt(value):
    if value == int(value):
        return str(int(value))
    text = f"{value:g}"
    return text[1:] if text.startswith("0.") else text


def plot_demand_scale(demand_scale, r_max_values, seeds, bf_token, save):
    scale_token = f"scale{fmt_token(demand_scale)}"
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    legend_handles = []

    # =============================================================================
    # LOAD LAMBDA HISTORIES AND PLOT MORNING/EVENING CONVERGENCE
    # =============================================================================

    for r_idx, r_max in enumerate(r_max_values):
        color = COLORS[r_idx % len(COLORS)]
        all_morning, all_evening = [], []

        for seed in seeds:
            path = os.path.join(
                PLOT_DIR,
                "..",
                "results",
                "cat5",
                scale_token,
                f"seed{seed}",
                f"lambda_history_r{fmt_token(r_max)}_{bf_token}.pkl",
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing lambda history: {path}")
            with open(path, "rb") as file:
                history = pickle.load(file)

            morning_vals, evening_vals = [], []
            for _repeat, _day, lambdas in history:
                morning = [pair[0] for pair in lambdas.values()]
                evening = [pair[1] for pair in lambdas.values()]
                morning_vals.append(float(np.mean(morning)) if morning else 0.0)
                evening_vals.append(float(np.mean(evening)) if evening else 0.0)

            all_morning.append(morning_vals)
            all_evening.append(evening_vals)

        min_len = min(len(values) for values in all_morning)
        morning_arr = np.array([values[:min_len] for values in all_morning])
        evening_arr = np.array([values[:min_len] for values in all_evening])
        steps = np.arange(min_len)

        morning_mean = np.mean(morning_arr, axis=0)
        morning_std = np.std(morning_arr, axis=0)
        evening_mean = np.mean(evening_arr, axis=0)
        evening_std = np.std(evening_arr, axis=0)

        ax.plot(steps, morning_mean, color=color, linestyle="-", linewidth=1.5)
        ax.fill_between(
            steps,
            morning_mean - 1.96 * morning_std,
            morning_mean + 1.96 * morning_std,
            color=color,
            alpha=0.15,
        )

        ax.plot(steps, evening_mean, color=color, linestyle="--", linewidth=1.5)
        ax.fill_between(
            steps,
            evening_mean - 1.96 * evening_std,
            evening_mean + 1.96 * evening_std,
            color=color,
            alpha=0.08,
        )

        legend_handles.append(
            mpatches.Patch(color=color, label=rf"$r_{{max}}$={_fmt(r_max)}")
        )

    ax.set_xlabel("Dual update step", fontsize=26)
    ax.set_ylabel(r"$\lambda$", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
    ax.legend(
        handles=legend_handles + STYLE_HANDLES,
        fontsize=18,
        loc="best",
        framealpha=0.4,
    )
    plt.tight_layout()

    if save:
        path = os.path.join(
            PLOT_DIR, f"lambda_convergence_{scale_token}_{bf_token}.png"
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
    parser.add_argument(
        "--r-max-values", nargs="+", type=float, default=DEFAULT_R_MAX_VALUES
    )
    parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
    parser.add_argument("--failure-cost-coef", type=float, default=0.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1]))
    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

    for demand_scale in args.demand_scales:
        plot_demand_scale(
            demand_scale,
            args.r_max_values,
            seeds,
            bf_token,
            args.save,
        )


if __name__ == "__main__":
    main()
