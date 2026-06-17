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
from matplotlib.ticker import AutoLocator, FuncFormatter, MultipleLocator

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import DEMAND_SCALES, R_MAX_VALUES

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIRIDIS = sns.color_palette("viridis", 11)


def _fmt(value):
    if value == int(value):
        return str(int(value))
    text = f"{value:g}"
    return text[1:] if text.startswith("0.") else text


def tick_fmt(val, _pos):
    if val == int(val):
        return str(int(val))
    s = f"{val:g}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def load_array(results_dir, name, num_seeds, bf_token):
    path = os.path.join(results_dir, f"{name}_{num_seeds}seeds_{bf_token}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing evaluation output: {path}")
    return np.load(path)


def save_or_show(fig, path, save):
    plt.tight_layout()
    if save:
        fig.savefig(path, format="png", bbox_inches="tight", dpi=600)
        print(f"Saved: {path}")
    plt.show()


def plot_box(
    data,
    r_values,
    ylabel,
    ylabel_fontsize,
    tick_labelsize,
    box_color,
    median_color,
    y_locator,
    output_path,
    save,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    box = ax.boxplot(
        data.transpose(), patch_artist=True, notch=False, vert=True, widths=0.6
    )
    for patch in box["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)
        patch.set_linewidth(1.5)
    for whisker in box["whiskers"]:
        whisker.set(color="black", linewidth=1.5, linestyle="--")
    for cap in box["caps"]:
        cap.set(color="black", linewidth=1.5)
    for median in box["medians"]:
        median.set(color=median_color, linewidth=1.5)
    for flier in box["fliers"]:
        flier.set(marker="o", color="red", alpha=0.75)
    ax.set_xlabel(r"$r_{max}$", fontsize=36)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
    ax.set_xticks(range(1, len(r_values) + 1))
    ax.set_xticklabels([_fmt(value) for value in r_values], fontsize=34)
    ax.invert_xaxis()
    ax.tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_major_locator(y_locator)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
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

    # (name, data, ylabel, ylabel_fontsize, tick_labelsize, box_color, median, locator)
    plots = [
        (
            "gini",
            gini,
            "Gini index",
            36,
            34,
            VIRIDIS[7],
            "black",
            MultipleLocator(0.05),
        ),
        (
            "costs_reb",
            cost_reb,
            "Weighted rebal. operations",
            26,
            34,
            VIRIDIS[2],
            "gold",
            AutoLocator(),
        ),
        (
            "costs_fails",
            cost_fail,
            "Failure rate [%]",
            36,
            34,
            VIRIDIS[2],
            "gold",
            AutoLocator(),
        ),
        (
            "costs_bikes",
            cost_bikes,
            "Number of vehicles",
            36,
            34,
            VIRIDIS[2],
            "gold",
            AutoLocator(),
        ),
        (
            "max_failure_rate_morning",
            max_fail[:, :, 0],
            "Max fail. [%] (morning)",
            28,
            32,
            VIRIDIS[4],
            "gold",
            MultipleLocator(2.5),
        ),
        (
            "max_failure_rate_evening",
            max_fail[:, :, 1],
            "Max fail. [%] (evening)",
            28,
            34,
            VIRIDIS[5],
            "gold",
            MultipleLocator(0.5),
        ),
    ]

    # =============================================================================
    # PLOT METRICS
    # =============================================================================

    for (
        name,
        data,
        ylabel,
        ylabel_fontsize,
        tick_labelsize,
        box_color,
        median_color,
        y_locator,
    ) in plots:
        plot_box(
            data,
            R_MAX_VALUES,
            ylabel,
            ylabel_fontsize,
            tick_labelsize,
            box_color,
            median_color,
            y_locator,
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
