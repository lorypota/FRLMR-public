"""
Combined 2x2 grid of per-category failure rates and rebalancing cost vs r_max,
for the 2-, 3-, 4-, and 5-category scenarios, with one shared legend.

Usage:
    uv run cmdp/plots/scenarios_grid.py --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from cmdp.config import R_MAX_VALUES, fmt_token
from common.config import get_scenario

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()

num_seeds = args.seeds[1] - args.seeds[0]
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

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
SCENARIOS = [2, 3, 4, 5]
REB_COLOR = "#d62728"


def load_scenario(M):
    results_dir = os.path.join(PLOT_DIR, "..", "results", f"cat{M}", "eval")
    scenario = get_scenario(M)
    active = scenario["active_cats"]
    dp = scenario["demand_params"]
    dep = {cat: {0: 12 * dp[i][0][1], 1: 12 * dp[i][1][1]} for i, cat in enumerate(active)}
    data = np.load(
        os.path.join(
            results_dir, f"failure_rates_per_cat_period_{num_seeds}seeds_{bf_token}.npy"
        )
    )
    rates = np.zeros_like(data)
    for i, cat in enumerate(active):
        for p in (0, 1):
            rates[:, :, i, p] = data[:, :, i, p] / dep[cat][p] * 100
    rates = rates[::-1]
    reb = np.array(
        np.load(os.path.join(results_dir, f"cost_reb_{num_seeds}seeds_{bf_token}.npy"))
    )[::-1]
    return active, rates, reb


sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

x = np.arange(len(R_MAX_VALUES))
xlabels = [f"{v * 100:g}" for v in reversed(R_MAX_VALUES)]
if xlabels and xlabels[0] == "100":
    xlabels[0] = "100\n(no constr.)"

fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True)

for idx, (ax, M) in enumerate(zip(axes.flat, SCENARIOS, strict=True)):
    active, rates, reb = load_scenario(M)
    for i, cat in enumerate(active):
        mm, ms = rates[:, :, i, 0].mean(1), rates[:, :, i, 0].std(1)
        em, es = rates[:, :, i, 1].mean(1), rates[:, :, i, 1].std(1)
        ax.plot(x, mm, color=CATEGORY_COLORS[cat], lw=1.8, marker="o", markersize=5)
        ax.fill_between(x, mm - ms, mm + ms, color=CATEGORY_COLORS[cat], alpha=0.15)
        ax.plot(
            x, em, color=CATEGORY_COLORS[cat], lw=1.8, ls=":", marker="s", markersize=4
        )
        ax.fill_between(x, em - es, em + es, color=CATEGORY_COLORS[cat], alpha=0.10)

    ax2 = ax.twinx()
    rm, rs = reb.mean(1), reb.std(1)
    ax2.plot(x, rm, color=REB_COLOR, lw=1.8, ls="-.", marker="D", markersize=5, alpha=0.8)
    ax2.fill_between(x, rm - rs, rm + rs, color=REB_COLOR, alpha=0.10)
    ax2.tick_params(axis="y", labelcolor=REB_COLOR, labelsize=13)
    ax2.grid(False)
    if idx % 2 == 1:  # right column
        ax2.set_ylabel("Rebalancing costs", color=REB_COLOR, fontsize=16)

    ax.text(
        0.16,
        0.95,
        f"{M} categories",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)
    if idx % 2 == 0:  # left column
        ax.set_ylabel("Failure rate (%)", fontsize=16)
    if idx >= 2:  # bottom row
        ax.set_xlabel(r"$r_{max}$ (%)", fontsize=16)

fig.tight_layout(rect=(0, 0.10, 1, 1))

legend_handles = [Line2D([0], [0], color=CATEGORY_COLORS[c], lw=2.2) for c in range(5)] + [
    Line2D([0], [0], color=REB_COLOR, lw=2.2, ls="-."),
    Line2D([0], [0], color="grey", lw=2.2, ls="-"),
    Line2D([0], [0], color="grey", lw=2.2, ls=":"),
]
legend_labels = [CATEGORY_NAMES[c] for c in range(5)] + [
    "Reb. costs",
    "Morning (0-12h)",
    "Evening (12-24h)",
]
fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=4,
    fontsize=14,
    bbox_to_anchor=(0.5, 0.02),
)

if args.save:
    out = os.path.join(PLOT_DIR, f"scenarios_grid_{bf_token}.png")
    plt.savefig(out, format="png", bbox_inches="tight", dpi=600)
    print(f"Saved: {out}")
