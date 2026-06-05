"""
Failure Rates of Learned Policies by r_max

Plots per-category failure rates (from evaluation) against r_max values.
Morning periods shown as solid lines, evening as dotted.
X-axis goes from 1.0 (loose) to 0.05 (tight).

Usage:
    uv run cmdp/plots/failure_rates_by_rmax.py --categories 5
    uv run cmdp/plots/failure_rates_by_rmax.py --categories 5 --save
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
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument(
    "--seeds",
    nargs=2,
    type=int,
    default=[100, 110],
    help="Seed range [start, end)",
)
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()

M = args.categories
num_seeds = args.seeds[1] - args.seeds[0]
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{M}", "eval")

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]

# Per-period departure rates: 12 * lambda_d for each (cat, period)
cat_period_departures = {}
for cat_idx, cat in enumerate(active_cats):
    cat_period_departures[cat] = {
        0: 12 * demand_params[cat_idx][0][1],  # morning
        1: 12 * demand_params[cat_idx][1][1],  # evening
    }

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

# Load evaluation results: shape (num_r_max, num_seeds, num_active_cats, 2)
data = np.load(
    os.path.join(
        RESULTS_DIR,
        f"failure_rates_per_cat_period_{num_seeds}seeds_{bf_token}.npy",
    )
)
if data.shape[0] != len(R_MAX_VALUES):
    raise ValueError(
        f"Expected {len(R_MAX_VALUES)} r_max points, found {data.shape[0]} in eval arrays"
    )
all_r_values = R_MAX_VALUES

# Plot the full configured sweep in cmdp.config.R_MAX_VALUES.
PLOT_R_MAX = list(all_r_values)
keep_idx = list(range(len(all_r_values)))

data = data[keep_idx]
x = np.arange(len(PLOT_R_MAX))

# Convert failure counts to rates (%)
# data[r, s, c, p] is per-area failure count; divide by departures to get rate
rates = np.zeros_like(data)
for cat_idx, cat in enumerate(active_cats):
    for p in (0, 1):
        rates[:, :, cat_idx, p] = (
            data[:, :, cat_idx, p] / cat_period_departures[cat][p] * 100
        )

# Reverse so x goes from loose (25%) to tight (5%)
rates = rates[::-1]

# Load rebalancing costs: shape (num_r_max, num_seeds)
reb_costs = np.load(
    os.path.join(RESULTS_DIR, f"cost_reb_{num_seeds}seeds_{bf_token}.npy")
)
reb_costs = np.array(reb_costs)[keep_idx][::-1]  # filter and reverse

# Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

for cat_idx, cat in enumerate(active_cats):
    # Morning: shape (num_r_max, num_seeds) -> mean/std over seeds
    morn = rates[:, :, cat_idx, 0]
    morn_mean = np.mean(morn, axis=1)
    morn_std = np.std(morn, axis=1)

    eve = rates[:, :, cat_idx, 1]
    eve_mean = np.mean(eve, axis=1)
    eve_std = np.std(eve, axis=1)

    # Morning: solid line (with label)
    ax.plot(
        x,
        morn_mean,
        color=CATEGORY_COLORS[cat],
        linewidth=1.5,
        marker="o",
        markersize=5,
        label=CATEGORY_NAMES[cat],
    )
    ax.fill_between(
        x,
        morn_mean - morn_std,
        morn_mean + morn_std,
        color=CATEGORY_COLORS[cat],
        alpha=0.15,
    )

    # Evening: dotted line (no label to avoid legend duplication)
    ax.plot(
        x,
        eve_mean,
        color=CATEGORY_COLORS[cat],
        linewidth=1.5,
        linestyle=":",
        marker="s",
        markersize=4,
    )
    ax.fill_between(
        x,
        eve_mean - eve_std,
        eve_mean + eve_std,
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
xlabels = [f"{v * 100:g}" for v in reversed(PLOT_R_MAX)]
if xlabels and xlabels[0] == "100":
    xlabels[0] = "100\n(no constr.)"
ax.set_xticklabels(xlabels)

# Combined legend with morning/evening note
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Add dummy entries for morning/evening explanation
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
    bbox_to_anchor=(0.45, 1.0),
    framealpha=0.9,
    handlelength=1.5,
    handletextpad=0.4,
    columnspacing=0.8,
)

ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)
# ax.set_title(f"Failure rates by $r_{{max}}$ ({M} categories)", fontsize=22)
plt.tight_layout()

if args.save:
    out_path = os.path.join(PLOT_DIR, f"failure_rates_by_rmax_{M}_cat_{bf_token}.png")
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

plt.show()
