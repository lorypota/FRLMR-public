"""
Failure Rates by Beta
=====================================

Plots per-category failure rates vs beta, with rebalancing costs on
a secondary y-axis. Similar to research_support/failure_rate_analysis
but for 5 categories.

Requires: failure_rates_per_cat_{M}_cat_{N}seeds.npy from evaluation.py

Usage:
    uv run beta/plots/failure_rates_by_beta.py --categories 5
    uv run beta/plots/failure_rates_by_beta.py --categories 5 --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from beta.config import BETAS
from common.config import get_scenario

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = None

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

M = args.categories
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{M}", "eval")

# Load per-category failure rates: shape (num_betas, num_seeds, num_cats)
fr_file = os.path.join(RESULTS_DIR, "failure_rates_per_cat_10seeds.npy")
fr_data = np.load(fr_file)
print(f"Loaded failure rates: {fr_data.shape}")

# Load costs without penalty
costs_file = os.path.join(RESULTS_DIR, "cost_reb_10seeds.npy")
if os.path.exists(costs_file):
    costs_data = np.load(costs_file)
    costs_mean = np.array([np.mean(costs_data[i]) for i in range(len(BETAS))])
    costs_std = np.array([np.std(costs_data[i]) for i in range(len(BETAS))])
else:
    costs_mean = None
    costs_std = None
    print(f"Costs file not found: {costs_file}")

scenario = get_scenario(M)
active_cats = scenario["active_cats"]

CATEGORY_NAMES = {
    0: "Remote (cat 0)",
    1: "Suburban-remote (cat 1)",
    2: "Suburban (cat 2)",
    3: "Suburban-central (cat 3)",
    4: "Central (cat 4)",
}
CATEGORY_COLORS = {
    0: "#2ca02c",
    1: "#8c564b",
    2: "#ff7f0e",
    3: "#9467bd",
    4: "#1f77b4",
}
CATEGORY_MARKERS = {0: "o", 1: "^", 2: "D", 3: "v", 4: "s"}

# Compute mean and std per category across seeds
# fr_data shape: (num_betas, num_seeds, num_cats)
num_cats = len(active_cats)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

for cat_idx, cat in enumerate(active_cats):
    cat_means = np.array([np.mean(fr_data[i, :, cat_idx]) for i in range(len(BETAS))])
    cat_stds = np.array([np.std(fr_data[i, :, cat_idx]) for i in range(len(BETAS))])

    ax.plot(
        BETAS,
        cat_means,
        f"{CATEGORY_MARKERS[cat]}-",
        color=CATEGORY_COLORS[cat],
        linewidth=2,
        markersize=8,
        label=CATEGORY_NAMES[cat],
    )
    ax.fill_between(
        BETAS,
        cat_means - cat_stds,
        cat_means + cat_stds,
        color=CATEGORY_COLORS[cat],
        alpha=0.15,
    )

# Secondary y-axis for costs
lines2, labels2 = [], []
if costs_mean is not None:
    ax2 = ax.twinx()
    ax2.plot(
        BETAS,
        costs_mean,
        "o--",
        color="#d62728",
        linewidth=2,
        markersize=6,
        label="Reb. costs",
    )
    ax2.fill_between(
        BETAS,
        costs_mean - costs_std,
        costs_mean + costs_std,
        color="#d62728",
        alpha=0.1,
    )
    ax2.set_ylabel("Rebalancing costs", fontsize=16, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=14)
    ax2.grid(False)
    lines2, labels2 = ax2.get_legend_handles_labels()

ax.set_xlabel(r"Fairness parameter $\beta$", fontsize=16)
ax.set_ylabel("Failure rate (%)", fontsize=16)
ax.tick_params(labelsize=14)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=12,
    loc="best",
    framealpha=0.9,
)

ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)
plt.tight_layout()

if args.save:
    out_path = os.path.join(PLOT_DIR, f"failure_rates_by_beta_{M}_cat.png")
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

plt.show()
