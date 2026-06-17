"""
Comparison of reward shaping (beta) vs the constrained MDP (r_max).

Usage:
    uv run cmdp/plots/comparison_reward_shaping.py --categories 5 \
        --failure-cost-coef 1.0 --save
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from beta.config import BETAS
from cmdp.config import R_MAX_VALUES, fmt_token
from common.config import NUM_EVAL_DAYS, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
args = parser.parse_args()
M = args.categories
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

BETA_DIR = os.path.join(PLOT_DIR, "..", "..", "beta", "results", f"cat{M}", "eval")
CMDP_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{M}", "eval")

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]
node_list = scenario["node_list"]
boundaries = scenario["boundaries"]
num_stations = sum(node_list)

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


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    if s.startswith("0."):
        s = s[1:]
    return s


# Whole-day realized requests (net departures, demand < 0) per category, per
# station per day, averaged over the eval seeds. generate_network is RNG-free, so
# seeding then generate_global_demand reproduces the demand the evaluation saw.
# This matches the denominator behind beta's saved whole-day rate and the Gini.
req = np.zeros(len(active_cats))
seeds = list(range(args.seeds[0], args.seeds[1]))
for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    all_days, _ = generate_global_demand(
        node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
    )
    seed_req = np.zeros(len(active_cats))
    for day in range(1, NUM_EVAL_DAYS):
        for hour in range(24):
            for station in range(num_stations):
                d = all_days[day][station][hour]
                if d < 0:
                    for idx in range(len(active_cats)):
                        if boundaries[idx] <= station < boundaries[idx + 1]:
                            seed_req[idx] += abs(d)
                            break
    for idx in range(len(active_cats)):
        seed_req[idx] /= (NUM_EVAL_DAYS - 1) * node_list[idx]
    req += seed_req
req /= len(seeds)

# Reward shaping: whole-day per-category rate is already saved (a rate, %).
beta_rate = np.load(os.path.join(BETA_DIR, "failure_rates_per_cat_10seeds.npy")).mean(
    axis=1
)

# CMDP: whole-day rate = (morning + evening failure counts) / whole-day requests.
cmdp_cnt = np.load(
    os.path.join(CMDP_DIR, f"failure_rates_per_cat_period_10seeds_{bf_token}.npy")
)
num_r = len(R_MAX_VALUES)
cmdp_rate = np.zeros((num_r, len(active_cats)))
for i in range(num_r):
    for ci in range(len(active_cats)):
        cnt = cmdp_cnt[i, :, ci, 0].mean() + cmdp_cnt[i, :, ci, 1].mean()
        cmdp_rate[i, ci] = cnt / req[ci] * 100

# Order the CMDP sweep loose -> tight (left -> right): r_max 100% ... 5%
order = sorted(range(num_r), key=lambda i: -R_MAX_VALUES[i])
cmdp_rate = cmdp_rate[order]
x_cmdp = np.arange(num_r)
cmdp_labels = [_fmt(round(R_MAX_VALUES[i] * 100, 4)) for i in order]
cmdp_labels[0] = "100\n(no constr.)"

ymax = max(beta_rate.max(), cmdp_rate.max())
ytop = np.ceil(ymax / 5) * 5

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), dpi=100)


def plot_panel(ax, x, rate):
    """rate: (len(x), cats). One line per category."""
    for ci, cat in enumerate(active_cats):
        ax.plot(
            x,
            rate[:, ci],
            f"{CATEGORY_MARKERS[cat]}-",
            color=CATEGORY_COLORS[cat],
            linewidth=2,
            markersize=8,
            label=CATEGORY_NAMES[cat],
        )


# Left: reward shaping
axL = axes[0]
plot_panel(axL, BETAS, beta_rate)
axL.set_xlabel(r"Fairness parameter $\beta$", fontsize=24)
axL.set_ylabel("Failure rate [%]", fontsize=24)
axL.set_ylim(0, ytop)
axL.tick_params(labelsize=20)
axL.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.6)
axL.legend(fontsize=16, loc="upper right", framealpha=0.9)

# Right: CMDP
axR = axes[1]
plot_panel(axR, x_cmdp, cmdp_rate)
axR.set_xlabel(r"$r_{max}$ [%]", fontsize=24)
axR.set_ylim(0, ytop)
axR.set_xticks(x_cmdp)
axR.set_xticklabels(cmdp_labels, fontsize=16)
axR.tick_params(labelsize=20)
axR.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.6)

plt.tight_layout()
# Align both x-axis labels to the same height. The right panel's two-line
# "100 (no constr.)" tick would otherwise push its label lower than the left.
axL.xaxis.set_label_coords(0.5, -0.10)
axR.xaxis.set_label_coords(0.5, -0.10)
if args.save:
    out = os.path.join(PLOT_DIR, f"comparison_reward_shaping_{M}_cat_{bf_token}.png")
    plt.savefig(out, format="png", bbox_inches="tight", dpi=600)
    print(f"Saved: {out}")
plt.show()
