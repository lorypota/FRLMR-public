"""
Spread vs floor: per-category failure rates next to the Gini index.

Left: whole-day per-category failure rate against r_max (loose -> tight). These
are the same five per-category rates the Gini index is computed over (one value
per category, whole-day basis), so the panel shows exactly what the Gini
summarizes. Right: boxplot of the Gini index of those rates against r_max.

Usage:
    uv run cmdp/plots/gini_vs_failrates.py --categories 5 --save
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

from cmdp.config import R_MAX_VALUES, fmt_token
from common.config import NUM_EVAL_DAYS, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
args = parser.parse_args()
M = args.categories
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{M}", "eval")

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]
node_list = scenario["node_list"]
boundaries = scenario["boundaries"]
num_stations = sum(node_list)

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
CATEGORY_MARKERS = {0: "o", 1: "^", 2: "D", 3: "v", 4: "s"}


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    if s.startswith("0."):
        s = s[1:]
    return s


def tick_fmt(val, _pos):
    if val == int(val):
        return str(int(val))
    s = f"{val:g}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


# Whole-day realized requests per category (per station per day), averaged over
# eval seeds. Same basis as the saved Gini and per-category rate. generate_network
# is RNG-free, so seeding then generate_global_demand reproduces the eval demand.
seeds = list(range(args.seeds[0], args.seeds[1]))
req = np.zeros(len(active_cats))
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

# Per-category whole-day failure rate, per seed: (morning + evening) / requests.
# cmdp_cnt shape: (num_r_max, num_seeds, num_active_cats, 2)
cmdp_cnt = np.load(
    os.path.join(
        RESULTS_DIR, f"failure_rates_per_cat_period_{len(seeds)}seeds_{bf_token}.npy"
    )
)
num_r = len(R_MAX_VALUES)
num_seeds = cmdp_cnt.shape[1]
rate = np.zeros((num_r, num_seeds, len(active_cats)))
for i in range(num_r):
    for s in range(num_seeds):
        for ci in range(len(active_cats)):
            cnt = cmdp_cnt[i, s, ci, 0] + cmdp_cnt[i, s, ci, 1]
            rate[i, s, ci] = cnt / req[ci] * 100

# Order the sweep loose -> tight (left -> right): r_max 100% ... 5%
order = sorted(range(num_r), key=lambda i: -R_MAX_VALUES[i])
rate = rate[order]
rate_mean = rate.mean(axis=1)  # (num_r, cats)
rate_std = rate.std(axis=1)
x = np.arange(num_r)
labels = [_fmt(round(R_MAX_VALUES[i] * 100, 4)) for i in order]
labels[0] = "100\n(no constr.)"

# Gini index (already saved, fraction). Boxplot keeps R_MAX_VALUES order and uses
# invert_xaxis, which puts 100 (loose) on the left to match the left panel.
gini = np.load(
    os.path.join(RESULTS_DIR, f"gini_{num_seeds}seeds_{bf_token}.npy")
).transpose()
gini_labels = [_fmt(round(r * 100, 4)) for r in R_MAX_VALUES]

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), dpi=100)

# Left: per-category whole-day failure rate
axL = axes[0]
for ci, cat in enumerate(active_cats):
    axL.plot(
        x,
        rate_mean[:, ci],
        f"{CATEGORY_MARKERS[cat]}-",
        color=CATEGORY_COLORS[cat],
        linewidth=2,
        markersize=8,
        label=CATEGORY_NAMES[cat],
    )
    axL.fill_between(
        x,
        rate_mean[:, ci] - rate_std[:, ci],
        rate_mean[:, ci] + rate_std[:, ci],
        color=CATEGORY_COLORS[cat],
        alpha=0.12,
    )
axL.set_xlabel(r"$r_{max}$ [%]", fontsize=24)
axL.set_ylabel("Failure rate [%]", fontsize=24)
axL.set_ylim(bottom=0)
axL.set_xticks(x)
axL.set_xticklabels(labels, fontsize=16)
axL.tick_params(labelsize=20)
axL.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.6)
axL.legend(fontsize=16, loc="upper right", framealpha=0.9)

# Right: Gini index boxplot
axR = axes[1]
box_color = sns.color_palette("viridis", 11)[7]
box = axR.boxplot(gini * 100, patch_artist=True, notch=False, vert=True, widths=0.6)
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
    median.set(color="black", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
axR.set_xlabel(r"$r_{max}$ [%]", fontsize=24)
axR.set_ylabel("Gini index [%]", fontsize=24)
axR.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.6)
axR.set_xticks(range(1, num_r + 1))
axR.set_xticklabels(gini_labels, fontsize=16)
axR.invert_xaxis()
axR.tick_params(labelsize=20)
axR.yaxis.set_major_locator(MultipleLocator(5))
axR.yaxis.set_major_formatter(FuncFormatter(tick_fmt))

plt.tight_layout()
axL.xaxis.set_label_coords(0.5, -0.10)
axR.xaxis.set_label_coords(0.5, -0.10)
if args.save:
    out = os.path.join(PLOT_DIR, f"gini_vs_failrates_{M}_cat_{bf_token}.png")
    plt.savefig(out, format="png", bbox_inches="tight")
    print(f"Saved: {out}")
plt.show()
