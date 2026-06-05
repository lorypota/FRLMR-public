"""
CMDP boxplots (Gini and cost components) from evaluation outputs.

Usage:
    uv run cmdp/plots/boxplots.py --categories 5 --failure-cost-coef 0.0 --save
    uv run cmdp/plots/boxplots.py --categories 5 --failure-cost-coef 1.0 --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoLocator, FuncFormatter, MultipleLocator

from cmdp.config import R_MAX_VALUES, fmt_token

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{args.categories}", "eval")
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

gini_raw = np.load(os.path.join(RESULTS_DIR, f"gini_10seeds_{bf_token}.npy"))
cost_reb_raw = np.load(os.path.join(RESULTS_DIR, f"cost_reb_10seeds_{bf_token}.npy"))
cost_fail_raw = np.load(os.path.join(RESULTS_DIR, f"cost_fail_10seeds_{bf_token}.npy"))
cost_bikes_raw = np.load(
    os.path.join(RESULTS_DIR, f"cost_bikes_10seeds_{bf_token}.npy")
)
max_fail_period_raw = np.load(
    os.path.join(RESULTS_DIR, f"max_failure_rate_per_period_10seeds_{bf_token}.npy")
)
if len(gini_raw) != len(R_MAX_VALUES):
    raise ValueError(
        f"Expected {len(R_MAX_VALUES)} r_max points, found {len(gini_raw)} in eval arrays"
    )
r_values = R_MAX_VALUES
gini = gini_raw.transpose()
cost_reb = cost_reb_raw.transpose()
cost_fail = cost_fail_raw.transpose()
cost_bikes = cost_bikes_raw.transpose()
max_fail_morning = max_fail_period_raw[:, :, 0].transpose()
max_fail_evening = max_fail_period_raw[:, :, 1].transpose()


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    if s.startswith("0."):
        s = s[1:]
    return s


def tick_fmt(val, pos):
    if val == int(val):
        return str(int(val))
    s = f"{val:g}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


r_max_labels = [_fmt(r) for r in r_values]
r_max_pct_labels = [_fmt(round(r * 100, 4)) for r in r_values]
num_r_max = len(r_values)

# GINI INDEX

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[7]
box = ax.boxplot(gini * 100, patch_artist=True, notch=False, vert=True, widths=0.6)
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
ax.set_xlabel(r"$r_{max}$ [%]", fontsize=36)
ax.set_ylabel("Gini index [%]", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_pct_labels, fontsize=26)
ax.invert_xaxis()
ax.tick_params(labelsize=34)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(PLOT_DIR, f"boxplot_gini_{args.categories}_cat_{bf_token}.png"),
        format="png",
    )
plt.show()

# MAX CATEGORY FAILURE RATE (MORNING + EVENING, SIDE BY SIDE)
# r_max is shown in percent so the x-axis matches the percent y-axis.

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=100)
r_max_pct_labels = [_fmt(round(r * 100, 4)) for r in r_values]

panels = [
    (axes[0], max_fail_morning, "Morning", 4, 2.5),
    (axes[1], max_fail_evening, "Evening", 5, 0.5),
]
for ax, data, title, palette_idx, y_step in panels:
    box_color = sns.color_palette("viridis", 11)[palette_idx]
    box = ax.boxplot(data, patch_artist=True, notch=False, vert=True, widths=0.6)
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
        median.set(color="gold", linewidth=1.5)
    for flier in box["fliers"]:
        flier.set(marker="o", color="red", alpha=0.75)
    # optional: draw the r_max target each box must stay under (same % units as y)
    # for k, r in enumerate(r_values, start=1):
    #     ax.hlines(r * 100, k - 0.3, k + 0.3, color="red", linewidth=2)
    ax.set_title(title, fontsize=26)
    ax.set_xlabel(r"$r_{max}$ [%]", fontsize=24)
    if ax is axes[0]:
        ax.set_ylabel("Max failure rate [%]", fontsize=22)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
    ax.set_xticks(range(1, num_r_max + 1))
    ax.set_xticklabels(r_max_pct_labels, fontsize=18)
    ax.invert_xaxis()
    ax.tick_params(labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"boxplot_max_failure_rate_{args.categories}_cat_{bf_token}.png",
        ),
        format="png",
    )
plt.show()

# REBALANCING COSTS

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_reb, patch_artist=True, notch=False, vert=True, widths=0.6)
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
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Weighted rebal. operations", fontsize=26)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.invert_xaxis()
ax.tick_params(labelsize=34)
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            PLOT_DIR, f"boxplot_costs_reb_{args.categories}_cat_{bf_token}.png"
        ),
        format="png",
    )
plt.show()

# FAILURE COSTS

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_fail, patch_artist=True, notch=False, vert=True, widths=0.6)
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
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Failure rate [%]", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.invert_xaxis()
ax.tick_params(labelsize=34)
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            PLOT_DIR, f"boxplot_costs_fails_{args.categories}_cat_{bf_token}.png"
        ),
        format="png",
    )
plt.show()

# NUMBER OF VEHICLES

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_bikes, patch_artist=True, notch=False, vert=True, widths=0.6)
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
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Number of vehicles", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.invert_xaxis()
ax.tick_params(labelsize=34)
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            PLOT_DIR, f"boxplot_costs_bikes_{args.categories}_cat_{bf_token}.png"
        ),
        format="png",
    )
plt.show()
