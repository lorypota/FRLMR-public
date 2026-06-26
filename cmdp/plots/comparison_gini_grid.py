"""
Combined 2x2 figure comparing reward shaping and the constraint.

Columns are the two methods, rows are the two views. The top row is the
per-category failure rate; the bottom row is the Gini index of those rates, so
each Gini panel summarizes the rates directly above it. Both methods keep the
failure penalty (the constraint at bf=1.0), so the comparison is on the
fairness mechanism alone and the whole figure uses one consistent setting.

Reward shaping drives the Gini index down monotonically because narrowing the
spread is effectively its objective. The constraint does not: its Gini falls
then flattens, because it enforces a floor on the worst category rather than
equalizing. Reading left to right, both columns go from no fairness pressure
to the strongest (beta 0 -> 1; r_max 100% -> 5%).

Panels:
    (a) Reward shaping: per-category failure rate vs beta
    (b) Constraint:     per-category failure rate vs r_max   [bf]
    (c) Gini index of the (a) rates vs beta
    (d) Gini index of the (b) rates vs r_max                 [bf]

Usage:
    uv run cmdp/plots/comparison_gini_grid.py --categories 5 --save
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
parser.add_argument(
    "--constraint-coef",
    type=float,
    default=1.0,
    help="bf for the constraint column; 1.0 keeps the failure penalty (matches "
    "reward shaping), 0.0 isolates the constraint and sharpens the Gini rise.",
)
parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
args = parser.parse_args()
M = args.categories
bf_token = f"bf{fmt_token(args.constraint_coef)}"

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
CATEGORY_COLORS = {0: "#2ca02c", 1: "#8c564b", 2: "#ff7f0e", 3: "#9467bd", 4: "#1f77b4"}
CATEGORY_MARKERS = {0: "o", 1: "^", 2: "D", 3: "v", 4: "s"}
BOX_COLOR = sns.color_palette("viridis", 11)[7]


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
# eval seeds. bf-independent. generate_network is RNG-free, so seeding then
# generate_global_demand reproduces the demand the evaluation saw.
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

# Reward shaping: per-category whole-day rate (already %), per seed. (num_beta, seed, cat)
beta_rate = np.load(os.path.join(BETA_DIR, "failure_rates_per_cat_10seeds.npy"))
beta_rate_mean = beta_rate.mean(axis=1)
beta_rate_std = beta_rate.std(axis=1)
# Reward shaping Gini (fraction), same gini_index as the constraint. (num_beta, seed)
beta_gini = np.load(os.path.join(BETA_DIR, "gini_10seeds.npy"))

# Constraint: per-category whole-day rate (%) per seed = (morning + evening) / req.
cmdp_cnt = np.load(
    os.path.join(CMDP_DIR, f"failure_rates_per_cat_period_10seeds_{bf_token}.npy")
)
num_r = len(R_MAX_VALUES)
num_seeds = cmdp_cnt.shape[1]
cmdp_rate = np.zeros((num_r, num_seeds, len(active_cats)))
for i in range(num_r):
    for s in range(num_seeds):
        for ci in range(len(active_cats)):
            cmdp_rate[i, s, ci] = (
                (cmdp_cnt[i, s, ci, 0] + cmdp_cnt[i, s, ci, 1]) / req[ci] * 100
            )
# Order loose -> tight (left -> right): r_max 100% ... 5%
order = sorted(range(num_r), key=lambda i: -R_MAX_VALUES[i])
cmdp_rate = cmdp_rate[order]
cmdp_rate_mean = cmdp_rate.mean(axis=1)
cmdp_rate_std = cmdp_rate.std(axis=1)
x_cmdp = np.arange(num_r)
cmdp_labels = [_fmt(round(R_MAX_VALUES[i] * 100, 4)) for i in order]
cmdp_labels[0] = "100\n(no constr.)"
# Constraint Gini (fraction). Boxplot keeps R_MAX_VALUES order + invert_xaxis,
# putting 100 (loose) on the left to match the rate panel above.
cmdp_gini = np.load(os.path.join(CMDP_DIR, f"gini_{num_seeds}seeds_{bf_token}.npy"))
gini_labels = [_fmt(round(r * 100, 4)) for r in R_MAX_VALUES]
gini_labels[R_MAX_VALUES.index(1.0)] = "100\n(no constr.)"

rate_top = np.ceil(max(beta_rate_mean.max(), cmdp_rate_mean.max()) / 5) * 5
gini_top = np.ceil(max(beta_gini.max(), cmdp_gini.max()) * 100 / 5) * 5

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(20, 14.5), dpi=100)
(axA, axB), (axC, axD) = axes
TITLE_KW = dict(loc="left", fontsize=24, fontweight="bold")


def plot_rates(ax, x, mean, std):
    for ci, cat in enumerate(active_cats):
        ax.plot(
            x,
            mean[:, ci],
            f"{CATEGORY_MARKERS[cat]}-",
            color=CATEGORY_COLORS[cat],
            linewidth=2,
            markersize=8,
            label=CATEGORY_NAMES[cat],
        )
        ax.fill_between(
            x,
            mean[:, ci] - std[:, ci],
            mean[:, ci] + std[:, ci],
            color=CATEGORY_COLORS[cat],
            alpha=0.12,
        )


def style_box(box):
    for patch in box["boxes"]:
        patch.set_facecolor(BOX_COLOR)
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


def grid_and_ticks(ax, size=20):
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.6)
    ax.tick_params(labelsize=size)


# (a) reward shaping rates
plot_rates(axA, BETAS, beta_rate_mean, beta_rate_std)
axA.set_title("(a)", **TITLE_KW)
axA.set_ylabel("Failure rate [%]", fontsize=24)
axA.set_ylim(0, rate_top)
axA.set_xticks(BETAS)
axA.set_xticklabels([_fmt(b) for b in BETAS], fontsize=16)
axA.legend(fontsize=16, loc="upper right", framealpha=0.9)
grid_and_ticks(axA)

# (b) constraint rates
plot_rates(axB, x_cmdp, cmdp_rate_mean, cmdp_rate_std)
axB.set_title("(b)", **TITLE_KW)
axB.set_ylim(0, rate_top)
axB.set_xticks(x_cmdp)
axB.set_xticklabels(cmdp_labels, fontsize=16)
grid_and_ticks(axB)

# (c) reward shaping Gini
box_c = axC.boxplot(
    (beta_gini * 100).transpose(), patch_artist=True, vert=True, widths=0.6
)
style_box(box_c)
axC.set_title("(c)", **TITLE_KW)
axC.set_xlabel(r"Fairness parameter $\beta$", fontsize=24)
axC.set_ylabel("Gini index [%]", fontsize=24)
axC.set_ylim(0, gini_top)
axC.set_xticks(range(1, len(BETAS) + 1))
axC.set_xticklabels([_fmt(b) for b in BETAS], fontsize=16)
axC.yaxis.set_major_locator(MultipleLocator(5))
axC.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
grid_and_ticks(axC)

# (d) constraint Gini
box_d = axD.boxplot(
    (cmdp_gini * 100).transpose(), patch_artist=True, vert=True, widths=0.6
)
style_box(box_d)
axD.set_title("(d)", **TITLE_KW)
axD.set_xlabel(r"$r_{max}$ [%]", fontsize=24)
axD.set_ylim(0, gini_top)
axD.set_xticks(range(1, num_r + 1))
axD.set_xticklabels(gini_labels, fontsize=16)
axD.invert_xaxis()
axD.yaxis.set_major_locator(MultipleLocator(5))
axD.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
grid_and_ticks(axD)

plt.tight_layout()
for ax in (axC, axD):
    ax.xaxis.set_label_coords(0.5, -0.10)
if args.save:
    out = os.path.join(PLOT_DIR, f"comparison_gini_grid_{M}_cat_{bf_token}.png")
    plt.savefig(out, format="png", bbox_inches="tight", dpi=600)
    print(f"Saved: {out}")
plt.show()
