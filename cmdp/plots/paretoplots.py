"""
CMDP Pareto-style cost vs max-failure-rate plots from evaluation outputs.

Generates one side-by-side figure with morning (period 0) and evening (period 1).

Usage:
    uv run cmdp/plots/paretoplots.py --categories 5 --failure-cost-coef 0.0 --save
    uv run cmdp/plots/paretoplots.py --categories 5 --failure-cost-coef 1.0 --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES, fmt_token

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    if s.startswith("0."):
        s = s[1:]
    return s


def compute_pareto_frontier(costs, fairness):
    """Return indices of Pareto-optimal points (minimizing both objectives)."""
    n = len(costs)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                costs[j] <= costs[i]
                and fairness[j] <= fairness[i]
                and (costs[j] < costs[i] or fairness[j] < fairness[i])
            ):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()
cat = args.categories
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results", f"cat{cat}", "eval")
bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

# Shape: (num_r_max, num_seeds, 2) where last dim is [morning, evening]
max_fr_raw = np.load(
    os.path.join(RESULTS_DIR, f"max_failure_rate_per_period_10seeds_{bf_token}.npy")
)
# Service cost = rebalancing + fleet only. Failures are governed by the
# constraints, so they are excluded here (note: cost_10seeds also folds in a
# failure_rate/10 term, which is intentionally left out of this tradeoff axis).
cost_reb_raw = np.load(os.path.join(RESULTS_DIR, f"cost_reb_10seeds_{bf_token}.npy"))
cost_bikes_raw = np.load(
    os.path.join(RESULTS_DIR, f"cost_bikes_10seeds_{bf_token}.npy")
)

if len(max_fr_raw) != len(R_MAX_VALUES):
    raise ValueError(
        f"Expected {len(R_MAX_VALUES)} r_max points, found {len(max_fr_raw)} in eval arrays"
    )

r_values = R_MAX_VALUES
num_r_max = len(r_values)

# Cost arrays: shape (num_r_max, num_seeds) -> rebalancing + fleet/100, seed-averaged
cost_reb = cost_reb_raw.transpose()
cost_bikes = cost_bikes_raw.transpose()
avg_costs = [
    float(np.mean(cost_reb[:, i] + cost_bikes[:, i] / 100)) for i in range(num_r_max)
]

# Max failure rates per period: average over seeds
avg_max_fr_morning = [np.mean(max_fr_raw[i, :, 0]) for i in range(num_r_max)]
avg_max_fr_evening = [np.mean(max_fr_raw[i, :, 1]) for i in range(num_r_max)]

period_data = [
    ("morning", avg_max_fr_morning),
    ("evening", avg_max_fr_evening),
]

labels = [rf"$r_{{max}}$={_fmt(round(r * 100, 4))}%" for r in r_values]

# Per-point label offsets per period
morning_label_offsets = {}
for value, offset in [
    (1.0, (-15, -25)),
    (0.20, (-25, -28)),
]:
    if value in r_values:
        morning_label_offsets[r_values.index(value)] = offset

evening_label_offsets = {}
for value, offset in [
    (1.0, (-15, -25)),
    (0.20, (8, -28)),
    (0.15, (-35, -25)),
    (0.125, (-30, 8)),
    (0.10, (7, 8)),
    (0.0875, (-15, -25)),
]:
    if value in r_values:
        evening_label_offsets[r_values.index(value)] = offset

DEFAULT_OFFSET = (8, 8)  # above and to the right

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=100)

for idx, (period_name, avg_fr) in enumerate(period_data):
    ax = axes[idx]

    pareto_indices = compute_pareto_frontier(avg_costs, avg_fr)
    dominated_indices = [i for i in range(num_r_max) if i not in pareto_indices]
    label_offsets = (
        morning_label_offsets if period_name == "morning" else evening_label_offsets
    )

    # Plot all points
    for i in range(num_r_max):
        marker = "s" if i in pareto_indices else "o"
        size = 120 if i in pareto_indices else 40
        ax.scatter(
            avg_costs[i], avg_fr[i], size, color="green", marker=marker, zorder=3
        )

    # Draw Pareto staircase
    pareto_sorted = sorted(pareto_indices, key=lambda i: avg_costs[i])
    for k in range(len(pareto_sorted) - 1):
        i, j = pareto_sorted[k], pareto_sorted[k + 1]
        ax.plot(
            [avg_costs[i], avg_costs[j]],
            [avg_fr[i], avg_fr[i]],
            color="blue",
            linewidth=1,
        )
        ax.plot(
            [avg_costs[j], avg_costs[j]],
            [avg_fr[i], avg_fr[j]],
            color="blue",
            linewidth=1,
        )

    # Label Pareto-optimal points
    for i in pareto_indices:
        xytext = label_offsets.get(i, DEFAULT_OFFSET)
        ax.annotate(
            labels[i],
            (avg_costs[i], avg_fr[i]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=16,
        )

    # Label dominated points with short labels
    for i in dominated_indices:
        xytext = label_offsets.get(i, DEFAULT_OFFSET)
        ax.annotate(
            f"{_fmt(round(r_values[i] * 100, 4))}%",
            (avg_costs[i], avg_fr[i]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=16,
        )

    # Legend
    dominated_vals = [
        f"{_fmt(round(r_values[i] * 100, 4))}%" for i in dominated_indices
    ]
    handles = []
    pareto_marker = ax.scatter(
        [], [], marker="s", color="green", s=120, label="Pareto-optimal"
    )
    handles.append(pareto_marker)
    if dominated_vals:
        dominated_label = "Dominated\n(" + ", ".join(dominated_vals) + ")"
        dominated_marker = ax.scatter(
            [], [], marker="o", color="green", s=40, label=dominated_label
        )
        handles.append(dominated_marker)
    ax.legend(
        handles=handles,
        fontsize=16,
        loc="best",
        framealpha=0.4,
    )

    ax.set_title(period_name.capitalize(), fontsize=26)
    ax.set_xlabel("Global service cost", fontsize=24)
    if idx == 0:
        ax.set_ylabel("Max failure rate (%)", fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)

    # Extend x-axis to the right so the rightmost label fits
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi + 5)

    if period_name == "evening":
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, max(yhi, 7.5))

plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"pareto_costs_maxfr_{cat}_cat_{bf_token}.png",
        ),
        format="png",
    )
plt.show()
