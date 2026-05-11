"""Plot Den Haag CMDP summary metrics across demand scales."""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from cmdp.config import fmt_token

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-cost-coef", type=float, default=0.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"
    summary_path = os.path.join(
        PLOT_DIR,
        "..",
        "results",
        "cat5",
        "eval",
        f"demand_scale_summary_{bf_token}.csv",
    )
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Missing demand-scale summary: {summary_path}. Run evaluation first."
        )

    rows_by_scale = defaultdict(list)
    with open(summary_path, newline="") as file:
        for row in csv.DictReader(file):
            rows_by_scale[float(row["demand_scale"])].append(row)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    metrics = [
        ("cost_mean", "Mean cost"),
        ("failure_rate_mean", "Mean failure rate [%]"),
        ("gini_mean", "Mean Gini"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for scale, rows in sorted(rows_by_scale.items()):
            rows = sorted(rows, key=lambda item: float(item["r_max"]))
            ax.plot(
                [float(row["r_max"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o",
                label=f"scale {scale:g}",
            )
        ax.set_xlabel(r"$r_{max}$")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
    axes[0].legend()
    plt.tight_layout()
    if args.save:
        path = os.path.join(PLOT_DIR, f"demand_scale_comparison_{bf_token}.png")
        fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.show()


if __name__ == "__main__":
    main()
