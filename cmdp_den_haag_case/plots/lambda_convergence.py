"""
Den Haag CMDP lambda convergence plots from saved lambda histories.

This mirrors the purpose of `cmdp/plots/lambda_convergence.py`, but reads
scale-specific Den Haag training histories. The plot shows the mean lambda over
all constrained category-period pairs for each selected r_max.

Usage:
    uv run cmdp_den_haag_case/plots/lambda_convergence.py --failure-cost-coef 0.0 --save
    uv run cmdp_den_haag_case/plots/lambda_convergence.py --demand-scales 0.005 0.01 --save
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import fmt_token
from cmdp_den_haag_case.config import DEMAND_SCALES

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_R_MAX_VALUES = [0.005, 0.01, 0.02, 0.04, 0.05]


def plot_demand_scale(demand_scale, r_max_values, seeds, bf_token, save):
    scale_token = f"scale{fmt_token(demand_scale)}"
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    # =============================================================================
    # LOAD LAMBDA HISTORIES AND PLOT MEAN LAMBDA
    # =============================================================================

    for r_max in r_max_values:
        all_means = []
        for seed in seeds:
            path = os.path.join(
                PLOT_DIR,
                "..",
                "results",
                "cat5",
                scale_token,
                f"seed{seed}",
                f"lambda_history_r{fmt_token(r_max)}_{bf_token}.pkl",
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing lambda history: {path}")
            with open(path, "rb") as file:
                history = pickle.load(file)
            values = []
            for _repeat, _day, lambdas in history:
                flat = [value for pair in lambdas.values() for value in pair]
                values.append(float(np.mean(flat)) if flat else 0.0)
            all_means.append(values)

        min_len = min(len(values) for values in all_means)
        arr = np.asarray([values[:min_len] for values in all_means])
        steps = np.arange(min_len)
        ax.plot(steps, np.mean(arr, axis=0), label=rf"$r_{{max}}$={r_max:g}")

    ax.set_xlabel("Dual update step")
    ax.set_ylabel(r"Mean $\lambda$")
    ax.set_title(f"Den Haag lambda convergence, scale {demand_scale}")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(
            PLOT_DIR, f"lambda_convergence_{scale_token}_{bf_token}.png"
        )
        fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.show()


def main():
    # =============================================================================
    # ARGUMENTS
    # =============================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-scales", nargs="+", type=float, default=DEMAND_SCALES)
    parser.add_argument(
        "--r-max-values", nargs="+", type=float, default=DEFAULT_R_MAX_VALUES
    )
    parser.add_argument("--seeds", nargs=2, type=int, default=[100, 110])
    parser.add_argument("--failure-cost-coef", type=float, default=0.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1]))
    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"

    for demand_scale in args.demand_scales:
        plot_demand_scale(
            demand_scale,
            args.r_max_values,
            seeds,
            bf_token,
            args.save,
        )


if __name__ == "__main__":
    main()
