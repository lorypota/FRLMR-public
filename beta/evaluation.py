"""
Evaluation Script
========================================================

Evaluates trained Q-learning policies for any category scenario (2-5).
Computes failure rates, Gini coefficient, and global service cost.

Usage:
    uv run beta/evaluation.py --categories 2
    uv run beta/evaluation.py --categories 5 --seeds 100 110

Output (saved to results/):
    results/gini_{M}_cat_{N}seeds.npy
    results/cost_{M}_cat_{N}seeds.npy
    results/cost_reb_{M}_cat_{N}seeds.npy
    results/cost_fail_{M}_cat_{N}seeds.npy
    results/cost_bikes_{M}_cat_{N}seeds.npy
    results/initial_bikes_{M}_cat_{N}seeds.npy
"""

import argparse
import os
import pickle
import random

import numpy as np
from inequalipy.gini import index as gini_index

from beta.config import BETAS
from beta.environment import FairEnv
from common.agent import RebalancingAgent
from common.config import GAMMA, NUM_EVAL_DAYS, PHI, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand
from common.network import generate_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_token(value):
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", required=True, type=int, choices=[2, 3, 4, 5])
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        default=[100, 110],
        help="Seed range [start, end) (default: 100 110)",
    )
    args = parser.parse_args()

    M = args.categories
    cat_dirname = f"cat{M}"
    scenario = get_scenario(M)
    node_list = scenario["node_list"]
    active_cats = scenario["active_cats"]
    demand_params = scenario["demand_params"]
    station_params = scenario["station_params"]

    num_stations = sum(node_list)

    boundaries = scenario["boundaries"]

    seeds = list(range(args.seeds[0], args.seeds[1]))
    num_seeds = len(seeds)

    # Result storage
    gini_values_tot = [[] for _ in range(len(BETAS))]
    costs_tot = [[] for _ in range(len(BETAS))]
    costs_rebalancing = [[] for _ in range(len(BETAS))]
    costs_failures = [[] for _ in range(len(BETAS))]
    costs_bikes = [[] for _ in range(len(BETAS))]
    initial_bikes = [[] for _ in range(len(BETAS))]
    failure_rates_per_cat = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)
            seed_results_dir = os.path.join(
                SCRIPT_DIR, "results", cat_dirname, f"seed{seed}"
            )
            seed_qtables_dir = os.path.join(
                SCRIPT_DIR, "q_tables", cat_dirname, f"seed{seed}"
            )
            b_token = f"b{fmt_token(beta)}"

            n_bikes = np.load(os.path.join(seed_results_dir, f"bikes_{b_token}.npy"))

            G = generate_network(node_list)
            all_days_demand, transformed_demand = generate_global_demand(
                node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
            )

            # Load trained agents
            agents = {}
            for cat in active_cats:
                agent = RebalancingAgent(cat)
                with open(
                    os.path.join(seed_qtables_dir, f"q_table_{b_token}_cat{cat}.pkl"),
                    "rb",
                ) as f:
                    agent.q_table = pickle.load(f)
                agent.set_epsilon(0.0)
                agents[cat] = agent

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA, station_params)
            state = eval_env.reset()

            # Per-category daily tracking
            # cat_index maps active_cats to their position in node_list order (0, 1, 2, ...)
            daily_cat_failures = {cat: [] for cat in active_cats}
            daily_global_failures = []
            daily_global_costs = []

            for day in range(NUM_EVAL_DAYS):
                cat_fails = {cat: 0 for cat in active_cats}
                global_fails = 0
                costs = 0

                for _time_period in (0, 1):
                    actions = np.zeros(num_stations, dtype=np.int64)
                    if day > 0:
                        for i in range(num_stations):
                            cat = G.nodes[i]["station"]
                            actions[i] = agents[cat].decide_action(state[i])

                    next_state, _reward, failures, _reb_costs = eval_env.step(actions)

                    # Accumulate failures per category using boundaries
                    for idx, cat in enumerate(active_cats):
                        cat_fails[cat] += np.sum(
                            failures[boundaries[idx] : boundaries[idx + 1]]
                        )
                    global_fails += np.sum(failures)

                    # Accumulate rebalancing costs using phi
                    for station, action in enumerate(actions):
                        if action != 0:
                            cat = G.nodes[station]["station"]
                            costs += PHI[cat]

                    state = next_state

                if day > 0:
                    for idx, cat in enumerate(active_cats):
                        daily_cat_failures[cat].append(cat_fails[cat] / node_list[idx])
                    daily_global_failures.append(global_fails)
                    daily_global_costs.append(costs)

                if day == 0:
                    initial_bikes[index].append(
                        sum(G.nodes[i]["bikes"] for i in range(num_stations))
                    )

            # Compute requests per category
            cat_requests = {cat: 0 for cat in active_cats}
            global_requests = 0

            for day in range(1, NUM_EVAL_DAYS):
                for hour in range(24):
                    for station in range(num_stations):
                        demand = all_days_demand[day][station][hour]
                        if demand < 0:
                            # Find which category this station belongs to
                            for idx, cat in enumerate(active_cats):
                                if boundaries[idx] <= station < boundaries[idx + 1]:
                                    cat_requests[cat] += abs(demand)
                                    break
                            global_requests += abs(demand)

            # Normalize requests
            for idx, cat in enumerate(active_cats):
                cat_requests[cat] = (
                    cat_requests[cat] / (NUM_EVAL_DAYS - 1) / node_list[idx]
                )
            global_requests = global_requests / (NUM_EVAL_DAYS - 1)

            # Compute failure rates
            cat_failure_rates = {}
            for cat in active_cats:
                cat_failure_rates[cat] = (
                    np.mean(daily_cat_failures[cat]) / cat_requests[cat] * 100
                )
            failure_rate_global = np.mean(daily_global_failures) / global_requests * 100

            # Gini coefficient (ordered from most central to most remote)
            failure_rates_list = [
                cat_failure_rates[cat] for cat in reversed(active_cats)
            ]
            gini = np.round(gini_index(failure_rates_list), 3)

            # Total cost
            total_cost = (
                np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10
            )

            gini_values_tot[index].append(gini)
            costs_tot[index].append(total_cost)
            costs_rebalancing[index].append(np.mean(daily_global_costs))
            costs_failures[index].append(failure_rate_global)
            costs_bikes[index].append(n_bikes)
            failure_rates_per_cat[index].append(
                [cat_failure_rates[cat] for cat in active_cats]
            )

            print(f"Gini={gini:.3f}, Cost={total_cost:.2f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    results_dir = os.path.join(SCRIPT_DIR, "results", cat_dirname, "eval")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f"gini_{num_seeds}seeds.npy"), gini_values_tot)
    np.save(os.path.join(results_dir, f"cost_{num_seeds}seeds.npy"), costs_tot)

    np.save(
        os.path.join(results_dir, f"cost_reb_{num_seeds}seeds.npy"),
        costs_rebalancing,
    )
    np.save(
        os.path.join(results_dir, f"cost_fail_{num_seeds}seeds.npy"),
        costs_failures,
    )
    np.save(
        os.path.join(results_dir, f"cost_bikes_{num_seeds}seeds.npy"),
        costs_bikes,
    )
    np.save(
        os.path.join(results_dir, f"initial_bikes_{num_seeds}seeds.npy"),
        initial_bikes,
    )
    # Shape: (num_betas, num_seeds, num_cats)
    np.save(
        os.path.join(
            results_dir,
            f"failure_rates_per_cat_{num_seeds}seeds.npy",
        ),
        failure_rates_per_cat,
    )


if __name__ == "__main__":
    main()
