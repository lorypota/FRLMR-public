"""
CMDP Evaluation Script
========================================================

Evaluates trained CMDP Q-learning policies for any category scenario (2-5).
Computes failure rates, Gini coefficient, global service cost,
and constraint satisfaction for constrained categories.

Usage:
    uv run cmdp/evaluation.py --categories 2
    uv run cmdp/evaluation.py --categories 2 --r-max-values 0.20 0.25
    uv run cmdp/evaluation.py --categories 5 --failure-cost-coef 0.0 --seeds 100 110
    uv run cmdp/evaluation.py --categories 5 --failure-cost-coef 1.0 --seeds 100 110

Output (saved to results/):
    results/gini_{M}_cat_{N}seeds.npy
    results/cost_{M}_cat_{N}seeds.npy
    results/constraint_sat_{M}_cat_{N}seeds.npy
    results/cost_reb_{M}_cat_{N}seeds.npy
    results/cost_fail_{M}_cat_{N}seeds.npy
    results/cost_bikes_{M}_cat_{N}seeds.npy
    results/initial_bikes_{M}_cat_{N}seeds.npy
"""

import argparse
import os
import pickle
import random

import inequalipy as ineq
import numpy as np

from cmdp.config import R_MAX_VALUES, compute_failure_thresholds, fmt_token
from cmdp.environment import CMDPEnv
from common.agent import RebalancingAgent
from common.config import GAMMA, NUM_EVAL_DAYS, PHI, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand
from common.network import generate_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    parser.add_argument(
        "--constrained-cats",
        nargs="+",
        type=int,
        default=None,
        help="Category indices that were constrained during training (default: all active)",
    )
    parser.add_argument(
        "--r-max-values",
        nargs="+",
        type=float,
        default=None,
        help="r_max values to evaluate",
    )
    parser.add_argument(
        "--failure-cost-coef",
        type=float,
        default=0.0,
        help="Base failure coefficient token used in training filenames",
    )
    args = parser.parse_args()

    M = args.categories
    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"
    cat_dirname = f"cat{M}"
    scenario = get_scenario(M)
    node_list = scenario["node_list"]
    active_cats = scenario["active_cats"]
    demand_params = scenario["demand_params"]
    station_params = scenario["station_params"]
    constrained_cats = set(
        args.constrained_cats if args.constrained_cats is not None else active_cats
    )

    r_max_values = args.r_max_values if args.r_max_values else R_MAX_VALUES

    num_stations = sum(node_list)
    boundaries = scenario["boundaries"]

    seeds = list(range(args.seeds[0], args.seeds[1]))
    num_seeds = len(seeds)

    # Result storage
    num_r_max = len(r_max_values)
    num_active_cats = len(active_cats)
    gini_values_tot = [[] for _ in range(num_r_max)]
    costs_tot = [[] for _ in range(num_r_max)]
    costs_rebalancing = [[] for _ in range(num_r_max)]
    costs_failures = [[] for _ in range(num_r_max)]
    costs_bikes = [[] for _ in range(num_r_max)]
    initial_bikes = [[] for _ in range(num_r_max)]
    constraint_satisfaction = [[] for _ in range(num_r_max)]
    max_failure_rates_per_period = np.zeros((num_r_max, num_seeds, 2))
    failure_rates_per_cat_period = np.zeros((num_r_max, num_seeds, num_active_cats, 2))

    for r_idx, r_max in enumerate(r_max_values):
        print(f"\nEvaluating r_max = {r_max}")

        failure_thresholds = compute_failure_thresholds(
            r_max, demand_params, active_cats, constrained_cats
        )

        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            seed_results_dir = os.path.join(
                SCRIPT_DIR, "results", cat_dirname, f"seed{seed}"
            )
            seed_qtables_dir = os.path.join(
                SCRIPT_DIR, "q_tables", cat_dirname, f"seed{seed}"
            )
            r_token = f"r{fmt_token(r_max)}"
            n_bikes = np.load(
                os.path.join(seed_results_dir, f"bikes_{r_token}_{bf_token}.npy")
            )

            G = generate_network(node_list)
            all_days_demand, transformed_demand = generate_global_demand(
                node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
            )

            # Load trained agents
            agents = {}
            for cat in active_cats:
                agent = RebalancingAgent(cat)
                with open(
                    os.path.join(
                        seed_qtables_dir, f"q_table_{r_token}_{bf_token}_cat{cat}.pkl"
                    ),
                    "rb",
                ) as f:
                    agent.q_table = pickle.load(f)
                agent.set_epsilon(0.0)
                agents[cat] = agent

            # Initialize environment (lambdas={} — rewards don't matter for greedy eval)
            eval_env = CMDPEnv(G, transformed_demand, {}, GAMMA, station_params)
            state = eval_env.reset()

            # Per-category daily tracking
            daily_cat_failures = {cat: [] for cat in active_cats}
            daily_global_failures = []
            daily_global_costs = []

            # Per-category per-period failure tracking for all active categories
            period_cat_failures = {}
            for cat in active_cats:
                period_cat_failures[cat] = {0: [], 1: []}

            for day in range(NUM_EVAL_DAYS):
                cat_fails = {cat: 0 for cat in active_cats}
                global_fails = 0
                costs = 0

                # Per-period tracking within this day
                period_fails_today = {
                    cat: {0: 0.0, 1: 0.0} for cat in period_cat_failures
                }

                for _time_period in (0, 1):
                    actions = np.zeros(num_stations, dtype=np.int64)
                    if day > 0:
                        for i in range(num_stations):
                            cat = G.nodes[i]["station"]
                            actions[i] = agents[cat].decide_action(state[i])

                    next_state, _reward, _base_reward, failures, _reb_costs = (
                        eval_env.step(actions)
                    )
                    period = eval_env.current_period

                    # Accumulate failures per category
                    for idx, cat in enumerate(active_cats):
                        cat_fails[cat] += np.sum(
                            failures[boundaries[idx] : boundaries[idx + 1]]
                        )
                    global_fails += np.sum(failures)

                    # Per-period per-category accumulation for constrained cats
                    for idx, cat in enumerate(active_cats):
                        if cat in period_cat_failures:
                            pf = (
                                np.sum(failures[boundaries[idx] : boundaries[idx + 1]])
                                / node_list[idx]
                            )
                            period_fails_today[cat][period] += pf

                    # Accumulate rebalancing costs
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

                    for cat in period_cat_failures:
                        for p in (0, 1):
                            period_cat_failures[cat][p].append(
                                period_fails_today[cat][p]
                            )

                if day == 0:
                    initial_bikes[r_idx].append(
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
            gini = np.round(ineq.gini(failure_rates_list), 3)

            # Total cost
            total_cost = (
                np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10
            )

            satisfaction_tol = 0.05
            n_constraints = 0
            n_satisfied = 0
            for cat in failure_thresholds:
                for p in (0, 1):
                    avg_fail = np.mean(period_cat_failures[cat][p])
                    threshold = failure_thresholds[cat][p]
                    n_constraints += 1
                    if avg_fail <= threshold * (1 + satisfaction_tol):
                        n_satisfied += 1
            satisfaction_fraction = (
                n_satisfied / n_constraints if n_constraints > 0 else 1.0
            )

            gini_values_tot[r_idx].append(gini)
            costs_tot[r_idx].append(total_cost)
            constraint_satisfaction[r_idx].append(satisfaction_fraction)
            costs_rebalancing[r_idx].append(np.mean(daily_global_costs))
            costs_failures[r_idx].append(failure_rate_global)
            costs_bikes[r_idx].append(n_bikes)

            # Store per-category per-period failure rates
            max_rate_by_period = [0.0, 0.0]
            for cat_idx_local, cat in enumerate(active_cats):
                for p in (0, 1):
                    avg_fail = np.mean(period_cat_failures[cat][p])
                    failure_rates_per_cat_period[r_idx, seed_idx, cat_idx_local, p] = (
                        avg_fail
                    )
                    lambda_d = demand_params[cat_idx_local][p][1]
                    rate_pct = (
                        (avg_fail / (12 * lambda_d) * 100) if lambda_d > 0 else 0.0
                    )
                    if rate_pct > max_rate_by_period[p]:
                        max_rate_by_period[p] = rate_pct
            max_failure_rates_per_period[r_idx, seed_idx, 0] = max_rate_by_period[0]
            max_failure_rates_per_period[r_idx, seed_idx, 1] = max_rate_by_period[1]

            print(
                f"Gini={gini:.3f}, Cost={total_cost:.2f}, "
                f"Constraints satisfied={n_satisfied}/{n_constraints}"
            )

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    results_dir = os.path.join(SCRIPT_DIR, "results", cat_dirname, "eval")
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"gini_{num_seeds}seeds_{bf_token}.npy"),
        gini_values_tot,
    )
    np.save(
        os.path.join(results_dir, f"cost_{num_seeds}seeds_{bf_token}.npy"), costs_tot
    )
    np.save(
        os.path.join(results_dir, f"constraint_sat_{num_seeds}seeds_{bf_token}.npy"),
        constraint_satisfaction,
    )
    np.save(
        os.path.join(
            results_dir,
            f"max_failure_rate_per_period_{num_seeds}seeds_{bf_token}.npy",
        ),
        max_failure_rates_per_period,
    )

    np.save(
        os.path.join(
            results_dir,
            f"failure_rates_per_cat_period_{num_seeds}seeds_{bf_token}.npy",
        ),
        failure_rates_per_cat_period,
    )

    np.save(
        os.path.join(results_dir, f"cost_reb_{num_seeds}seeds_{bf_token}.npy"),
        costs_rebalancing,
    )
    np.save(
        os.path.join(results_dir, f"cost_fail_{num_seeds}seeds_{bf_token}.npy"),
        costs_failures,
    )
    np.save(
        os.path.join(results_dir, f"cost_bikes_{num_seeds}seeds_{bf_token}.npy"),
        costs_bikes,
    )
    np.save(
        os.path.join(results_dir, f"initial_bikes_{num_seeds}seeds_{bf_token}.npy"),
        initial_bikes,
    )


if __name__ == "__main__":
    main()
