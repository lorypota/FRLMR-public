"""
Den Haag CMDP Evaluation Script
========================================================

Evaluates trained Den Haag CMDP Q-learning policies.
Computes failure rates, Gini coefficient, global service cost,
and constraint satisfaction for constrained categories.

Usage:
    uv run cmdp_den_haag_case/evaluation.py
    uv run cmdp_den_haag_case/evaluation.py --demand-scales 0.01
    uv run cmdp_den_haag_case/evaluation.py --r-max-values 0.005 0.01 0.02
"""

import argparse
import csv
import os
import pickle
import random

import inequalipy as ineq
import numpy as np

from cmdp.config import compute_failure_thresholds, fmt_token
from cmdp_den_haag_case.config import (
    DEMAND_SCALES,
    R_MAX_VALUES,
    build_den_haag_network,
    build_den_haag_scenario,
)
from cmdp_den_haag_case.zone_model import (
    ZoneCMDPEnv,
    ZoneRebalancingAgent,
    generate_separate_event_demand,
)
from common.config import GAMMA, NUM_EVAL_DAYS, TIME_SLOTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trained Den Haag CMDP output: {path}\n")
    return path


def summarize_scale_results(
    demand_scale,
    r_max_values,
    gini_values,
    costs,
    constraint_satisfaction,
    costs_rebalancing,
    costs_failures,
    costs_bikes,
    max_failure_rates_per_period,
):
    rows = []
    for r_idx, r_max in enumerate(r_max_values):
        constraints = np.asarray(constraint_satisfaction[r_idx], dtype=float)
        max_failure_rates = max_failure_rates_per_period[r_idx]
        rows.append(
            {
                "demand_scale": demand_scale,
                "r_max": r_max,
                "gini_mean": float(np.mean(gini_values[r_idx])),
                "gini_std": float(np.std(gini_values[r_idx])),
                "cost_mean": float(np.mean(costs[r_idx])),
                "cost_std": float(np.std(costs[r_idx])),
                "constraint_satisfaction_rate": float(np.mean(constraints)),
                "reb_cost_mean": float(np.mean(costs_rebalancing[r_idx])),
                "failure_rate_mean": float(np.mean(costs_failures[r_idx])),
                "bikes_mean": float(np.mean(costs_bikes[r_idx])),
                "max_morning_failure_rate_mean": float(
                    np.mean(max_failure_rates[:, 0])
                ),
                "max_evening_failure_rate_mean": float(
                    np.mean(max_failure_rates[:, 1])
                ),
            }
        )
    return rows


def save_scale_results(
    results_dir,
    num_seeds,
    bf_token,
    gini_values_tot,
    costs_tot,
    constraint_satisfaction,
    max_failure_rates_per_period,
    failure_rates_per_cat_period,
    costs_rebalancing,
    costs_failures,
    costs_bikes,
    initial_bikes,
):
    os.makedirs(results_dir, exist_ok=True)
    np.save(
        os.path.join(results_dir, f"gini_{num_seeds}seeds_{bf_token}.npy"),
        gini_values_tot,
    )
    np.save(
        os.path.join(results_dir, f"cost_{num_seeds}seeds_{bf_token}.npy"),
        costs_tot,
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


def save_demand_scale_summary(results_dir, bf_token, summary_rows):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"demand_scale_summary_{bf_token}.csv")
    fieldnames = [
        "demand_scale",
        "r_max",
        "gini_mean",
        "gini_std",
        "cost_mean",
        "cost_std",
        "constraint_satisfaction_rate",
        "reb_cost_mean",
        "failure_rate_mean",
        "bikes_mean",
        "max_morning_failure_rate_mean",
        "max_evening_failure_rate_mean",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    np.savez_compressed(
        os.path.join(results_dir, f"demand_scale_summary_{bf_token}.npz"),
        demand_scale=np.asarray([row["demand_scale"] for row in summary_rows]),
        r_max=np.asarray([row["r_max"] for row in summary_rows]),
        gini_mean=np.asarray([row["gini_mean"] for row in summary_rows]),
        cost_mean=np.asarray([row["cost_mean"] for row in summary_rows]),
        constraint_satisfaction_rate=np.asarray(
            [row["constraint_satisfaction_rate"] for row in summary_rows]
        ),
        failure_rate_mean=np.asarray(
            [row["failure_rate_mean"] for row in summary_rows]
        ),
        reb_cost_mean=np.asarray([row["reb_cost_mean"] for row in summary_rows]),
    )
    return csv_path


def evaluate_demand_scale(
    demand_scale,
    r_max_values,
    seeds,
    constrained_cats_arg,
    failure_cost_coef,
):
    bf_token = f"bf{fmt_token(failure_cost_coef)}"
    cat_dirname = "cat5"
    demand_dirname = f"scale{fmt_token(demand_scale)}"
    scenario = build_den_haag_scenario(demand_scale=demand_scale)
    node_list = scenario["node_list"]
    active_cats = scenario["active_cats"]
    demand_params = scenario["demand_params"]
    station_params = scenario["station_params"]
    constrained_cats = set(
        constrained_cats_arg if constrained_cats_arg is not None else active_cats
    )

    num_stations = sum(node_list)
    boundaries = scenario["boundaries"]

    num_seeds = len(seeds)

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
        print(f"\nEvaluating r_max = {r_max}, demand_scale = {demand_scale}")

        failure_thresholds = compute_failure_thresholds(
            r_max, demand_params, active_cats, constrained_cats
        )

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            seed_results_dir = os.path.join(
                SCRIPT_DIR, "results", cat_dirname, demand_dirname, f"seed{seed}"
            )
            seed_qtables_dir = os.path.join(
                SCRIPT_DIR, "q_tables", cat_dirname, demand_dirname, f"seed{seed}"
            )
            r_token = f"r{fmt_token(r_max)}"
            n_bikes = np.load(
                require_file(
                    os.path.join(seed_results_dir, f"bikes_{r_token}_{bf_token}.npy")
                )
            )

            graph = build_den_haag_network(scenario)
            all_days_demand, transformed_demand = generate_separate_event_demand(
                node_list, NUM_EVAL_DAYS, demand_params, TIME_SLOTS
            )

            agents = {}
            for cat in active_cats:
                agent = ZoneRebalancingAgent(cat)
                with open(
                    require_file(
                        os.path.join(
                            seed_qtables_dir,
                            f"q_table_{r_token}_{bf_token}_cat{cat}.pkl",
                        )
                    ),
                    "rb",
                ) as file:
                    agent.q_table = pickle.load(file)
                agent.set_epsilon(0.0)
                agents[cat] = agent

            eval_env = ZoneCMDPEnv(graph, transformed_demand, {}, GAMMA, station_params)
            state = eval_env.reset()

            daily_cat_failures = {cat: [] for cat in active_cats}
            daily_global_failures = []
            daily_global_costs = []

            period_cat_failures = {}
            for cat in active_cats:
                period_cat_failures[cat] = {0: [], 1: []}

            for day in range(NUM_EVAL_DAYS):
                cat_fails = {cat: 0 for cat in active_cats}
                global_fails = 0
                costs = 0
                period_fails_today = {
                    cat: {0: 0.0, 1: 0.0} for cat in period_cat_failures
                }

                for _time_period in (0, 1):
                    actions = np.zeros(num_stations, dtype=float)
                    if day > 0:
                        for station in range(num_stations):
                            cat = graph.nodes[station]["station"]
                            actions[station] = agents[cat].decide_action(state[station])

                    next_state, _reward, _base_reward, failures, reb_costs = (
                        eval_env.step(actions)
                    )
                    period = eval_env.current_period

                    for idx, cat in enumerate(active_cats):
                        cat_fails[cat] += np.sum(
                            failures[boundaries[idx] : boundaries[idx + 1]]
                        )
                    global_fails += np.sum(failures)

                    for idx, cat in enumerate(active_cats):
                        pf = (
                            np.sum(failures[boundaries[idx] : boundaries[idx + 1]])
                            / node_list[idx]
                        )
                        period_fails_today[cat][period] += pf

                    costs += float(np.sum(reb_costs) / GAMMA)

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
                        sum(graph.nodes[i]["bikes"] for i in range(num_stations))
                    )

            cat_requests = {cat: 0 for cat in active_cats}
            global_requests = 0

            for day in range(NUM_EVAL_DAYS):
                for hour in range(24):
                    for station in range(num_stations):
                        departures = all_days_demand[day]["departures"][station][hour]
                        for idx, cat in enumerate(active_cats):
                            if boundaries[idx] <= station < boundaries[idx + 1]:
                                cat_requests[cat] += departures
                                break
                        global_requests += departures

            for idx, cat in enumerate(active_cats):
                cat_requests[cat] = cat_requests[cat] / NUM_EVAL_DAYS / node_list[idx]
            global_requests = global_requests / NUM_EVAL_DAYS

            cat_failure_rates = {}
            for cat in active_cats:
                cat_failure_rates[cat] = (
                    np.mean(daily_cat_failures[cat]) / cat_requests[cat] * 100
                    if cat_requests[cat] > 0
                    else 0.0
                )
            failure_rate_global = (
                np.mean(daily_global_failures) / global_requests * 100
                if global_requests > 0
                else 0.0
            )

            failure_rates_list = [
                cat_failure_rates[cat] for cat in reversed(active_cats)
            ]
            gini = np.round(ineq.gini(failure_rates_list), 3)

            total_cost = (
                np.mean(daily_global_costs) + n_bikes / 100 + failure_rate_global / 10
            )

            satisfaction_tol = 0.05
            satisfied = True
            for cat in failure_thresholds:
                for p in (0, 1):
                    avg_fail = np.mean(period_cat_failures[cat][p])
                    threshold = failure_thresholds[cat][p]
                    if avg_fail > threshold * (1 + satisfaction_tol):
                        satisfied = False

            gini_values_tot[r_idx].append(gini)
            costs_tot[r_idx].append(total_cost)
            constraint_satisfaction[r_idx].append(satisfied)
            costs_rebalancing[r_idx].append(np.mean(daily_global_costs))
            costs_failures[r_idx].append(failure_rate_global)
            costs_bikes[r_idx].append(n_bikes)

            seed_idx = seeds.index(seed)
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

            constraint_str = "SAT" if satisfied else "VIOL"
            print(
                f"Gini={gini:.3f}, Cost={total_cost:.2f}, Constraint={constraint_str}"
            )

    print("\n" + "=" * 60)
    print(f"Saving results for demand_scale = {demand_scale}...")
    results_dir = os.path.join(
        SCRIPT_DIR, "results", cat_dirname, demand_dirname, "eval"
    )
    save_scale_results(
        results_dir,
        num_seeds,
        bf_token,
        gini_values_tot,
        costs_tot,
        constraint_satisfaction,
        max_failure_rates_per_period,
        failure_rates_per_cat_period,
        costs_rebalancing,
        costs_failures,
        costs_bikes,
        initial_bikes,
    )
    return summarize_scale_results(
        demand_scale,
        r_max_values,
        gini_values_tot,
        costs_tot,
        constraint_satisfaction,
        costs_rebalancing,
        costs_failures,
        costs_bikes,
        max_failure_rates_per_period,
    )


def main():
    parser = argparse.ArgumentParser()
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
        default=R_MAX_VALUES,
        help="r_max values to evaluate",
    )
    parser.add_argument(
        "--failure-cost-coef",
        type=float,
        default=0.0,
        help="Base failure coefficient token used in training filenames",
    )
    parser.add_argument(
        "--demand-scales",
        nargs="+",
        type=float,
        default=DEMAND_SCALES,
        help="Demand scales used during Den Haag CMDP training",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1]))
    summary_rows = []

    for demand_scale in args.demand_scales:
        summary_rows.extend(
            evaluate_demand_scale(
                demand_scale,
                args.r_max_values,
                seeds,
                args.constrained_cats,
                args.failure_cost_coef,
            )
        )

    summary_path = save_demand_scale_summary(
        os.path.join(SCRIPT_DIR, "results", "cat5", "eval"),
        f"bf{fmt_token(args.failure_cost_coef)}",
        summary_rows,
    )
    print(f"Saved demand-scale summary: {summary_path}")


if __name__ == "__main__":
    main()
