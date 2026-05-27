"""
Empirical Den Haag CMDP training script.

Usage:
    uv run cmdp_den_haag_case/training.py --r-max 0.15 --seed 100
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import psutil

import wandb
from cmdp.config import compute_failure_thresholds, fmt_token
from cmdp_den_haag_case.config import build_den_haag_network, build_den_haag_scenario
from cmdp_den_haag_case.zone_model import (
    ZoneCMDPEnv,
    ZoneRebalancingAgent,
    generate_separate_event_demand,
)
from common.config import CPU_CORES, GAMMA, NUM_TRAIN_DAYS, TIME_SLOTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-max", default=0.15, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--demand-scale",
        default=1.0,
        type=float,
        help="Multiplier applied after loading ODiN lambdas.",
    )
    parser.add_argument(
        "--constrained-cats",
        nargs="+",
        type=int,
        default=None,
        help="Category indices to constrain (default: all active categories)",
    )
    parser.add_argument("--eta", default=0.1, type=float, help="Dual step size")
    parser.add_argument(
        "--n-dual", default=100, type=int, help="Days between dual variable updates"
    )
    parser.add_argument(
        "--num-repeats",
        default=100,
        type=int,
        help="Number of training repeats",
    )
    parser.add_argument(
        "--run-group", default=None, type=str, help="Wandb group ID for grouping runs"
    )
    parser.add_argument(
        "--cpu-cores", default=CPU_CORES, type=str, help="CPU core range"
    )
    parser.add_argument(
        "--failure-cost-coef",
        default=0.0,
        type=float,
        help="Coefficient for base failure penalty in CMDP reward",
    )
    return parser.parse_args()


def apply_cpu_affinity(cpu_cores: str) -> None:
    try:
        start_core, end_core = (int(part) for part in cpu_cores.split("-"))
        available = psutil.cpu_count() or 1
        selected = [
            core for core in range(start_core, end_core + 1) if core < available
        ]
        if selected:
            psutil.Process().cpu_affinity(selected)
    except (AttributeError, NotImplementedError, OSError, ValueError) as exc:
        print(f"Skipping CPU affinity setup: {exc}")


def main() -> None:
    args = parse_args()
    apply_cpu_affinity(args.cpu_cores)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # LOAD SCENARIO CONFIG
    # =============================================================================

    scenario = build_den_haag_scenario(
        demand_scale=args.demand_scale,
    )
    demand_params = scenario["demand_params"]
    zone_demand_params = scenario["zone_demand_params"]
    node_list = scenario["node_list"]
    active_cats = scenario["active_cats"]
    station_params = scenario["station_params"]
    boundaries = scenario["boundaries"]

    r_token = f"r{fmt_token(args.r_max)}"
    bf_token = f"bf{fmt_token(args.failure_cost_coef)}"
    cat_dirname = "cat5"
    demand_dirname = f"scale{fmt_token(args.demand_scale)}"
    seed_dirname = f"seed{args.seed}"
    q_tables_dir = os.path.join(
        output_dir, "q_tables", cat_dirname, demand_dirname, seed_dirname
    )
    results_dir = os.path.join(
        output_dir, "results", cat_dirname, demand_dirname, seed_dirname
    )
    os.makedirs(q_tables_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    constrained_cats = set(
        args.constrained_cats if args.constrained_cats is not None else active_cats
    )

    # =============================================================================
    # DUAL VARIABLE SETUP
    # =============================================================================

    lambdas = {cat: [0.0, 0.0] for cat in active_cats if cat in constrained_cats}
    failure_thresholds = compute_failure_thresholds(
        args.r_max, demand_params, active_cats, constrained_cats
    )
    base_epsilon_decay = ZoneRebalancingAgent(0).epsilon_decay
    epsilon_decay_by_category = {
        cat: base_epsilon_decay
        * scenario["reference_station_counts_by_category"][cat]
        / node_list[cat_idx]
        for cat_idx, cat in enumerate(active_cats)
    }

    # =============================================================================
    # WANDB
    # =============================================================================

    wandb_run = wandb.init(
        project="fairmss",
        group=f"cmdp-den-haag-cat5-{args.run_group}",
        name=(
            f"denhaag_{scenario['demand_year_group']}_rmax{args.r_max}"
            f"_bf{args.failure_cost_coef}_seed{args.seed}"
        ),
        config={
            "method": "cmdp_den_haag_case",
            "r_max": args.r_max,
            "failure_cost_coef": args.failure_cost_coef,
            "categories": 5,
            "seed": args.seed,
            "eta": args.eta,
            "n_dual": args.n_dual,
            "constrained_cats": list(constrained_cats),
            "gamma": GAMMA,
            "num_repeats": args.num_repeats,
            "num_train_days": NUM_TRAIN_DAYS,
            "node_list": node_list,
            "active_cats": active_cats,
            "demand_year_group": scenario["demand_year_group"],
            "demand_scale": args.demand_scale,
            "epsilon_decay_by_category": {
                str(cat): value for cat, value in epsilon_decay_by_category.items()
            },
        },
    )

    # =============================================================================
    # SETUP
    # =============================================================================

    agents = {
        cat: ZoneRebalancingAgent(cat, epsilon_decay=epsilon_decay_by_category[cat])
        for cat in active_cats
    }
    graph = build_den_haag_network(scenario)
    np.random.seed(args.seed)
    random.seed(args.seed)
    _all_days_demand_vectors, transformed_demand_vectors = (
        generate_separate_event_demand(
            node_list, NUM_TRAIN_DAYS, zone_demand_params, TIME_SLOTS
        )
    )

    num_stations = int(np.sum(node_list))
    cmdp_train_until = {cat: args.num_repeats for cat in active_cats}
    failure_accumulator = {cat: [0.0, 0.0] for cat in lambdas}
    all_cat_failure_accumulator = {cat: [0.0, 0.0] for cat in active_cats}
    base_return_accumulator = 0.0
    reb_cost_accumulator = 0.0
    day_counter = 0
    lambda_history = []
    dual_history = []

    daily_returns = []
    daily_base_returns = []
    daily_failures = []
    daily_reb_costs = []
    daily_total_bikes = []
    daily_nonzero_actions = []
    daily_mean_abs_action = []
    daily_cat_failures = []
    daily_cat_period_failures = []
    daily_cat_bikes = []
    daily_cat_nonzero_actions = []
    daily_cat_mean_abs_action = []

    env = ZoneCMDPEnv(
        graph,
        transformed_demand_vectors,
        lambdas,
        GAMMA,
        station_params,
        failure_cost_coef=args.failure_cost_coef,
    )
    state = env.reset()

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================

    global_step = 0
    start = time.time()
    for repeat in range(args.num_repeats):
        for day in range(NUM_TRAIN_DAYS):
            ret = 0
            base_ret = 0
            reb_ret = 0
            fails = 0
            cat_daily_fails = {cat: 0.0 for cat in active_cats}
            cat_period_fails = {cat: {} for cat in active_cats}
            cat_daily_nonzero_actions = {cat: 0 for cat in active_cats}
            cat_daily_abs_actions = {cat: 0.0 for cat in active_cats}
            nonzero_actions_day = 0
            abs_actions_sum_day = 0.0

            for _time_period in (0, 1):
                actions = np.zeros(num_stations, dtype=float)
                if not (repeat == 0 and day == 0):
                    for station in range(num_stations):
                        cat = graph.nodes[station]["station"]
                        actions[station] = agents[cat].decide_action(state[station])

                nonzero_actions_day += int(np.count_nonzero(actions))
                abs_actions_sum_day += float(np.sum(np.abs(actions)))
                for cat_idx, cat in enumerate(active_cats):
                    start_idx = boundaries[cat_idx]
                    end_idx = boundaries[cat_idx + 1]
                    cat_actions = actions[start_idx:end_idx]
                    cat_daily_nonzero_actions[cat] += int(np.count_nonzero(cat_actions))
                    cat_daily_abs_actions[cat] += float(np.sum(np.abs(cat_actions)))

                next_state, reward, base_reward, failures, reb_costs = env.step(actions)
                ret += np.sum(reward)
                base_ret += np.sum(base_reward)
                reb_ret += np.sum(reb_costs)
                fails += np.sum(failures)

                # Accumulate per-category per-period failures for dual update
                period = env.current_period
                for cat_idx, cat in enumerate(active_cats):
                    cat_failures = np.sum(
                        failures[boundaries[cat_idx] : boundaries[cat_idx + 1]]
                    )
                    cat_daily_fails[cat] += cat_failures
                    cat_period_fails[cat][period] = cat_failures / node_list[cat_idx]
                    # Track all categories for plotting
                    all_cat_failure_accumulator[cat][period] += (
                        cat_failures / node_list[cat_idx]
                    )
                    if cat in failure_accumulator:
                        # Normalize by number of areas in this category
                        failure_accumulator[cat][period] += (
                            cat_failures / node_list[cat_idx]
                        )

                if not (day == 0 and repeat == 0):
                    for station in range(num_stations):
                        cat = graph.nodes[station]["station"]
                        if repeat < cmdp_train_until[cat]:
                            agents[cat].update_q_table(
                                state[station],
                                actions[station],
                                reward[station],
                                next_state[station],
                            )
                            agents[cat].update_epsilon()

                state = next_state

            # Dual variable update every n_dual days
            dual_update_info = {}
            day_counter += 1
            base_return_accumulator += base_ret
            reb_cost_accumulator += reb_ret
            if day_counter >= args.n_dual:
                # Compute f_hat for all active categories (for plotting)
                all_cat_f_hat = {
                    cat: [
                        all_cat_failure_accumulator[cat][p] / args.n_dual
                        for p in (0, 1)
                    ]
                    for cat in active_cats
                }
                for cat in list(failure_accumulator.keys()):
                    # Only update if this category is still training
                    if repeat < cmdp_train_until[cat]:
                        for period in (0, 1):
                            f_hat = failure_accumulator[cat][period] / args.n_dual
                            f_bar = failure_thresholds[cat][period]
                            violation = f_hat - f_bar
                            pname = "morning" if period == 0 else "evening"
                            dual_update_info[f"dual/cat{cat}_{pname}_f_hat"] = f_hat
                            dual_update_info[f"dual/cat{cat}_{pname}_f_bar"] = f_bar
                            dual_update_info[f"dual/cat{cat}_{pname}_violation"] = (
                                violation
                            )
                            lambdas[cat][period] = float(
                                max(0.0, lambdas[cat][period] + args.eta * violation)
                            )

                # Log snapshots
                lambda_history.append(
                    (
                        repeat,
                        day,
                        {cat: list(values) for cat, values in lambdas.items()},
                    )
                )
                dual_history.append(
                    (
                        repeat,
                        day,
                        all_cat_f_hat,
                        base_return_accumulator / args.n_dual,
                        reb_cost_accumulator / args.n_dual,
                    )
                )

                # Reset accumulators
                failure_accumulator = {cat: [0.0, 0.0] for cat in lambdas}
                all_cat_failure_accumulator = {cat: [0.0, 0.0] for cat in active_cats}
                base_return_accumulator = 0.0
                reb_cost_accumulator = 0.0
                day_counter = 0

            if not (repeat == 0 and day == 0):
                global_step += 1
                daily_returns.append(ret)
                daily_base_returns.append(base_ret)
                daily_failures.append(fails)
                daily_reb_costs.append(reb_ret)
                daily_nonzero_actions.append(nonzero_actions_day)
                daily_mean_abs_action.append(abs_actions_sum_day / (2 * num_stations))
                daily_cat_failures.append([cat_daily_fails[cat] for cat in active_cats])
                daily_cat_period_failures.append(
                    [
                        [
                            cat_period_fails[cat].get(0, 0.0),
                            cat_period_fails[cat].get(1, 0.0),
                        ]
                        for cat in active_cats
                    ]
                )
                daily_cat_nonzero_actions.append(
                    [cat_daily_nonzero_actions[cat] for cat in active_cats]
                )
                daily_cat_mean_abs_action.append(
                    [
                        cat_daily_abs_actions[cat] / (2 * node_list[cat_idx])
                        for cat_idx, cat in enumerate(active_cats)
                    ]
                )
                cat_bikes_end = []
                for cat_idx, _cat in enumerate(active_cats):
                    start_idx = boundaries[cat_idx]
                    end_idx = boundaries[cat_idx + 1]
                    cat_bikes_end.append(
                        int(
                            sum(
                                graph.nodes[i]["bikes"]
                                for i in range(start_idx, end_idx)
                            )
                        )
                    )
                daily_cat_bikes.append(cat_bikes_end)
                daily_total_bikes.append(int(sum(cat_bikes_end)))

                # wandb logging
                log_dict = {
                    "repeat": repeat,
                    "day": day,
                    "global_step": global_step,
                    "elapsed_time": time.time() - start,
                    "daily_return": ret,
                    "daily_base_return": base_ret,
                    "daily_failures": fails,
                    "daily_reb_costs": reb_ret,
                    "daily_total_bikes": daily_total_bikes[-1],
                    "daily_nonzero_actions": nonzero_actions_day,
                    "daily_mean_abs_action": daily_mean_abs_action[-1],
                }
                for cat in active_cats:
                    log_dict[f"failures/cat{cat}"] = cat_daily_fails[cat]
                for cat_idx, cat in enumerate(active_cats):
                    for period in (0, 1):
                        pname = "morning" if period == 0 else "evening"
                        lambda_d = demand_params[cat_idx][period][1]
                        per_area = cat_period_fails[cat].get(period, 0.0)
                        rate = per_area / (12 * lambda_d) if lambda_d > 0 else 0.0
                        log_dict[f"failure_rate/cat{cat}_{pname}"] = rate
                for cat, lam in lambdas.items():
                    log_dict[f"lambda/cat{cat}_morning"] = lam[0]
                    log_dict[f"lambda/cat{cat}_evening"] = lam[1]
                for cat in active_cats:
                    log_dict[f"epsilon/cat{cat}"] = agents[cat].epsilon
                log_dict.update(dual_update_info)
                wandb.log(log_dict)

    wandb_run.finish()

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    for cat in active_cats:
        with open(
            os.path.join(q_tables_dir, f"q_table_{r_token}_{bf_token}_cat{cat}.pkl"),
            "wb",
        ) as file:
            pickle.dump(agents[cat].q_table, file)

    np.save(
        os.path.join(results_dir, f"learning_curve_{r_token}_{bf_token}.npy"),
        daily_returns,
    )
    np.save(
        os.path.join(results_dir, f"base_learning_curve_{r_token}_{bf_token}.npy"),
        daily_base_returns,
    )
    np.save(
        os.path.join(results_dir, f"bikes_{r_token}_{bf_token}.npy"),
        sum(graph.nodes[i]["bikes"] for i in range(num_stations)),
    )
    with open(
        os.path.join(results_dir, f"lambda_history_{r_token}_{bf_token}.pkl"),
        "wb",
    ) as file:
        pickle.dump(lambda_history, file)
    with open(
        os.path.join(results_dir, f"final_lambdas_{r_token}_{bf_token}.pkl"),
        "wb",
    ) as file:
        pickle.dump(dict(lambdas), file)
    with open(
        os.path.join(results_dir, f"dual_history_{r_token}_{bf_token}.pkl"),
        "wb",
    ) as file:
        pickle.dump(dual_history, file)

    np.savez_compressed(
        os.path.join(results_dir, f"train_diag_{r_token}_{bf_token}.npz"),
        daily_return=np.asarray(daily_returns),
        daily_base_return=np.asarray(daily_base_returns),
        daily_failures=np.asarray(daily_failures),
        daily_reb_costs=np.asarray(daily_reb_costs),
        daily_total_bikes=np.asarray(daily_total_bikes),
        daily_nonzero_actions=np.asarray(daily_nonzero_actions),
        daily_mean_abs_action=np.asarray(daily_mean_abs_action),
        daily_cat_failures=np.asarray(daily_cat_failures),
        daily_cat_period_failures=np.asarray(daily_cat_period_failures),
        daily_cat_bikes=np.asarray(daily_cat_bikes),
        daily_cat_nonzero_actions=np.asarray(daily_cat_nonzero_actions),
        daily_cat_mean_abs_action=np.asarray(daily_cat_mean_abs_action),
    )

    with open(
        os.path.join(results_dir, f"meta_{r_token}_{bf_token}.json"), "w"
    ) as file:
        json.dump(
            {
                "args": vars(args),
                "tokens": {"r": r_token, "bf": bf_token},
                "scenario": {
                    "categories": 5,
                    "node_list": node_list,
                    "reference_node_list": scenario["reference_node_list"],
                    "active_cats": active_cats,
                    "boundaries": boundaries.tolist(),
                    "demand_year_group": scenario["demand_year_group"],
                    "demand_rates_path": scenario["demand_rates_path"],
                    "station_assignments_path": scenario["station_assignments_path"],
                    "model_unit": scenario["model_unit"],
                    "demand_allocation": scenario["demand_allocation"],
                    "initial_bikes_path": scenario["initial_bikes_path"],
                    "initial_bikes_timestamp": scenario["initial_bikes_timestamp"],
                    "zone_records": scenario["zone_records"],
                    "zone_capacities": scenario["zone_capacities"],
                    "zone_initial_bikes": scenario["zone_initial_bikes"],
                    "zone_raw_initial_bikes": scenario["zone_raw_initial_bikes"],
                    "station_counts_by_category": {
                        str(cat): int(count)
                        for cat, count in scenario["station_counts_by_category"].items()
                    },
                    "station_capacity_sums_by_category": {
                        str(cat): float(capacity_sum)
                        for cat, capacity_sum in scenario[
                            "station_capacity_sums_by_category"
                        ].items()
                    },
                    "reference_station_counts_by_category": {
                        str(cat): int(count)
                        for cat, count in scenario[
                            "reference_station_counts_by_category"
                        ].items()
                    },
                    "epsilon_decay_by_category": {
                        str(cat): float(value)
                        for cat, value in epsilon_decay_by_category.items()
                    },
                    "demand_scale": scenario["demand_scale"],
                    "demand_generation": "separate_poisson_arrival_departure_events",
                    "demand_params": demand_params,
                    "zone_demand_params": zone_demand_params,
                    "raw_category_demand_params": scenario[
                        "raw_category_demand_params"
                    ],
                },
                "constrained_cats": sorted(constrained_cats),
                "failure_thresholds": {
                    str(cat): [float(v) for v in vals]
                    for cat, vals in failure_thresholds.items()
                },
                "cmdp_train_until": {
                    str(cat): value for cat, value in cmdp_train_until.items()
                },
            },
            file,
            indent=2,
        )

    print(
        f"Finished Den Haag CMDP training with seed {args.seed}, "
        f"year group {scenario['demand_year_group']}, and r_max {args.r_max}"
    )
    print(f"Final lambdas: {dict(lambdas)}")


if __name__ == "__main__":
    main()
