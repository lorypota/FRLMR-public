"""
CMDP training script.

Usage:
    uv run cmdp/training.py --r-max 0.075 --categories 5 --seed 100
    uv run cmdp/training.py --r-max 0.075 --categories 5 --seed 100 --failure-cost-coef 1.0
"""

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
from cmdp.environment import CMDPEnv
from common.agent import RebalancingAgent
from common.config import (
    CPU_CORES,
    GAMMA,
    NUM_TRAIN_DAYS,
    TIME_SLOTS,
    get_scenario,
)
from common.demand import generate_global_demand
from common.network import generate_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--r-max", default=0.15, type=float)
parser.add_argument("--categories", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--output-dir", default=SCRIPT_DIR, type=str, help="Output directory for results"
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
    help="Number of training repeats (default: 100)",
)
parser.add_argument(
    "--run-group", default=None, type=str, help="Wandb group ID for grouping runs"
)
parser.add_argument("--cpu-cores", default=CPU_CORES, type=str, help="CPU core range")
parser.add_argument(
    "--failure-cost-coef",
    default=0.0,
    type=float,
    help="Coefficient for base failure penalty in CMDP reward (default: 0.0)",
)
args = parser.parse_args()

# Limit resource usage app-reken12
# os.system(f"procgov64 --nowait --minws 10M --maxws {MAX_MEMORY_MB}M -p {os.getpid()}")
p = psutil.Process()
p.cpu_affinity(
    list(
        range(int(args.cpu_cores.split("-")[0]), int(args.cpu_cores.split("-")[1]) + 1)
    )
)

r_max = args.r_max
eta = args.eta
n_dual = args.n_dual
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
failure_cost_coef = args.failure_cost_coef


# =============================================================================
# LOAD SCENARIO CONFIG
# =============================================================================

scenario = get_scenario(args.categories)
demand_params = scenario["demand_params"]
node_list = scenario["node_list"]
active_cats = scenario["active_cats"]
station_params = scenario["station_params"]
r_token = f"r{fmt_token(r_max)}"
bf_token = f"bf{fmt_token(failure_cost_coef)}"
cat_dirname = f"cat{args.categories}"
seed_dirname = f"seed{args.seed}"
boundaries = scenario["boundaries"]
q_tables_dir = os.path.join(output_dir, "q_tables", cat_dirname, seed_dirname)
results_dir = os.path.join(output_dir, "results", cat_dirname, seed_dirname)
os.makedirs(q_tables_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

constrained_cats = set(
    args.constrained_cats if args.constrained_cats is not None else active_cats
)

# =============================================================================
# DUAL VARIABLE SETUP
# =============================================================================

# Initialize dual variables (lambdas) for constrained categories only
lambdas = {cat: [0.0, 0.0] for cat in active_cats if cat in constrained_cats}
failure_thresholds = compute_failure_thresholds(
    r_max, demand_params, active_cats, constrained_cats
)

# Failure accumulator and tracking
failure_accumulator = {cat: [0.0, 0.0] for cat in lambdas}
# Track failures for ALL active categories (not just constrained) for plotting
all_cat_failure_accumulator = {cat: [0.0, 0.0] for cat in active_cats}
base_return_accumulator = 0.0
reb_cost_accumulator = 0.0
day_counter = 0
lambda_history = []
dual_history = []  # (repeat, day, {cat: [morn_f_hat, eve_f_hat]}, avg_base_return)

# =============================================================================
# WANDB
# =============================================================================

wandb.init(
    project="fairmss",
    group=f"cmdp-{args.categories}cat-{args.run_group}",
    name=f"rmax{r_max}_bf{failure_cost_coef}_seed{args.seed}",
    config={
        "method": "cmdp",
        "r_max": r_max,
        "failure_cost_coef": failure_cost_coef,
        "categories": args.categories,
        "seed": args.seed,
        "eta": eta,
        "n_dual": n_dual,
        "constrained_cats": list(constrained_cats),
        "gamma": GAMMA,
        "num_repeats": args.num_repeats,
        "num_train_days": NUM_TRAIN_DAYS,
        "node_list": node_list,
        "active_cats": active_cats,
    },
)

# =============================================================================
# SETUP
# =============================================================================

agents = {i: RebalancingAgent(i) for i in range(5)}

G = generate_network(node_list)
np.random.seed(args.seed)
random.seed(args.seed)
_all_days_demand_vectors, transformed_demand_vectors = generate_global_demand(
    node_list, NUM_TRAIN_DAYS, demand_params, TIME_SLOTS
)

num_stations = np.sum(node_list)
# CMDP training is non-stationary because dual variables keep changing.
# Keep all active category agents trainable for the full CMDP run.
cmdp_train_until = {cat: args.num_repeats for cat in active_cats}
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

env = CMDPEnv(
    G,
    transformed_demand_vectors,
    lambdas,
    GAMMA,
    station_params,
    failure_cost_coef=failure_cost_coef,
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
            actions = np.zeros(num_stations, dtype=np.int64)
            if not (repeat == 0 and day == 0):
                for i in range(num_stations):
                    cat = G.nodes[i]["station"]
                    actions[i] = agents[cat].decide_action(state[i])

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
                    cat = G.nodes[i]["station"]
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
        if day_counter >= n_dual:
            # Compute f_hat for all active categories (for plotting)
            all_cat_f_hat = {
                cat: [all_cat_failure_accumulator[cat][p] / n_dual for p in (0, 1)]
                for cat in active_cats
            }

            for cat in list(failure_accumulator.keys()):
                # Only update if this category is still training
                if repeat < cmdp_train_until[cat]:
                    for p in (0, 1):
                        f_hat = failure_accumulator[cat][p] / n_dual
                        f_bar = failure_thresholds[cat][p]
                        violation = f_hat - f_bar
                        pname = "morning" if p == 0 else "evening"
                        dual_update_info[f"dual/cat{cat}_{pname}_f_hat"] = f_hat
                        dual_update_info[f"dual/cat{cat}_{pname}_f_bar"] = f_bar
                        dual_update_info[f"dual/cat{cat}_{pname}_violation"] = violation
                        lambdas[cat][p] = max(0.0, lambdas[cat][p] + eta * violation)

            # Log snapshots
            lambda_history.append(
                (repeat, day, {c: list(v) for c, v in lambdas.items()})
            )
            dual_history.append(
                (
                    repeat,
                    day,
                    all_cat_f_hat,
                    base_return_accumulator / n_dual,
                    reb_cost_accumulator / n_dual,
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
                    int(sum(G.nodes[i]["bikes"] for i in range(start_idx, end_idx)))
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
                for p in (0, 1):
                    pname = "morning" if p == 0 else "evening"
                    lambda_d = demand_params[cat_idx][p][1]
                    per_area = cat_period_fails[cat].get(p, 0.0)
                    rate = per_area / (12 * lambda_d) if lambda_d > 0 else 0.0
                    log_dict[f"failure_rate/cat{cat}_{pname}"] = rate
            for cat, lam in lambdas.items():
                log_dict[f"lambda/cat{cat}_morning"] = lam[0]
                log_dict[f"lambda/cat{cat}_evening"] = lam[1]
            for cat in active_cats:
                log_dict[f"epsilon/cat{cat}"] = agents[cat].epsilon
            log_dict.update(dual_update_info)
            wandb.log(log_dict)

wandb.finish()
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

total_bikes = sum(G.nodes[i]["bikes"] for i in range(num_stations))
np.save(
    os.path.join(results_dir, f"bikes_{r_token}_{bf_token}.npy"),
    total_bikes,
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

with open(os.path.join(results_dir, f"meta_{r_token}_{bf_token}.json"), "w") as file:
    json.dump(
        {
            "args": vars(args),
            "tokens": {"r": r_token, "bf": bf_token},
            "scenario": {
                "categories": args.categories,
                "node_list": node_list,
                "active_cats": active_cats,
                "boundaries": boundaries.tolist(),
            },
            "constrained_cats": sorted(constrained_cats),
            "failure_thresholds": {
                str(cat): [float(v) for v in vals]
                for cat, vals in failure_thresholds.items()
            },
            "cmdp_train_until": {str(cat): v for cat, v in cmdp_train_until.items()},
        },
        file,
        indent=2,
    )

print(
    f"Finished simulation with seed: {args.seed}, categories: {args.categories} and r_max: {r_max}"
)
print(f"Final lambdas: {dict(lambdas)}")
