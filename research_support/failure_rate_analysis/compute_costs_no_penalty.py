"""
Compute Costs Without Failure Penalty
======================================

This script computes the rebalancing + bike costs WITHOUT the failure penalty.
Uses the same evaluation logic as evaluate_baseline.py but saves costs separately.

Output:
    results/costs_no_penalty_2_cat_3seeds.npy - (11 betas x 3 seeds)
"""

import os
import pickle
import random

import numpy as np

from beta.config import BETAS
from beta.environment import FairEnv
from common.agent import RebalancingAgent
from common.config import GAMMA, NUM_EVAL_DAYS, PHI, TIME_SLOTS, get_scenario
from common.demand import generate_global_demand
from common.network import generate_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [100, 101, 102]

scenario = get_scenario(2)
NODE_LIST = scenario["node_list"]
ACTIVE_CATS = scenario["active_cats"]
DEMAND_PARAMS = scenario["demand_params"]
STATION_PARAMS = scenario["station_params"]

NUM_REMOTE = NODE_LIST[0]  # Category 0 stations (underserved areas)
NUM_CENTRAL = NODE_LIST[1]  # Category 4 stations (well-served areas)
NUM_STATIONS = NUM_REMOTE + NUM_CENTRAL

# Path to baseline results
BASELINE_DIR = os.path.join(SCRIPT_DIR, "..", "baseline")

# =============================================================================
# MAIN
# =============================================================================


def main():
    costs_no_penalty = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)
        print(f"\nEvaluating beta = {beta}")

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")

            np.random.seed(seed)
            random.seed(seed)

            # Load number of bikes
            bikes_file = os.path.join(
                BASELINE_DIR, "results", f"bikes_2_cat_{beta}_{seed}.npy"
            )
            n_bikes = np.load(bikes_file)

            # Generate network and demand
            G = generate_network(NODE_LIST)
            all_days_demand, transformed_demand = generate_global_demand(
                NODE_LIST, NUM_EVAL_DAYS, DEMAND_PARAMS, TIME_SLOTS
            )

            # Load trained Q-tables
            agent_remote = RebalancingAgent(0)
            agent_central = RebalancingAgent(4)

            q_table_remote = os.path.join(
                BASELINE_DIR, "q_tables", f"q_table_{beta}_2_{seed}_cat0.pkl"
            )
            q_table_central = os.path.join(
                BASELINE_DIR, "q_tables", f"q_table_{beta}_2_{seed}_cat4.pkl"
            )
            with open(q_table_remote, "rb") as f:
                agent_remote.q_table = pickle.load(f)
            with open(q_table_central, "rb") as f:
                agent_central.q_table = pickle.load(f)

            agent_remote.set_epsilon(0.0)
            agent_central.set_epsilon(0.0)

            # Initialize environment
            eval_env = FairEnv(G, transformed_demand, beta, GAMMA, STATION_PARAMS)
            state = eval_env.reset()

            daily_rebalancing_costs = []

            # Run evaluation
            for day in range(NUM_EVAL_DAYS):
                costs = 0

                for _ in (0, 1):
                    actions = np.zeros(NUM_STATIONS, dtype=np.int64)
                    if day > 0:
                        for station in range(NUM_STATIONS):
                            if G.nodes[station]["station"] == 0:
                                actions[station] = agent_remote.decide_action(
                                    state[station]
                                )
                            else:
                                actions[station] = agent_central.decide_action(
                                    state[station]
                                )

                    next_state, reward, failures = eval_env.step(actions)

                    # Calculate rebalancing costs only
                    for station, action in enumerate(actions):
                        if action != 0:
                            if station < NUM_REMOTE:
                                costs += PHI[0]
                            else:
                                costs += PHI[4]

                    state = next_state

                if day > 0:
                    daily_rebalancing_costs.append(costs)

            # Cost = rebalancing_cost + bike_cost (NO failure penalty)
            total_cost = np.mean(daily_rebalancing_costs) + n_bikes / 100

            costs_no_penalty[beta_index].append(total_cost)
            print(f"Cost (no penalty)={total_cost:.2f}")

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "costs_no_penalty_2_cat_3seeds.npy")
    np.save(output_file, costs_no_penalty)
    print(f"\nSaved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (Costs without failure penalty):")
    print("-" * 40)
    for i, beta in enumerate(BETAS):
        mean_cost = np.mean(costs_no_penalty[i])
        std_cost = np.std(costs_no_penalty[i])
        print(f"Beta {beta:.1f}: {mean_cost:.2f} ± {std_cost:.2f}")


if __name__ == "__main__":
    main()
