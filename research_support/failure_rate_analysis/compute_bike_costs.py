"""
Compute Bike Costs Only
========================

Extracts just the bike cost component: n_bikes / 100
Uses already-saved bikes data from baseline.

Output:
    results/bike_costs_2_cat_3seeds.npy - (11 betas x 3 seeds)
"""

import os

import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
SEEDS = [100, 101, 102]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Path to baseline results
BASELINE_DIR = os.path.join(SCRIPT_DIR, "..", "baseline")


def main():
    bike_costs = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)

        for seed in SEEDS:
            # Load number of bikes
            bikes_file = os.path.join(
                BASELINE_DIR, "results", f"bikes_2_cat_{beta}_{seed}.npy"
            )
            n_bikes = np.load(bikes_file)

            # Bike cost = n_bikes / 100
            bike_cost = n_bikes / 100
            bike_costs[beta_index].append(bike_cost)

            print(
                f"Beta {beta}, Seed {seed}: n_bikes={n_bikes}, bike_cost={bike_cost:.2f}"
            )

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "bike_costs_2_cat_3seeds.npy")
    np.save(output_file, bike_costs)
    print(f"\nSaved: {output_file}")

    # Print summary
    print("\n" + "=" * 40)
    print("Summary (Bike costs only = n_bikes/100):")
    print("-" * 40)
    for i, beta in enumerate(BETAS):
        mean_cost = np.mean(bike_costs[i])
        std_cost = np.std(bike_costs[i])
        print(f"Beta {beta:.1f}: {mean_cost:.2f} ± {std_cost:.2f}")


if __name__ == "__main__":
    main()
