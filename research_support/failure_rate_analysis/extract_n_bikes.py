"""
Extract Number of Bikes in System
==================================

Extracts the total number of bikes (n_bikes) from baseline results.

Output:
    results/n_bikes_2_cat_3seeds.npy - (11 betas x 3 seeds)
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
    n_bikes_all = [[] for _ in range(len(BETAS))]

    for beta in BETAS:
        beta_index = int(beta * 10)

        for seed in SEEDS:
            # Load number of bikes
            bikes_file = os.path.join(
                BASELINE_DIR, "results", f"bikes_2_cat_{beta}_{seed}.npy"
            )
            n_bikes = np.load(bikes_file)

            n_bikes_all[beta_index].append(float(n_bikes))

            print(f"Beta {beta}, Seed {seed}: n_bikes = {n_bikes}")

    # Save results
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "n_bikes_2_cat_3seeds.npy")
    np.save(output_file, n_bikes_all)
    print(f"\nSaved: {output_file}")

    # Print summary
    print("\n" + "=" * 40)
    print("Summary (Number of bikes in system):")
    print("-" * 40)
    for i, beta in enumerate(BETAS):
        mean_bikes = np.mean(n_bikes_all[i])
        std_bikes = np.std(n_bikes_all[i])
        print(f"Beta {beta:.1f}: {mean_bikes:.0f} ± {std_bikes:.0f} bikes")


if __name__ == "__main__":
    main()
