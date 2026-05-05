"""
Baseline Training
======================================================

This script runs the Q-learning training for reproducing the Cederle et al. baseline.

Configuration:
- 2 categories (remote + central)
- 11 beta values (0.0 to 1.0)
- 3 seeds (100, 101, 102)
- Total: 33 training runs

Each run took approximately 4.6 minutes, total runtime of about 2.5 hours.

Output (saved to this folder):
    q_tables/q_table_{beta}_2_{seed}_cat{0,4}.pkl
    results/bikes_2_cat_{beta}_{seed}.npy
    results/learning_curve_2_cat_{beta}_{seed}.npy
"""

import os
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Using 3 seeds instead of 10 to reduce runtime
SEEDS = [100, 101, 102]

# Only 2-category case (simplest: remote vs central)
CATEGORIES = [2]

# All 11 beta values to trace the full Pareto front
# Beta controls fairness: 0.0 = impartial, 1.0 = maximum fairness priority
BETAS = range(0, 11)  # Will be divided by 10 in training.py to get 0.0-1.0

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Total runs: {len(SEEDS) * len(CATEGORIES) * len(BETAS)}")
    print("=" * 60)
    print()

    run_count = 0
    total_runs = len(SEEDS) * len(CATEGORIES) * len(BETAS)
    training_times = []

    for seed in SEEDS:
        for cat in CATEGORIES:
            for beta in BETAS:
                run_count += 1
                print(
                    f"[{run_count}/{total_runs}] Training: seed={seed}, categories={cat}, beta={beta / 10}"
                )

                start_time = time.time()
                training_script = os.path.join(
                    script_dir, "..", "..", "beta", "training.py"
                )
                cmd = f'uv run {training_script} --beta {beta} --categories {cat} --seed {seed} --output-dir "{script_dir}"'
                exit_code = os.system(cmd)
                elapsed = time.time() - start_time
                training_times.append(elapsed)

                if exit_code != 0:
                    print(
                        f"ERROR: Training failed for seed={seed}, cat={cat}, beta={beta}"
                    )
                else:
                    print(f"Completed in {elapsed:.1f} seconds")

    # Save training times
    times_file = os.path.join(script_dir, "training_times.txt")
    with open(times_file, "w") as f:
        f.write(
            f"Configuration: {CATEGORIES[0]} categories, {len(BETAS)} beta values, {len(SEEDS)} seeds ({', '.join(map(str, SEEDS))})\n"
        )
        f.write(f"Total runs: {total_runs}\n\n")
        f.write("Times in seconds per training run:\n")
        f.write("-" * 35 + "\n")
        for t in training_times:
            f.write(f"{t},\n")
        f.write("\nSummary Statistics:\n")
        f.write("-" * 19 + "\n")
        avg_time = sum(training_times) / len(training_times)
        f.write(
            f"Total training time: {sum(training_times):.2f} seconds (about {sum(training_times) / 3600:.1f} hours)\n"
        )
        f.write(
            f"Average per run: {avg_time:.2f} seconds (about {avg_time / 60:.1f} minutes)\n"
        )
        f.write(f"Min: {min(training_times):.2f} seconds\n")
        f.write(f"Max: {max(training_times):.2f} seconds\n")

    print()
    print("=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
