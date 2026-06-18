"""
Launch CMDP training sweep across all configured r_max values.

Runs both base-failure settings (failure_cost_coef `1.0` and `0.0`).

Usage:
    uv run cmdp/run_training.py
"""

import os
import subprocess
from datetime import datetime

from cmdp.config import R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
categories = [5, 4, 3, 2]
FAILURE_COST_COEFS = [0.0, 1.0]

TOTAL_CORES = 20  # cores 0-19
CORES_PER_PROCESS = TOTAL_CORES // len(R_MAX_VALUES)

run_group_base = datetime.now().strftime("%Y%m%d_%H%M%S")

print(
    f"Starting CMDP training (parallel: {len(R_MAX_VALUES)} r_max values, "
    f"{CORES_PER_PROCESS} cores each) for failure_cost_coef values {FAILURE_COST_COEFS}..."
)

training_script = os.path.join(SCRIPT_DIR, "training.py")

for failure_cost_coef in FAILURE_COST_COEFS:
    run_group = f"{run_group_base}_bf{failure_cost_coef}"
    print(f"\n=== failure_cost_coef={failure_cost_coef} (run_group={run_group}) ===")
    for seed in seeds:
        for category in categories:
            processes = []

            for i, r_max in enumerate(R_MAX_VALUES):
                core_start = i * CORES_PER_PROCESS
                core_end = core_start + CORES_PER_PROCESS - 1
                cpu_cores = f"{core_start}-{core_end}"

                cmd = [
                    "uv",
                    "run",
                    training_script,
                    "--r-max",
                    str(r_max),
                    "--categories",
                    str(category),
                    "--seed",
                    str(seed),
                    "--failure-cost-coef",
                    str(failure_cost_coef),
                    "--run-group",
                    run_group,
                    "--cpu-cores",
                    cpu_cores,
                ]

                print(
                    f"  Launching r_max={r_max}, failure_cost_coef={failure_cost_coef} on cores {cpu_cores}"
                )
                proc = subprocess.Popen(cmd)
                processes.append((proc, r_max))

            print(
                f"  Waiting for {len(processes)} processes (seed={seed}, cat={category}, bf={failure_cost_coef})..."
            )

            failed = False
            for proc, r_max in processes:
                exit_code = proc.wait()
                if exit_code != 0:
                    print(f"  !!! r_max={r_max} failed (exit code {exit_code})")
                    failed = True

            if failed:
                # Kill any still-running processes
                for proc, _ in processes:
                    proc.kill()
                print(
                    f"\n!!!!!!\n\n\n\n\nSEED {seed}, CAT {category}, R_MAX {r_max} FAILED!\n\n\n\n\n!!!!!!\n"
                )
                # sys.exit(1)

            print(f"  Seed {seed} done for failure_cost_coef={failure_cost_coef}.\n")
