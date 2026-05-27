"""
Launch Den Haag CMDP training sweep across configured demand scales and r_max values.

Usage:
    uv run cmdp_den_haag_case/run_training.py
"""

import os
import subprocess
from datetime import datetime

from cmdp_den_haag_case.config import DEMAND_SCALES, R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
FAILURE_COST_COEFS = [0.0]

TOTAL_CORES = 20  # cores 0-19
CORES_PER_PROCESS = TOTAL_CORES // len(R_MAX_VALUES)

run_group_base = datetime.now().strftime("%Y%m%d_%H%M%S")

print(
    f"Starting Den Haag CMDP training (parallel: {len(R_MAX_VALUES)} r_max "
    f"values, {CORES_PER_PROCESS} cores each) for demand scales "
    f"{DEMAND_SCALES} and failure_cost_coef values {FAILURE_COST_COEFS}..."
)

training_script = os.path.join(SCRIPT_DIR, "training.py")

for demand_scale in DEMAND_SCALES:
    for failure_cost_coef in FAILURE_COST_COEFS:
        run_group = f"{run_group_base}_scale{demand_scale}_bf{failure_cost_coef}"
        print(
            f"\n=== demand_scale={demand_scale}, failure_cost_coef={failure_cost_coef} "
            f"(run_group={run_group}) ==="
        )
        for seed in seeds:
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
                    "--seed",
                    str(seed),
                    "--demand-scale",
                    str(demand_scale),
                    "--failure-cost-coef",
                    str(failure_cost_coef),
                    "--run-group",
                    run_group,
                    "--cpu-cores",
                    cpu_cores,
                ]

                print(
                    f"  Launching r_max={r_max}, demand_scale={demand_scale}, "
                    f"failure_cost_coef={failure_cost_coef} on cores {cpu_cores}"
                )
                proc = subprocess.Popen(cmd)
                processes.append((proc, r_max))

            print(
                f"  Waiting for {len(processes)} processes "
                f"(seed={seed}, demand_scale={demand_scale}, "
                f"bf={failure_cost_coef})..."
            )

            failed = False
            for proc, r_max in processes:
                exit_code = proc.wait()
                if exit_code != 0:
                    print(f"  !!! r_max={r_max} failed (exit code {exit_code})")
                    failed = True

            if failed:
                for proc, _ in processes:
                    proc.kill()
                print(
                    f"\n!!!!!!\n\n\n\n\nSEED {seed}, DEMAND_SCALE {demand_scale} "
                    f"FAILED!\n\n\n\n\n!!!!!!\n"
                )

            print(
                f"  Seed {seed} done for demand_scale={demand_scale}, "
                f"failure_cost_coef={failure_cost_coef}.\n"
            )
