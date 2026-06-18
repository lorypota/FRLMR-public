"""
Launch CMDP evaluation sweep across all configured categories.

Evaluates both base-failure settings (failure_cost_coef `0.0` and `1.0`).
Each evaluation.py call sweeps every r_max value internally, so categories
are the unit launched in parallel here.

Usage:
    uv run cmdp/run_evaluation.py
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = [100, 110]
categories = [5, 4, 3, 2]
FAILURE_COST_COEFS = [0.0, 1.0]

print(
    f"Starting CMDP evaluation (parallel: {len(categories)} categories) "
    f"for failure_cost_coef values {FAILURE_COST_COEFS}..."
)

evaluation_script = os.path.join(SCRIPT_DIR, "evaluation.py")

for failure_cost_coef in FAILURE_COST_COEFS:
    print(f"\n=== failure_cost_coef={failure_cost_coef} ===")
    processes = []

    for c in categories:
        cmd = [
            "uv",
            "run",
            evaluation_script,
            "--categories",
            str(c),
            "--seeds",
            str(seeds[0]),
            str(seeds[1]),
            "--failure-cost-coef",
            str(failure_cost_coef),
        ]

        print(f"  Launching categories={c}, failure_cost_coef={failure_cost_coef}")
        proc = subprocess.Popen(cmd)
        processes.append((proc, c))

    print(f"  Waiting for {len(processes)} processes (bf={failure_cost_coef})...")

    failed = False
    for proc, c in processes:
        exit_code = proc.wait()
        if exit_code != 0:
            print(f"  !!! categories={c} failed (exit code {exit_code})")
            failed = True

    if failed:
        for proc, _ in processes:
            proc.kill()
        print("Aborting due to failure.")
        sys.exit(1)

    print(f"  All categories done for failure_cost_coef={failure_cost_coef}.\n")
