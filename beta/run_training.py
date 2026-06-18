"""
Launch beta training sweep across all configured beta values.

Usage:
    uv run beta/run_training.py
"""

import os
import subprocess
import sys
from datetime import datetime

from beta.config import BETAS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = range(100, 110)
categories = [5, 4, 3, 2]

TOTAL_CORES = 20  # cores 0-19
CORES_PER_PROCESS = TOTAL_CORES // len(BETAS)

run_group = datetime.now().strftime("%Y%m%d_%H%M%S")

print(
    f"Starting beta training (parallel: {len(BETAS)} beta values, "
    f"{CORES_PER_PROCESS} cores each)..."
)

training_script = os.path.join(SCRIPT_DIR, "training.py")

for s in seeds:
    for c in categories:
        processes = []

        for i, b in enumerate(BETAS):
            core_start = i * CORES_PER_PROCESS
            core_end = core_start + CORES_PER_PROCESS - 1
            cpu_cores = f"{core_start}-{core_end}"

            cmd = [
                "uv",
                "run",
                training_script,
                "--beta",
                str(b),
                "--categories",
                str(c),
                "--seed",
                str(s),
                "--run-group",
                run_group,
                "--cpu-cores",
                cpu_cores,
            ]

            print(f"  Launching beta={b} on cores {cpu_cores}")
            proc = subprocess.Popen(cmd)
            processes.append((proc, b))

        print(f"  Waiting for {len(processes)} processes (seed={s}, cat={c})...")

        failed = False
        for proc, b in processes:
            exit_code = proc.wait()
            if exit_code != 0:
                print(f"  !!! beta={b} failed (exit code {exit_code})")
                failed = True

        if failed:
            for proc, _ in processes:
                proc.kill()
            print("Aborting due to failure.")
            sys.exit(1)

        print(f"  Seed {s} done.\n")
