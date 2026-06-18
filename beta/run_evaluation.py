"""
Launch beta evaluation sweep across all configured categories.

Each evaluation.py call sweeps every beta value internally, so categories
are the unit launched in parallel here.

Usage:
    uv run beta/run_evaluation.py
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

seeds = [100, 110]
categories = [5, 4, 3, 2]

print(f"Starting beta evaluation (parallel: {len(categories)} categories)...")

evaluation_script = os.path.join(SCRIPT_DIR, "evaluation.py")

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
    ]

    print(f"  Launching categories={c}")
    proc = subprocess.Popen(cmd)
    processes.append((proc, c))

print(f"  Waiting for {len(processes)} processes...")

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

print("All categories done.\n")
