"""
Generate all CMDP plots for a given category scenario.

Usage:
    uv run cmdp/plots/generate_all.py --categories 5 --failure-cost-coef 0.0
    uv run cmdp/plots/generate_all.py --categories 5 --failure-cost-coef 1.0
"""

import argparse
import subprocess
import sys

scripts = [
    "cmdp/plots/boxplots.py",
    "cmdp/plots/paretoplots.py",
    "cmdp/plots/lambda_convergence.py",
    "cmdp/plots/failure_rates_by_rmax.py",
]

parser = argparse.ArgumentParser()
parser.add_argument("--categories", required=True, type=int)
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()

for script in scripts:
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}")
    env = {"MPLBACKEND": "Agg"}
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--categories",
            str(args.categories),
            "--save",
            "--failure-cost-coef",
            str(args.failure_cost_coef),
        ],
        env={**__import__("os").environ, **env},
    )
    if result.returncode != 0:
        print(f"FAILED: {script}")

print(f"\n{'='*60}\nAll done.\n{'='*60}")
