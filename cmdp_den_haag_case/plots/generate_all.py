"""Generate all Den Haag CMDP plots.

Usage:
    uv run cmdp_den_haag_case/plots/generate_all.py
    uv run cmdp_den_haag_case/plots/generate_all.py --demand-scales 0.005 0.01

"""

import argparse
import os
import subprocess
import sys

scripts = [
    "cmdp_den_haag_case/plots/boxplots.py",
    "cmdp_den_haag_case/plots/paretoplots.py",
    "cmdp_den_haag_case/plots/failure_rates_by_rmax.py",
    "cmdp_den_haag_case/plots/lambda_convergence.py",
    "cmdp_den_haag_case/plots/demand_scale_comparison.py",
]

parser = argparse.ArgumentParser()
parser.add_argument("--demand-scales", nargs="+", type=float, default=None)
parser.add_argument("--failure-cost-coef", type=float, default=0.0)
args = parser.parse_args()

for script in scripts:
    print(f"\n{'=' * 60}\nRunning {script}...\n{'=' * 60}")
    cmd = [
        sys.executable,
        script,
        "--save",
        "--failure-cost-coef",
        str(args.failure_cost_coef),
    ]
    if (
        script != "cmdp_den_haag_case/plots/demand_scale_comparison.py"
        and args.demand_scales is not None
    ):
        cmd.append("--demand-scales")
        cmd.extend(str(scale) for scale in args.demand_scales)
    result = subprocess.run(
        cmd,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=False,
    )
    if result.returncode != 0:
        print(f"FAILED: {script}")

print(f"\n{'=' * 60}\nAll done.\n{'=' * 60}")
