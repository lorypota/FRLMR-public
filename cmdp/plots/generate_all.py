"""
Generate all CMDP plots for every category and base-failure setting
(failure_cost_coef 0.0 and 1.0) that has evaluation results.

Usage:
    uv run cmdp/plots/generate_all.py
"""

import glob
import os
import subprocess
import sys

from cmdp.config import fmt_token

categories = [2, 3, 4, 5]
failure_cost_coefs = [0.0, 1.0]
scripts = [
    "cmdp/plots/boxplots.py",
    "cmdp/plots/paretoplots.py",
    "cmdp/plots/lambda_convergence.py",
    "cmdp/plots/failure_rates_by_rmax.py",
]

generated = []
for bf in failure_cost_coefs:
    bf_token = f"bf{fmt_token(bf)}"
    for cat in categories:
        eval_dir = os.path.join("cmdp", "results", f"cat{cat}", "eval")
        if not glob.glob(os.path.join(eval_dir, f"gini_*seeds_{bf_token}.npy")):
            print(f"Skipping cat{cat} {bf_token}: no evaluation results.")
            continue
        for script in scripts:
            print(
                f"\n{'=' * 60}\nRunning {script} (cat{cat} {bf_token})...\n{'=' * 60}"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--categories",
                    str(cat),
                    "--save",
                    "--failure-cost-coef",
                    str(bf),
                ],
                env={**os.environ, "MPLBACKEND": "Agg"},
                check=False,
            )
            if result.returncode != 0:
                print(f"FAILED: {script} (cat{cat} {bf_token})")
        generated.append(f"cat{cat} {bf_token}")

print(
    f"\n{'=' * 60}\nGenerated plots for: {', '.join(generated) or 'none'}\n{'=' * 60}"
)
