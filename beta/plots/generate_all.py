"""
Generate all beta plots for every category that has evaluation results.

Usage:
    uv run beta/plots/generate_all.py
"""

import os
import subprocess
import sys

categories = [2, 3, 4, 5]
scripts = [
    "beta/plots/boxplots.py",
    "beta/plots/paretoplots.py",
    "beta/plots/learning_curves.py",
    "beta/plots/failure_rates_by_beta.py",
]

generated = []
for cat in categories:
    if not os.path.isdir(os.path.join("beta", "results", f"cat{cat}", "eval")):
        print(f"Skipping cat{cat}: no evaluation results.")
        continue
    for script in scripts:
        print(f"\n{'=' * 60}\nRunning {script} (cat{cat})...\n{'=' * 60}")
        result = subprocess.run(
            [sys.executable, script, "--categories", str(cat), "--save"],
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        if result.returncode != 0:
            print(f"FAILED: {script} (cat{cat})")
    generated.append(cat)

print(f"\n{'=' * 60}\nGenerated plots for categories: {generated}\n{'=' * 60}")
