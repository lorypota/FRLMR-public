"""Splintering urbanism analysis: coverage gaps between area categories.

Classifies buurten into terciles by demographic variables and compares
coverage between categories. Tests whether low-income or high-migration
areas have systematically worse bike coverage.

Run:
    uv run preliminary_studies/empirical_analysis/analysis_splintering.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from internal.cbs_income import fetch_income_data
from internal.coverage_utils import load_buurten
from internal.paths import (
    ANALYSIS_CACHE_DIR,
    ANALYSIS_FIGURES_DIR,
    ANALYSIS_TABLES_DIR,
    GEODATA_DIR,
    ensure_output_dirs,
)
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Variables to classify buurten by, with tercile direction
# "ascending" means tercile 1 = lowest values, tercile 3 = highest
CLASSIFICATION_VARS = {
    "percentage_met_herkomstland_buiten_europa": {
        "label": "Non-European background",
        "source": "buurten",
        "ascending": True,
        "tercile_labels": ["Low migration", "Medium migration", "High migration"],
    },
    "personenautos_per_huishouden": {
        "label": "Car ownership",
        "source": "buurten",
        "ascending": True,
        "tercile_labels": [
            "Low car ownership",
            "Medium car ownership",
            "High car ownership",
        ],
    },
    "gemiddeld_inkomen_per_inwoner": {
        "label": "Income",
        "source": "income",
        "ascending": True,
        "tercile_labels": ["Low income", "Medium income", "High income"],
    },
    "gemiddelde_woz_waarde": {
        "label": "Property value (WOZ)",
        "source": "income",
        "ascending": True,
        "tercile_labels": ["Low WOZ", "Medium WOZ", "High WOZ"],
    },
}


def classify_terciles(values):
    """Assign tercile labels (1, 2, 3) based on values. NaN stays NaN."""
    result = pd.Series(np.nan, index=values.index)
    valid = values.dropna()
    if len(valid) < 6:
        return result
    thresholds = [valid.quantile(1 / 3), valid.quantile(2 / 3)]
    result.loc[valid.index] = np.where(
        valid <= thresholds[0],
        1,
        np.where(valid <= thresholds[1], 2, 3),
    )
    return result


def run():
    ensure_output_dirs()

    logger.info("Loading data...")
    buurten = load_buurten()

    # Load cached per-buurt per-hour coverage
    cache_files = sorted(ANALYSIS_CACHE_DIR.glob("coverage_*.csv"))
    if not cache_files:
        logger.error("No cached coverage data. Run analysis_temporal.py first.")
        sys.exit(1)

    logger.info("Loading %d cached coverage files...", len(cache_files))
    coverage_df = pd.concat([pd.read_csv(f) for f in cache_files])

    # Load income data
    income_df = fetch_income_data(cache_dir=GEODATA_DIR)

    # Build buurt feature table
    buurt_features = pd.DataFrame({"buurt_idx": range(len(buurten))})
    buurt_features["buurtcode"] = buurten["buurtcode"].values

    for var, info in CLASSIFICATION_VARS.items():
        if info["source"] == "buurten" and var in buurten.columns:
            buurt_features[var] = buurten[var].values
        elif info["source"] == "income" and income_df is not None:
            merged = buurt_features[["buurt_idx", "buurtcode"]].merge(
                income_df[["buurtcode", var]], on="buurtcode", how="left"
            )
            buurt_features[var] = merged[var].values

    # Classify each buurt into terciles
    for var in CLASSIFICATION_VARS:
        if var in buurt_features.columns:
            buurt_features[f"{var}_tercile"] = classify_terciles(buurt_features[var])

    # Merge tercile labels into hourly coverage data
    coverage_with_class = coverage_df.merge(buurt_features, on="buurt_idx", how="left")

    all_test_results = []
    all_category_summaries = []

    for var, info in CLASSIFICATION_VARS.items():
        tercile_col = f"{var}_tercile"
        if tercile_col not in coverage_with_class.columns:
            continue
        if coverage_with_class[tercile_col].notna().sum() == 0:
            logger.warning("No tercile data for %s, skipping", var)
            continue

        logger.info("Analyzing: %s", info["label"])

        # Aggregate to per-buurt means first so tests use neighborhoods as units.
        buurt_means = (
            coverage_with_class[coverage_with_class[tercile_col].notna()]
            .groupby(["buurt_idx", tercile_col])
            .agg(
                mean_distance=("mean_distance", "mean"),
                median_distance=("mean_distance", "median"),
                std_distance=("mean_distance", "std"),
                mean_pct_500m=("pct_within_500m", "mean"),
            )
            .reset_index()
        )
        cat_summary = (
            buurt_means.groupby(tercile_col)
            .agg(
                mean_distance=("mean_distance", "mean"),
                median_distance=("mean_distance", "median"),
                std_distance=("mean_distance", "std"),
                mean_pct_500m=("mean_pct_500m", "mean"),
                n_buurten=("buurt_idx", "nunique"),
            )
            .reset_index()
        )
        cat_summary["n_observations"] = cat_summary["n_buurten"]
        cat_summary["variable"] = var
        cat_summary["tercile_label"] = [
            info["tercile_labels"][int(t) - 1] for t in cat_summary[tercile_col]
        ]
        all_category_summaries.append(cat_summary)

        # Statistical tests: compare tercile 1 vs tercile 3
        t1 = buurt_means[buurt_means[tercile_col] == 1]["mean_distance"].dropna()
        t3 = buurt_means[buurt_means[tercile_col] == 3]["mean_distance"].dropna()

        if len(t1) >= 5 and len(t3) >= 5:
            mw_stat, mw_p = stats.mannwhitneyu(t1, t3, alternative="two-sided")
            # Rank-biserial correlation as effect size
            n1, n3 = len(t1), len(t3)
            r_rb = 1 - (2 * mw_stat) / (n1 * n3)
        else:
            mw_stat, mw_p, r_rb = np.nan, np.nan, np.nan

        # Kruskal-Wallis across all three terciles
        groups = [
            buurt_means[buurt_means[tercile_col] == t]["mean_distance"].dropna()
            for t in [1, 2, 3]
        ]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        all_test_results.append(
            {
                "variable": var,
                "label": info["label"],
                "mann_whitney_U": mw_stat,
                "mann_whitney_p": mw_p,
                "rank_biserial_r": r_rb,
                "kruskal_wallis_H": kw_stat,
                "kruskal_wallis_p": kw_p,
                "n_tercile_1": len(t1),
                "n_tercile_3": len(t3),
                "mean_dist_tercile_1": t1.mean() if len(t1) > 0 else np.nan,
                "mean_dist_tercile_3": t3.mean() if len(t3) > 0 else np.nan,
            }
        )

        # Plot boxplot only for car ownership (the only significant result)
        if var == "personenautos_per_huishouden":
            _plot_boxplot(coverage_with_class, buurt_features, var, tercile_col, info)

    # Save results
    if all_test_results:
        pd.DataFrame(all_test_results).to_csv(
            ANALYSIS_TABLES_DIR / "splintering_tests.csv", index=False
        )
        logger.info("Saved splintering_tests.csv")

    if all_category_summaries:
        pd.concat(all_category_summaries).to_csv(
            ANALYSIS_TABLES_DIR / "splintering_category_summary.csv", index=False
        )
        logger.info("Saved splintering_category_summary.csv")

    logger.info("Done. Outputs in %s and %s", ANALYSIS_FIGURES_DIR, ANALYSIS_TABLES_DIR)


def _plot_boxplot(coverage_with_class, buurt_features, var, tercile_col, info):
    """Box plot of mean distance by tercile."""
    data = coverage_with_class[coverage_with_class[tercile_col].notna()].copy()
    if data.empty:
        return

    # Compute tercile boundaries for tick labels
    valid_vals = buurt_features[var].dropna()
    t1_max = valid_vals.quantile(1 / 3)
    t2_max = valid_vals.quantile(2 / 3)

    range_strs = [
        f"(\u2264{t1_max:.2f})",
        f"({t1_max:.2f}\u2013{t2_max:.2f})",
        f"(>{t2_max:.2f})",
    ]

    # Aggregate to per-buurt means first (not per-hour)
    buurt_means = (
        data.groupby(["buurt_idx", tercile_col])["mean_distance"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = []
    labels = []
    for t in [1, 2, 3]:
        vals = buurt_means[buurt_means[tercile_col] == t]["mean_distance"].dropna()
        if len(vals) > 0:
            groups.append(vals)
            labels.append(
                f"{info['tercile_labels'][int(t) - 1]}\n{range_strs[int(t) - 1]}"
            )

    if groups:
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
        colors = ["#4CAF50", "#FFC107", "#F44336"]
        for patch, color in zip(bp["boxes"], colors[: len(groups)], strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Mean distance to nearest bike (m)")
    ax.set_title(f"Coverage by {info['label']} tercile")
    plt.tight_layout()

    safe_name = var.replace("percentage_met_herkomstland_buiten_europa", "migration")
    safe_name = safe_name.replace("personenautos_per_huishouden", "cars")
    safe_name = safe_name.replace("gemiddeld_inkomen_per_inwoner", "income")
    safe_name = safe_name.replace("gemiddelde_woz_waarde", "woz")
    plt.savefig(ANALYSIS_FIGURES_DIR / f"splintering_boxplot_{safe_name}.png", dpi=150)
    plt.close()
    logger.info("Saved splintering_boxplot_%s.png", safe_name)


if __name__ == "__main__":
    run()
