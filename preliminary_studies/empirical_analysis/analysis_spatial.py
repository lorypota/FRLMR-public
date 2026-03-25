"""Spatial inequality analysis: per-buurt coverage correlated with demographics.

Computes Gini/Theil inequality metrics and Spearman correlations between
coverage and demographic variables (income, WOZ, migration background,
car ownership).

Run:
    uv run preliminary_studies/empirical_analysis/analysis_spatial.py
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

# Demographic variables from buurten GeoJSON (available for most buurten)
DEMO_VARS_BUURTEN = {
    "percentage_met_herkomstland_buiten_europa": "% non-European background",
    "personenautos_per_huishouden": "Cars per household",
    "bevolkingsdichtheid_inwoners_per_km2": "Population density (per km2)",
    "percentage_koopwoningen": "% owner-occupied",
}

# Income variables from CBS StatLine
DEMO_VARS_INCOME = {
    "gemiddeld_inkomen_per_inwoner": "Mean income per resident (x1000 EUR)",
    "gemiddelde_woz_waarde": "Mean WOZ property value (x1000 EUR)",
    "pct_huishoudens_laag_inkomen": "% low-income households",
}


def gini_coefficient(values, weights=None):
    """Compute the Gini coefficient of a distribution."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[mask], weights[mask]

    if len(values) < 2:
        return np.nan

    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]

    cum_w = np.cumsum(weights)
    total_w = cum_w[-1]
    cum_vals = np.cumsum(values * weights)
    total_val = cum_vals[-1]

    if total_val == 0:
        return 0.0

    # Area under Lorenz curve
    lorenz = cum_vals / total_val
    pop_frac = cum_w / total_w
    area_under = np.trapezoid(lorenz, pop_frac)
    return 1 - 2 * area_under


def theil_index(values, weights=None):
    """Compute the Theil T index of a distribution."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & (values > 0) & np.isfinite(weights)
    values, weights = values[mask], weights[mask]

    if len(values) < 2:
        return np.nan

    mean_val = np.average(values, weights=weights)
    if mean_val == 0:
        return 0.0

    ratios = values / mean_val
    return float(np.average(ratios * np.log(ratios), weights=weights))


def run():
    ensure_output_dirs()

    logger.info("Loading buurt data...")
    buurten = load_buurten()

    # Load cached coverage data
    cache_files = sorted(ANALYSIS_CACHE_DIR.glob("coverage_*.csv"))
    if not cache_files:
        logger.error(
            "No cached coverage data found in %s. Run analysis_temporal.py first.",
            ANALYSIS_CACHE_DIR,
        )
        sys.exit(1)

    logger.info("Loading %d cached coverage files...", len(cache_files))
    all_buurt_coverage = []
    for f in cache_files:
        df = pd.read_csv(f)
        all_buurt_coverage.append(df)

    coverage_df = pd.concat(all_buurt_coverage)

    # Average coverage across all timestamps per buurt
    buurt_coverage = (
        coverage_df.groupby("buurt_idx")
        .agg(
            mean_distance=("mean_distance", "mean"),
            std_distance=("mean_distance", "std"),
            mean_pct_100m=("pct_within_100m", "mean"),
            mean_pct_250m=("pct_within_250m", "mean"),
            mean_pct_500m=("pct_within_500m", "mean"),
            total_addresses=("total_addresses", "first"),
            n_observations=("mean_distance", "count"),
        )
        .reset_index()
    )

    # Merge with buurt demographics
    buurt_summary = buurt_coverage.copy()
    for col in DEMO_VARS_BUURTEN:
        if col in buurten.columns:
            buurt_summary[col] = buurten.loc[buurt_summary["buurt_idx"], col].values

    # Merge CBS income data
    income_df = fetch_income_data(cache_dir=GEODATA_DIR)
    if income_df is not None:
        buurt_codes = buurten.loc[buurt_summary["buurt_idx"], "buurtcode"].values
        buurt_summary["buurtcode"] = buurt_codes
        buurt_summary = buurt_summary.merge(income_df, on="buurtcode", how="left")
    else:
        logger.warning("No income data available, skipping income correlations")

    # Add buurt names
    buurt_summary["buurtnaam"] = buurten.loc[
        buurt_summary["buurt_idx"], "buurtnaam"
    ].values

    # Save full summary table
    buurt_summary.to_csv(ANALYSIS_TABLES_DIR / "spatial_buurt_summary.csv", index=False)
    logger.info("Saved spatial_buurt_summary.csv (%d buurten)", len(buurt_summary))

    # Inequality metrics
    valid = buurt_summary["mean_distance"].notna() & (
        buurt_summary["total_addresses"] > 0
    )
    gini = gini_coefficient(
        buurt_summary.loc[valid, "mean_distance"].values,
        buurt_summary.loc[valid, "total_addresses"].values,
    )
    theil = theil_index(
        buurt_summary.loc[valid, "mean_distance"].values,
        buurt_summary.loc[valid, "total_addresses"].values,
    )
    inequality_df = pd.DataFrame(
        [
            {"metric": "gini_mean_distance", "value": gini, "n_buurten": valid.sum()},
            {"metric": "theil_mean_distance", "value": theil, "n_buurten": valid.sum()},
        ]
    )
    inequality_df.to_csv(ANALYSIS_TABLES_DIR / "spatial_inequality.csv", index=False)
    logger.info("Gini: %.3f, Theil: %.3f", gini, theil)

    # Correlations
    all_demo_vars = {**DEMO_VARS_BUURTEN, **DEMO_VARS_INCOME}
    corr_results = []
    for var, label in all_demo_vars.items():
        if var not in buurt_summary.columns:
            continue
        mask = buurt_summary["mean_distance"].notna() & buurt_summary[var].notna()
        n = mask.sum()
        if n < 5:
            continue
        rho, pval = stats.spearmanr(
            buurt_summary.loc[mask, "mean_distance"],
            buurt_summary.loc[mask, var],
        )
        corr_results.append(
            {
                "variable": var,
                "label": label,
                "spearman_rho": rho,
                "p_value": pval,
                "n": n,
            }
        )
        logger.info("  %s: rho=%.3f, p=%.4f (n=%d)", label, rho, pval, n)

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(ANALYSIS_TABLES_DIR / "spatial_correlations.csv", index=False)
    logger.info("Saved spatial_correlations.csv")

    # --- Plots ---
    _plot_scatter(buurt_summary, corr_df, all_demo_vars)
    _plot_choropleth(buurt_summary, buurten)

    logger.info("Done. Outputs in %s and %s", ANALYSIS_FIGURES_DIR, ANALYSIS_TABLES_DIR)


def _plot_scatter(buurt_summary, corr_df, all_demo_vars):
    """Create scatter plots for each demographic variable."""
    plot_vars = [
        ("percentage_met_herkomstland_buiten_europa", "spatial_scatter_migration.png"),
        ("personenautos_per_huishouden", "spatial_scatter_cars.png"),
        ("gemiddeld_inkomen_per_inwoner", "spatial_scatter_income.png"),
        ("gemiddelde_woz_waarde", "spatial_scatter_woz.png"),
        ("pct_huishoudens_laag_inkomen", "spatial_scatter_low_income.png"),
    ]

    for var, filename in plot_vars:
        if var not in buurt_summary.columns:
            continue
        mask = buurt_summary["mean_distance"].notna() & buurt_summary[var].notna()
        if mask.sum() < 5:
            continue

        x = buurt_summary.loc[mask, var]
        y = buurt_summary.loc[mask, "mean_distance"]
        sizes = buurt_summary.loc[mask, "total_addresses"] / 50

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, s=sizes, alpha=0.6, edgecolors="k", linewidths=0.3)

        # Add correlation annotation
        corr_row = corr_df[corr_df["variable"] == var]
        if not corr_row.empty:
            rho = corr_row.iloc[0]["spearman_rho"]
            pval = corr_row.iloc[0]["p_value"]
            ax.annotate(
                f"Spearman rho = {rho:.3f}\np = {pval:.4f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        label = all_demo_vars.get(var, var)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean distance to nearest bike (m)")
        ax.set_title(f"Coverage vs {label}")
        plt.tight_layout()
        plt.savefig(ANALYSIS_FIGURES_DIR / filename, dpi=150)
        plt.close()
        logger.info("Saved %s", filename)


def _plot_choropleth(buurt_summary, buurten):
    """Plot mean coverage as a choropleth map."""
    gdf = buurten.copy()
    gdf = gdf.merge(
        buurt_summary[["buurt_idx", "mean_distance", "mean_pct_500m"]],
        left_index=True,
        right_on="buurt_idx",
        how="left",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    gdf.plot(
        column="mean_distance",
        ax=ax1,
        legend=True,
        cmap="RdYlGn_r",
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "Mean distance (m)"},
    )
    ax1.set_title("Mean distance to nearest bike")
    ax1.set_axis_off()

    gdf.plot(
        column="mean_pct_500m",
        ax=ax2,
        legend=True,
        cmap="RdYlGn",
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "% within 500m"},
    )
    ax2.set_title("% addresses within 500m of a bike")
    ax2.set_axis_off()

    plt.suptitle("Bike coverage by neighborhood (buurt)")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURES_DIR / "spatial_choropleth.png", dpi=150)
    plt.close()
    logger.info("Saved spatial_choropleth.png")


if __name__ == "__main__":
    run()
