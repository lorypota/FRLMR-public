"""Temporal coverage analysis: how bike coverage changes over the day.

Computes hourly coverage metrics, identifies commute patterns, and tests
for significant differences between time periods.

Run:
    uv run preliminary_studies/empirical_analysis/analysis_temporal.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from internal.coverage_utils import (
    PROVIDER,
    aggregate_by_buurt,
    assign_houses_to_buurten,
    compute_coverage_metrics,
    compute_nearest_distances,
    get_docked_bike_positions,
    load_buurten,
    load_houses,
    subsample_hourly_wide,
    wgs84_to_rd,
)
from internal.paths import (
    ANALYSIS_CACHE_DIR,
    ANALYSIS_FIGURES_DIR,
    ANALYSIS_TABLES_DIR,
    DATA_DIR,
    ensure_output_dirs,
)
from internal.processed_data_utils import (
    discover_docked_dates,
    load_docked_day,
    load_station_day,
)
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_day(year, month, day, houses, houses_rd, weights, buurten, buurt_idx):
    """Compute hourly coverage for one day.

    Returns list of dicts with per-hour city-wide and per-buurt metrics,
    or None if data is missing.
    """
    docked = load_docked_day(DATA_DIR, PROVIDER, year, month, day)
    stations = load_station_day(DATA_DIR, PROVIDER, year, month, day)

    if docked is None or stations is None:
        logger.warning(
            "Missing docked or station data for %04d-%02d-%02d, skipping",
            year,
            month,
            day,
        )
        return None

    docked_h = subsample_hourly_wide(docked)

    n_buurten = len(buurten)

    day_results = []
    buurt_results = []

    for ts, docked_row in docked_h.iterrows():
        hour = pd.Timestamp(ts).hour
        bikes = get_docked_bike_positions(docked_row, stations)
        if len(bikes) == 0:
            continue

        bikes_rd = wgs84_to_rd(bikes)
        distances = compute_nearest_distances(houses_rd, bikes_rd)
        metrics = compute_coverage_metrics(distances, weights)

        day_results.append(
            {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "hour": hour,
                "n_bikes": len(bikes),
                "mean_distance": metrics["mean_distance"],
                "median_distance": metrics["median_distance"],
                "pct_within_100m": metrics["pct_within_100m"],
                "pct_within_250m": metrics["pct_within_250m"],
                "pct_within_500m": metrics["pct_within_500m"],
                **{f"band_{k}": v for k, v in metrics["band_proportions"].items()},
            }
        )

        buurt_agg = aggregate_by_buurt(distances, weights, buurt_idx, n_buurten)
        buurt_agg["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        buurt_agg["hour"] = hour
        buurt_results.append(buurt_agg)

    return day_results, buurt_results


def run():
    ensure_output_dirs()

    logger.info("Loading house and buurt data...")
    houses = load_houses()
    houses_rd = wgs84_to_rd(houses[:, :2])
    weights = houses[:, 2]
    buurten = load_buurten()
    buurt_idx = assign_houses_to_buurten(houses_rd, buurten)

    # Discover available dates
    dates = discover_docked_dates(DATA_DIR, PROVIDER)
    if not dates:
        logger.error("No data available in %s", DATA_DIR)
        sys.exit(1)
    logger.info("Found %d dates to process", len(dates))

    all_city = []
    all_buurt = []

    for i, (y, m, d) in enumerate(dates):
        logger.info("Processing %04d-%02d-%02d (%d/%d)...", y, m, d, i + 1, len(dates))
        result = process_day(y, m, d, houses, houses_rd, weights, buurten, buurt_idx)
        if result is None:
            continue
        day_city, day_buurt = result
        all_city.extend(day_city)
        all_buurt.extend(day_buurt)

        # Cache per-day buurt results
        if day_buurt:
            cache_df = pd.concat(day_buurt)
            cache_path = ANALYSIS_CACHE_DIR / f"coverage_{y:04d}{m:02d}{d:02d}.csv"
            cache_df.to_csv(cache_path)

    if not all_city:
        logger.error("No data processed successfully")
        sys.exit(1)

    city_df = pd.DataFrame(all_city)
    city_df["weekday"] = pd.to_datetime(city_df["date"]).dt.dayofweek < 5

    # Save summary table
    summary = (
        city_df.groupby("hour")
        .agg(
            mean_distance=("mean_distance", "mean"),
            std_distance=("mean_distance", "std"),
            mean_pct_500m=("pct_within_500m", "mean"),
            std_pct_500m=("pct_within_500m", "std"),
            mean_n_bikes=("n_bikes", "mean"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(ANALYSIS_TABLES_DIR / "temporal_summary.csv", index=False)
    logger.info("Saved temporal_summary.csv")

    # Weekday vs weekend summary
    wd_summary = (
        city_df.groupby(["hour", "weekday"])
        .agg(
            mean_distance=("mean_distance", "mean"),
            mean_pct_500m=("pct_within_500m", "mean"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    wd_summary.to_csv(ANALYSIS_TABLES_DIR / "temporal_weekday_weekend.csv", index=False)

    # Peak/trough analysis
    best_hour = summary.loc[summary["mean_pct_500m"].idxmax()]
    worst_hour = summary.loc[summary["mean_pct_500m"].idxmin()]

    # Statistical test: morning (7-9) vs afternoon (15-17) coverage
    morning = (
        city_df[city_df["hour"].between(7, 9)].groupby("date")["pct_within_500m"].mean()
    )
    afternoon = (
        city_df[city_df["hour"].between(15, 17)]
        .groupby("date")["pct_within_500m"]
        .mean()
    )
    common_dates = morning.index.intersection(afternoon.index)
    if len(common_dates) >= 3:
        stat, pval = stats.wilcoxon(
            morning.loc[common_dates], afternoon.loc[common_dates]
        )
        test_result = {
            "test": "Wilcoxon",
            "statistic": stat,
            "p_value": pval,
            "n": len(common_dates),
        }
    else:
        test_result = {
            "test": "Wilcoxon",
            "statistic": np.nan,
            "p_value": np.nan,
            "n": len(common_dates),
        }

    peaks_df = pd.DataFrame(
        [
            {
                "metric": "best_coverage_hour",
                "hour": int(best_hour["hour"]),
                "pct_within_500m": best_hour["mean_pct_500m"],
            },
            {
                "metric": "worst_coverage_hour",
                "hour": int(worst_hour["hour"]),
                "pct_within_500m": worst_hour["mean_pct_500m"],
            },
            {
                "metric": "morning_vs_afternoon",
                "hour": np.nan,
                "pct_within_500m": np.nan,
                **test_result,
            },
        ]
    )
    peaks_df.to_csv(ANALYSIS_TABLES_DIR / "temporal_peaks.csv", index=False)
    logger.info("Saved temporal_peaks.csv")

    # --- Plots ---
    _plot_hourly_profile(summary)
    _plot_weekday_weekend(wd_summary)
    _plot_distance_bands(city_df)

    logger.info("Done. Outputs in %s and %s", ANALYSIS_FIGURES_DIR, ANALYSIS_TABLES_DIR)


def _plot_hourly_profile(summary):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    hours = summary["hour"]
    ax1.plot(
        hours,
        summary["mean_distance"],
        "o-",
        color="tab:red",
        label="Mean distance (m)",
    )
    ax1.fill_between(
        hours,
        summary["mean_distance"] - summary["std_distance"],
        summary["mean_distance"] + summary["std_distance"],
        alpha=0.2,
        color="tab:red",
    )
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean distance to nearest bike (m)", color="tab:red")
    ax1.set_xticks(range(0, 24))

    ax2 = ax1.twinx()
    ax2.plot(
        hours, summary["mean_pct_500m"], "s-", color="tab:blue", label="% within 500m"
    )
    ax2.fill_between(
        hours,
        summary["mean_pct_500m"] - summary["std_pct_500m"],
        summary["mean_pct_500m"] + summary["std_pct_500m"],
        alpha=0.2,
        color="tab:blue",
    )
    ax2.set_ylabel("% addresses within 500m", color="tab:blue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("Docked-bike coverage over 24 hours")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURES_DIR / "temporal_hourly_profile.png", dpi=150)
    plt.close()
    logger.info("Saved temporal_hourly_profile.png")


def _plot_weekday_weekend(wd_summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for is_wd, label, color in [
        (True, "Weekday", "tab:blue"),
        (False, "Weekend", "tab:orange"),
    ]:
        sub = wd_summary[wd_summary["weekday"] == is_wd]
        if sub.empty:
            continue
        ax1.plot(sub["hour"], sub["mean_distance"], "o-", color=color, label=label)
        ax2.plot(sub["hour"], sub["mean_pct_500m"], "o-", color=color, label=label)

    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean distance to nearest bike (m)")
    ax1.set_title("Mean distance")
    ax1.set_xticks(range(0, 24))
    ax1.legend()

    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("% addresses within 500m")
    ax2.set_title("Coverage rate")
    ax2.set_xticks(range(0, 24))
    ax2.legend()

    plt.suptitle("Weekday vs Weekend docked-bike coverage")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURES_DIR / "temporal_weekday_weekend.png", dpi=150)
    plt.close()
    logger.info("Saved temporal_weekday_weekend.png")


def _plot_distance_bands(city_df):
    band_cols = [c for c in city_df.columns if c.startswith("band_")]
    if not band_cols:
        return

    hourly = city_df.groupby("hour")[band_cols].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        hourly.index,
        *[hourly[c] for c in band_cols],
        labels=[c.replace("band_", "") for c in band_cols],
        alpha=0.8,
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of addresses")
    ax.set_title("Distance band distribution over 24 hours")
    ax.set_xticks(range(0, 24))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURES_DIR / "temporal_distance_bands.png", dpi=150)
    plt.close()
    logger.info("Saved temporal_distance_bands.png")


if __name__ == "__main__":
    run()
