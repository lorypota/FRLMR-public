"""Legacy statistical coverage analysis for docked-bike access in Den Haag.

Archived broad analysis pipeline kept for reference. The main entry point in this
folder is now `statistical_analysis.py`.

Runs one or more analysis steps:
- temporal: hourly city-wide coverage summaries and cache generation
- spatial: buurt-level inequality, demographic correlations, and grouped comparisons

Run:
    uv run preliminary_studies/empirical_analysis/legacy/analysis_statistics_legacy.py
    uv run preliminary_studies/empirical_analysis/legacy/analysis_statistics_legacy.py --step temporal
    uv run preliminary_studies/empirical_analysis/legacy/analysis_statistics_legacy.py --step temporal --start-date 2026-01-01 --end-date 2026-12-31
"""

import argparse
import concurrent.futures
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from internal.cbs_income import fetch_income_data
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
    DATA_DIR,
    GEODATA_DIR,
    analysis_run_paths,
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

DEMO_VARS_BUURTEN = {
    "percentage_met_herkomstland_buiten_europa": "% non-European background",
    "personenautos_per_huishouden": "Cars per household",
    "bevolkingsdichtheid_inwoners_per_km2": "Population density (per km2)",
    "percentage_koopwoningen": "% owner-occupied",
}

DEMO_VARS_INCOME = {
    "gemiddeld_inkomen_per_inwoner": "Mean income per resident (x1000 EUR)",
    "gemiddelde_woz_waarde": "Mean WOZ property value (x1000 EUR)",
    "pct_huishoudens_laag_inkomen": "% low-income households",
}

CLASSIFICATION_VARS = {
    "percentage_met_herkomstland_buiten_europa": {
        "label": "Non-European background",
        "source": "buurten",
        "tercile_labels": ["Low migration", "Medium migration", "High migration"],
    },
    "personenautos_per_huishouden": {
        "label": "Car ownership",
        "source": "buurten",
        "tercile_labels": [
            "Low car ownership",
            "Medium car ownership",
            "High car ownership",
        ],
    },
    "gemiddeld_inkomen_per_inwoner": {
        "label": "Income",
        "source": "income",
        "tercile_labels": ["Low income", "Medium income", "High income"],
    },
    "gemiddelde_woz_waarde": {
        "label": "Property value (WOZ)",
        "source": "income",
        "tercile_labels": ["Low WOZ", "Medium WOZ", "High WOZ"],
    },
}


def _load_cached_coverage(run_paths) -> pd.DataFrame:
    cache_files = sorted(run_paths["coverage_dir"].glob("coverage_*.csv"))
    if not cache_files:
        logger.error(
            "No cached coverage data found in %s. Run the temporal step first.",
            run_paths["coverage_dir"],
        )
        sys.exit(1)
    logger.info("Loading %d cached coverage files...", len(cache_files))
    return pd.concat([pd.read_csv(path) for path in cache_files], ignore_index=True)


def _load_processed_dates(run_paths) -> list[str] | None:
    if not run_paths["processed_dates_path"].exists():
        return None
    with open(run_paths["processed_dates_path"]) as f:
        return json.load(f)


def _clear_cache(run_paths) -> None:
    for path in run_paths["coverage_dir"].glob("coverage_*.csv"):
        path.unlink()
    if run_paths["processed_dates_path"].exists():
        run_paths["processed_dates_path"].unlink()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _filter_dates(
    dates: list[tuple[int, int, int]],
    start_date: date | None,
    end_date: date | None,
) -> list[tuple[int, int, int]]:
    filtered = []
    for year, month, day in dates:
        current = date(year, month, day)
        if start_date is not None and current < start_date:
            continue
        if end_date is not None and current > end_date:
            continue
        filtered.append((year, month, day))
    return filtered


def gini_coefficient(values, weights=None):
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
    cum_vals = np.cumsum(values * weights)
    total_w = cum_w[-1]
    total_val = cum_vals[-1]
    if total_val == 0:
        return 0.0

    lorenz = cum_vals / total_val
    pop_frac = cum_w / total_w
    area_under = np.trapezoid(lorenz, pop_frac)
    return 1 - 2 * area_under


def theil_index(values, weights=None):
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


def classify_terciles(values):
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


def _process_temporal_day(
    year,
    month,
    day,
    houses_rd,
    weights,
    n_buurten,
    buurt_idx,
    total_addresses,
):
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

    day_results = []
    buurt_results = []
    for ts, docked_row in subsample_hourly_wide(docked).iterrows():
        bikes = get_docked_bike_positions(docked_row, stations)
        if len(bikes) == 0:
            continue

        hour = pd.Timestamp(ts).hour
        distances = compute_nearest_distances(houses_rd, wgs84_to_rd(bikes))
        metrics = compute_coverage_metrics(distances, weights)
        day_results.append(
            {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "hour": hour,
                "year": year,
                "month": month,
                "n_bikes": len(bikes),
                "mean_distance": metrics["mean_distance"],
                "median_distance": metrics["median_distance"],
                "pct_within_100m": metrics["pct_within_100m"],
                "pct_within_250m": metrics["pct_within_250m"],
                "pct_within_500m": metrics["pct_within_500m"],
                "covered_addresses_500m": metrics["pct_within_500m"]
                / 100
                * total_addresses,
                "covered_addresses_per_bike_500m": (
                    metrics["pct_within_500m"] / 100 * total_addresses / len(bikes)
                ),
                **{f"band_{k}": v for k, v in metrics["band_proportions"].items()},
            }
        )
        buurt_agg = aggregate_by_buurt(distances, weights, buurt_idx, n_buurten)
        buurt_agg["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        buurt_agg["hour"] = hour
        buurt_results.append(buurt_agg)

    return day_results, buurt_results


def _process_temporal_day_task(task):
    return _process_temporal_day(*task)


def run_temporal(start_date: date | None, end_date: date | None, max_workers: int):
    ensure_output_dirs()
    run_paths = analysis_run_paths(start_date, end_date)
    logger.info("Loading house and buurt data...")
    houses = load_houses()
    houses_rd = wgs84_to_rd(houses[:, :2])
    weights = houses[:, 2]
    total_addresses = float(weights.sum())
    buurten = load_buurten()
    buurt_idx = assign_houses_to_buurten(houses_rd, buurten)

    dates = discover_docked_dates(DATA_DIR, PROVIDER)
    dates = _filter_dates(dates, start_date, end_date)
    if not dates:
        logger.error("No data available in %s", DATA_DIR)
        sys.exit(1)

    _clear_cache(run_paths)
    all_city = []
    tasks = [
        (
            year,
            month,
            day,
            houses_rd,
            weights,
            len(buurten),
            buurt_idx,
            total_addresses,
        )
        for year, month, day in dates
    ]
    logger.info(
        "Processing %d dates from %s to %s with %d workers...",
        len(dates),
        date(*dates[0]),
        date(*dates[-1]),
        max_workers,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {
            executor.submit(_process_temporal_day_task, task): task[:3]
            for task in tasks
        }
        for i, future in enumerate(
            concurrent.futures.as_completed(future_to_date),
            start=1,
        ):
            year, month, day = future_to_date[future]
            logger.info(
                "Finished %04d-%02d-%02d (%d/%d)",
                year,
                month,
                day,
                i,
                len(tasks),
            )
            result = future.result()
            if result is None:
                continue
            day_city, day_buurt = result
            all_city.extend(day_city)
            if day_buurt:
                pd.concat(day_buurt).to_csv(
                    run_paths["coverage_dir"]
                    / f"coverage_{year:04d}{month:02d}{day:02d}.csv"
                )

    if not all_city:
        logger.error("No data processed successfully")
        sys.exit(1)

    city_df = pd.DataFrame(all_city)
    city_df["weekday"] = pd.to_datetime(city_df["date"]).dt.dayofweek < 5
    city_df.to_csv(run_paths["tables_dir"] / "temporal_daily_hour.csv", index=False)
    processed_dates = sorted(city_df["date"].drop_duplicates().tolist())
    with open(run_paths["processed_dates_path"], "w") as f:
        json.dump(processed_dates, f, indent=2)

    summary = (
        city_df.groupby("hour")
        .agg(
            mean_distance=("mean_distance", "mean"),
            std_distance=("mean_distance", "std"),
            mean_pct_500m=("pct_within_500m", "mean"),
            std_pct_500m=("pct_within_500m", "std"),
            mean_n_bikes=("n_bikes", "mean"),
            mean_covered_addresses_per_bike_500m=(
                "covered_addresses_per_bike_500m",
                "mean",
            ),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(run_paths["tables_dir"] / "temporal_summary.csv", index=False)

    yearly_summary = (
        city_df.groupby(["year", "month", "hour"])
        .agg(
            mean_distance=("mean_distance", "mean"),
            std_distance=("mean_distance", "std"),
            mean_pct_500m=("pct_within_500m", "mean"),
            std_pct_500m=("pct_within_500m", "std"),
            mean_n_bikes=("n_bikes", "mean"),
            mean_covered_addresses_per_bike_500m=(
                "covered_addresses_per_bike_500m",
                "mean",
            ),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    yearly_summary.to_csv(
        run_paths["tables_dir"] / "temporal_year_month_hour.csv",
        index=False,
    )

    wd_summary = (
        city_df.groupby(["year", "month", "hour", "weekday"])
        .agg(
            mean_distance=("mean_distance", "mean"),
            mean_pct_500m=("pct_within_500m", "mean"),
            mean_n_bikes=("n_bikes", "mean"),
            mean_covered_addresses_per_bike_500m=(
                "covered_addresses_per_bike_500m",
                "mean",
            ),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    wd_summary.to_csv(
        run_paths["tables_dir"] / "temporal_weekday_weekend.csv",
        index=False,
    )


def run_spatial(start_date: date | None, end_date: date | None):
    ensure_output_dirs()
    run_paths = analysis_run_paths(start_date, end_date)
    logger.info("Loading buurt data...")
    buurten = load_buurten()
    coverage_df = _load_cached_coverage(run_paths)
    processed_dates = _load_processed_dates(run_paths)
    if processed_dates is not None and processed_dates:
        logger.info(
            "Using temporal cache for %s to %s (%d dates)",
            processed_dates[0],
            processed_dates[-1],
            len(processed_dates),
        )

    buurt_summary = (
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

    for col in DEMO_VARS_BUURTEN:
        if col in buurten.columns:
            buurt_summary[col] = buurten.loc[buurt_summary["buurt_idx"], col].values

    income_df = fetch_income_data(cache_dir=GEODATA_DIR)
    if income_df is not None:
        buurt_summary["buurtcode"] = buurten.loc[
            buurt_summary["buurt_idx"], "buurtcode"
        ].values
        buurt_summary = buurt_summary.merge(income_df, on="buurtcode", how="left")

    buurt_summary["buurtnaam"] = buurten.loc[
        buurt_summary["buurt_idx"], "buurtnaam"
    ].values
    buurt_summary.to_csv(
        run_paths["tables_dir"] / "spatial_buurt_summary.csv",
        index=False,
    )

    valid = buurt_summary["mean_distance"].notna() & (
        buurt_summary["total_addresses"] > 0
    )
    pd.DataFrame(
        [
            {
                "metric": "gini_mean_distance",
                "value": gini_coefficient(
                    buurt_summary.loc[valid, "mean_distance"].values,
                    buurt_summary.loc[valid, "total_addresses"].values,
                ),
                "n_buurten": valid.sum(),
            },
            {
                "metric": "theil_mean_distance",
                "value": theil_index(
                    buurt_summary.loc[valid, "mean_distance"].values,
                    buurt_summary.loc[valid, "total_addresses"].values,
                ),
                "n_buurten": valid.sum(),
            },
        ]
    ).to_csv(run_paths["tables_dir"] / "spatial_inequality.csv", index=False)

    corr_results = []
    for var, label in {**DEMO_VARS_BUURTEN, **DEMO_VARS_INCOME}.items():
        if var not in buurt_summary.columns:
            continue
        mask = buurt_summary["mean_distance"].notna() & buurt_summary[var].notna()
        if mask.sum() < 5:
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
                "n": mask.sum(),
            }
        )

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(run_paths["tables_dir"] / "spatial_correlations.csv", index=False)
    _plot_scatter_cars(buurt_summary, corr_df, run_paths)
    _plot_choropleth(buurt_summary, buurten, run_paths)
    _run_spatial_grouped_comparisons(buurten, coverage_df, income_df, run_paths)


def _run_spatial_grouped_comparisons(buurten, coverage_df, income_df, run_paths):
    logger.info("Running grouped spatial comparisons...")

    buurt_features = pd.DataFrame({"buurt_idx": range(len(buurten))})
    buurt_features["buurtcode"] = buurten["buurtcode"].values

    for var, info in CLASSIFICATION_VARS.items():
        if info["source"] == "buurten" and var in buurten.columns:
            buurt_features[var] = buurten[var].values
        elif (
            info["source"] == "income"
            and income_df is not None
            and var in income_df.columns
        ):
            merged = buurt_features[["buurt_idx", "buurtcode"]].merge(
                income_df[["buurtcode", var]],
                on="buurtcode",
                how="left",
            )
            buurt_features[var] = merged[var].values

    for var in CLASSIFICATION_VARS:
        if var in buurt_features.columns:
            buurt_features[f"{var}_tercile"] = classify_terciles(buurt_features[var])

    coverage_with_class = coverage_df.merge(buurt_features, on="buurt_idx", how="left")
    all_test_results = []
    all_category_summaries = []

    for var, info in CLASSIFICATION_VARS.items():
        tercile_col = f"{var}_tercile"
        if tercile_col not in coverage_with_class.columns:
            continue
        if coverage_with_class[tercile_col].notna().sum() == 0:
            continue

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

        t1 = buurt_means[buurt_means[tercile_col] == 1]["mean_distance"].dropna()
        t3 = buurt_means[buurt_means[tercile_col] == 3]["mean_distance"].dropna()
        if len(t1) >= 5 and len(t3) >= 5:
            mw_stat, mw_p = stats.mannwhitneyu(t1, t3, alternative="two-sided")
            r_rb = 1 - (2 * mw_stat) / (len(t1) * len(t3))
        else:
            mw_stat, mw_p, r_rb = np.nan, np.nan, np.nan

        groups = [
            buurt_means[buurt_means[tercile_col] == tercile]["mean_distance"].dropna()
            for tercile in [1, 2, 3]
        ]
        groups = [group for group in groups if len(group) >= 5]
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

        if var == "personenautos_per_huishouden":
            _plot_boxplot(
                coverage_with_class,
                buurt_features,
                var,
                tercile_col,
                info,
                run_paths,
            )

    if all_test_results:
        pd.DataFrame(all_test_results).to_csv(
            run_paths["tables_dir"] / "spatial_grouped_tests.csv",
            index=False,
        )
    if all_category_summaries:
        pd.concat(all_category_summaries).to_csv(
            run_paths["tables_dir"] / "spatial_grouped_summary.csv",
            index=False,
        )


def _plot_scatter_cars(buurt_summary, corr_df, run_paths):
    var = "personenautos_per_huishouden"
    if var not in buurt_summary.columns:
        return
    mask = buurt_summary["mean_distance"].notna() & buurt_summary[var].notna()
    if mask.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        buurt_summary.loc[mask, var],
        buurt_summary.loc[mask, "mean_distance"],
        s=buurt_summary.loc[mask, "total_addresses"] / 50,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
    )
    corr_row = corr_df[corr_df["variable"] == var]
    if not corr_row.empty:
        row = corr_row.iloc[0]
        ax.annotate(
            f"Spearman rho = {row['spearman_rho']:.3f}\np = {row['p_value']:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.set_xlabel("Cars per household")
    ax.set_ylabel("Mean distance to nearest bike (m)")
    ax.set_title("Coverage vs Cars per household")
    plt.tight_layout()
    plt.savefig(run_paths["figures_dir"] / "spatial_scatter_cars.png", dpi=150)
    plt.close()


def _plot_choropleth(buurt_summary, buurten, run_paths):
    gdf = buurten.copy().merge(
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
    plt.savefig(run_paths["figures_dir"] / "spatial_choropleth.png", dpi=150)
    plt.close()


def _plot_boxplot(
    coverage_with_class,
    buurt_features,
    var,
    tercile_col,
    info,
    run_paths,
):
    data = coverage_with_class[coverage_with_class[tercile_col].notna()].copy()
    if data.empty:
        return

    valid_vals = buurt_features[var].dropna()
    t1_max = valid_vals.quantile(1 / 3)
    t2_max = valid_vals.quantile(2 / 3)
    range_strs = [
        f"(<={t1_max:.2f})",
        f"({t1_max:.2f} to {t2_max:.2f})",
        f"(>{t2_max:.2f})",
    ]
    buurt_means = (
        data.groupby(["buurt_idx", tercile_col])["mean_distance"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = []
    labels = []
    for tercile in [1, 2, 3]:
        values = buurt_means[buurt_means[tercile_col] == tercile][
            "mean_distance"
        ].dropna()
        if len(values) > 0:
            groups.append(values)
            labels.append(
                f"{info['tercile_labels'][int(tercile) - 1]}\n{range_strs[int(tercile) - 1]}"
            )
    if groups:
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
        for patch, color in zip(
            bp["boxes"],
            ["#4CAF50", "#FFC107", "#F44336"][: len(groups)],
            strict=True,
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Mean distance to nearest bike (m)")
    ax.set_title(f"Coverage by {info['label']} tercile")
    plt.tight_layout()
    safe_name = (
        var.replace("percentage_met_herkomstland_buiten_europa", "migration")
        .replace("personenautos_per_huishouden", "cars")
        .replace("gemiddeld_inkomen_per_inwoner", "income")
        .replace("gemiddelde_woz_waarde", "woz")
    )
    plt.savefig(
        run_paths["figures_dir"] / f"spatial_grouped_boxplot_{safe_name}.png",
        dpi=150,
    )
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["temporal", "spatial", "all"],
        default="all",
        help="Which analysis step to run.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Number of worker processes used by the temporal step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.step in {"temporal", "all"}:
        run_temporal(args.start_date, args.end_date, args.max_workers)
    if args.step in {"spatial", "all"}:
        run_spatial(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
