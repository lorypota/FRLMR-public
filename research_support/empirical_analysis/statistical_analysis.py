"""Prepare statistical-analysis tables and thesis figures for key empirical findings.

Usage:
    uv run research_support/empirical_analysis/statistical_analysis.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from internal.coverage_utils import load_buurten, load_houses, wgs84_to_rd
from internal.paths import DATA_DIR, GEODATA_DIR, OUTPUT_DIR, ensure_output_dirs
from internal.processed_data_utils import (
    discover_docked_dates,
    discover_station_dates,
    latest_date,
    load_docked_day,
    load_station_day,
)
from matplotlib import ticker
from scipy.spatial import cKDTree

STAT_ANALYSIS_DIR = OUTPUT_DIR / "statistical_analysis"
STAT_DATA_DIR = STAT_ANALYSIS_DIR / "summary_data"
STAT_FIGURES_DIR = STAT_ANALYSIS_DIR / "figures"
COVERAGE_RUNS_DIR = STAT_ANALYSIS_DIR / "coverage_runs"
PROVIDERS = ["donkey_denHaag", "ns_ov_fiets"]
DEN_HAAG_BBOX_LAT = (52.00, 52.13)
DEN_HAAG_BBOX_LON = (4.20, 4.42)
RECENT_WINDOW_DAYS = 15
COVERAGE_RADIUS_M = 500

FULL_YEAR_RUNS = {
    2022: "20220101_20221231",
    2023: "20230101_20231231",
    2024: "20240101_20241231",
    2025: "20250101_20251231",
}

PARTIAL_START_RUNS = {
    2021: "20211001_20211231",
}

RECENT_WINDOW_RUNS = {
    2025: "20250101_20250320",
    2026: "20260101_20260320",
}

DENSITY_TIME_RUNS = {
    2021: "20211001_20211231",
    2022: "20220101_20221231",
    2023: "20230101_20231231",
    2024: "20240101_20241231",
    2025: "20250101_20251231",
    2026: "20260101_20260320",
}

BACKGROUND = "#f6f0e8"
PANEL_BACKGROUND = "#fbf8f2"
TEXT = "#23313b"
SUBTLE_TEXT = "#66727c"
GRID = "#d9d0c3"
BLUE = "#2f6690"
ORANGE = "#d67a5c"
TEAL = "#2a9d8f"
GOLD = "#d8a34f"
RUST = "#b75d3a"
SLATE = "#8f98a3"
SAGE = "#7aa37a"

DENSITY_COLORS = {
    "T1 lowest density": RUST,
    "T2 middle density": GOLD,
    "T3 highest density": TEAL,
}


def _analysis_tables_dir(run_tag: str) -> Path:
    return COVERAGE_RUNS_DIR / run_tag / "tables"


def _coverage_dir(run_tag: str) -> Path:
    return COVERAGE_RUNS_DIR / run_tag / "buurt_hour_coverage"


def _load_temporal_daily(run_tag: str) -> pd.DataFrame:
    path = _analysis_tables_dir(run_tag) / "temporal_daily_hour.csv"
    return pd.read_csv(
        path,
        usecols=[
            "date",
            "hour",
            "n_bikes",
            "mean_distance",
            "pct_within_500m",
            "covered_addresses_per_bike_500m",
        ],
        parse_dates=["date"],
    )


def _load_spatial_inequality(run_tag: str) -> dict[str, float]:
    path = _analysis_tables_dir(run_tag) / "spatial_inequality.csv"
    df = pd.read_csv(path, usecols=["metric", "value"])
    return dict(zip(df["metric"], df["value"].astype(float), strict=True))


def _gini_coefficient(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    values = np.asarray(values[mask], dtype=float)
    weights = np.asarray(weights[mask], dtype=float)
    if len(values) < 2:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_w = np.cumsum(weights)
    cum_vals = np.cumsum(values * weights)
    total_w = cum_w[-1]
    total_val = cum_vals[-1]
    if total_val == 0:
        return 0.0

    lorenz = cum_vals / total_val
    pop_frac = cum_w / total_w
    area_under = np.trapezoid(lorenz, pop_frac)
    return float(1 - 2 * area_under)


def _load_quarterly_gini(run_tag: str) -> pd.DataFrame:
    frames = []
    for path in sorted(_coverage_dir(run_tag).glob("coverage_*.csv")):
        frames.append(
            pd.read_csv(
                path,
                usecols=["buurt_idx", "total_addresses", "mean_distance", "date"],
                parse_dates=["date"],
            )
        )

    coverage = pd.concat(frames, ignore_index=True)
    coverage["quarter"] = coverage["date"].dt.quarter
    buurt_quarter = coverage.groupby(["quarter", "buurt_idx"], as_index=False).agg(
        mean_distance=("mean_distance", "mean"),
        total_addresses=("total_addresses", "first"),
    )

    rows = []
    for quarter, group in buurt_quarter.groupby("quarter", sort=True):
        valid = group["mean_distance"].notna() & (group["total_addresses"] > 0)
        rows.append(
            {
                "quarter": int(quarter),
                "gini_mean_distance": _gini_coefficient(
                    group.loc[valid, "mean_distance"].to_numpy(),
                    group.loc[valid, "total_addresses"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_coverage_snapshots(run_tag: str) -> pd.DataFrame:
    frames = []
    for path in sorted(_coverage_dir(run_tag).glob("coverage_*.csv")):
        frames.append(
            pd.read_csv(
                path,
                usecols=[
                    "buurt_idx",
                    "total_addresses",
                    "mean_distance",
                    "pct_within_500m",
                    "date",
                ],
                parse_dates=["date"],
            )
        )
    return pd.concat(frames, ignore_index=True)


def _filter_to_den_haag(stations: pd.DataFrame) -> pd.DataFrame:
    return stations[
        stations["lat"].between(*DEN_HAAG_BBOX_LAT)
        & stations["lon"].between(*DEN_HAAG_BBOX_LON)
    ].copy()


def _den_haag_station_ids_for_provider(provider: str) -> set[str] | None:
    latest = latest_date(discover_station_dates(DATA_DIR, provider))
    if latest is None:
        return None
    stations = load_station_day(DATA_DIR, provider, *latest)
    if stations is None:
        return None
    stations = _filter_to_den_haag(stations)
    return set(stations["station_id"].astype(str))


def _load_latest_station_snapshot() -> tuple[tuple[int, int, int], pd.DataFrame]:
    primary_provider = PROVIDERS[0]
    latest_station = latest_date(discover_station_dates(DATA_DIR, primary_provider))
    if latest_station is None:
        raise FileNotFoundError(f"No station snapshots found for {primary_provider}")

    frames = []
    for provider in PROVIDERS:
        provider_latest = (
            latest_station
            if provider == primary_provider
            else latest_date(discover_station_dates(DATA_DIR, provider))
        )
        if provider_latest is None:
            continue
        stations = load_station_day(DATA_DIR, provider, *provider_latest)
        if stations is None:
            if provider == primary_provider:
                raise FileNotFoundError(
                    f"Failed to load station snapshot for {latest_station}"
                )
            continue
        if provider != primary_provider:
            stations = _filter_to_den_haag(stations)
        if stations.empty:
            continue
        stations = stations.copy()
        stations["station_id_str"] = stations["station_id"].astype(str)
        stations["provider"] = provider
        frames.append(stations)

    combined = pd.concat(frames, ignore_index=True)
    return latest_station, combined


def _load_recent_common_docked_station_ids(
    window_days: int = RECENT_WINDOW_DAYS,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    primary_provider = PROVIDERS[0]
    primary_dates: list[tuple[int, int, int]] = []
    all_ids: set[str] = set()

    for provider in PROVIDERS:
        provider_dates = discover_docked_dates(DATA_DIR, provider)
        if len(provider_dates) < window_days:
            if provider == primary_provider:
                raise ValueError(
                    f"Need at least {window_days} docked days for {primary_provider}, "
                    f"found {len(provider_dates)}"
                )
            continue

        bbox_ids = (
            None
            if provider == primary_provider
            else _den_haag_station_ids_for_provider(provider)
        )

        selected_dates = provider_dates[-window_days:]
        common_cols: set[str] | None = None
        for year, month, day in selected_dates:
            day_df = load_docked_day(DATA_DIR, provider, year, month, day)
            if day_df is None:
                continue
            cols = set(day_df.columns.astype(str))
            if bbox_ids is not None:
                cols &= bbox_ids
            common_cols = cols if common_cols is None else common_cols & cols
            if provider == primary_provider:
                primary_dates.append((year, month, day))

        if common_cols:
            all_ids.update(common_cols)

    if not primary_dates or not all_ids:
        raise FileNotFoundError(
            f"No docked snapshots found for recent window of {primary_provider}"
        )

    return primary_dates, sorted(all_ids)


def _load_recent_common_docked_window(
    window_days: int = RECENT_WINDOW_DAYS,
) -> tuple[list[tuple[int, int, int]], pd.DataFrame]:
    primary_provider = PROVIDERS[0]
    primary_dates: list[tuple[int, int, int]] = []
    provider_wides: list[pd.DataFrame] = []

    for provider in PROVIDERS:
        provider_dates = discover_docked_dates(DATA_DIR, provider)
        if len(provider_dates) < window_days:
            if provider == primary_provider:
                raise ValueError(
                    f"Need at least {window_days} docked days for {primary_provider}, "
                    f"found {len(provider_dates)}"
                )
            continue

        bbox_ids = (
            None
            if provider == primary_provider
            else _den_haag_station_ids_for_provider(provider)
        )

        selected_dates = provider_dates[-window_days:]
        frames: list[pd.DataFrame] = []
        common_cols: set[str] | None = None

        for year, month, day in selected_dates:
            day_df = load_docked_day(DATA_DIR, provider, year, month, day)
            if day_df is None:
                continue
            day_df = day_df.copy()
            day_df.columns = day_df.columns.astype(str)
            if bbox_ids is not None:
                day_df = day_df[[c for c in day_df.columns if c in bbox_ids]]
            cols = set(day_df.columns)
            common_cols = cols if common_cols is None else common_cols & cols
            frames.append(day_df)
            if provider == primary_provider:
                primary_dates.append((year, month, day))

        if not frames or not common_cols:
            continue

        ordered_cols = sorted(common_cols)
        provider_wide = pd.concat(
            [frame[ordered_cols] for frame in frames]
        ).sort_index()
        provider_wides.append(provider_wide)

    if not primary_dates or not provider_wides:
        raise FileNotFoundError(
            f"No docked snapshots found for recent window of {primary_provider}"
        )

    wide = pd.concat(provider_wides, axis=1, join="inner").sort_index()
    return primary_dates, wide


def _load_houses_rd() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    houses = load_houses()
    houses_rd = wgs84_to_rd(houses[:, :2])
    house_weights = houses[:, 2]
    return houses, houses_rd, house_weights


def _load_area_layer(name: str) -> gpd.GeoDataFrame:
    if name == "pc4":
        paths = [GEODATA_DIR / "pc4_den_haag.geojson"]
    elif name == "pc6":
        paths = sorted((GEODATA_DIR / "pc6_den_haag").glob("*.geojson"))
    elif name == "buurten":
        paths = [GEODATA_DIR / "buurten_den_haag.geojson"]
    elif name == "wijken":
        paths = [GEODATA_DIR / "wijken_den_haag.geojson"]
    elif name == "k20_zones":
        paths = [GEODATA_DIR / "cmdp_service_zones_k20.geojson"]
    else:
        raise ValueError(f"Unsupported area layer: {name}")

    if not paths:
        raise FileNotFoundError(f"No geodata files found for {name}")

    gdfs = [gpd.read_file(path) for path in paths]
    area = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(area, geometry="geometry", crs=gdfs[0].crs).to_crs(
        "EPSG:28992"
    )


def _ensure_statistical_analysis_dirs() -> None:
    ensure_output_dirs()
    for path in (STAT_ANALYSIS_DIR, STAT_DATA_DIR, STAT_FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _configure_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": SUBTLE_TEXT,
            "ytick.color": SUBTLE_TEXT,
            "text.color": TEXT,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _summarize_temporal(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_days": int(df["date"].nunique()),
        "mean_n_bikes": float(df["n_bikes"].mean()),
        "mean_pct_500m": float(df["pct_within_500m"].mean()),
        "mean_distance": float(df["mean_distance"].mean()),
        "mean_covered_addresses_per_bike_500m": float(
            df["covered_addresses_per_bike_500m"].mean()
        ),
    }


def _density_thresholds() -> tuple[float, float]:
    buurten = load_buurten()
    valid = buurten["bevolkingsdichtheid_inwoners_per_km2"].dropna()
    return (
        float(valid.quantile(1 / 3)),
        float(valid.quantile(2 / 3)),
    )


def _density_label(value: float, t1_max: float, t2_max: float) -> str | None:
    if pd.isna(value):
        return None
    if value <= t1_max:
        return "T1 lowest density"
    if value <= t2_max:
        return "T2 middle density"
    return "T3 highest density"


def _density_label_with_range(label: str, t1_max: float, t2_max: float) -> str:
    if label == "T1 lowest density":
        return f"T1 lowest density (<= {t1_max:.0f}/km2)"
    if label == "T2 middle density":
        return f"T2 middle density ({t1_max:.0f} to {t2_max:.0f}/km2)"
    return f"T3 highest density (> {t2_max:.0f}/km2)"


def build_annual_access_trend() -> None:
    rows = []
    for year, run_tag in FULL_YEAR_RUNS.items():
        temporal = _load_temporal_daily(run_tag)
        inequality = _load_spatial_inequality(run_tag)
        rows.append(
            {
                "year": year,
                "period_type": "full_year",
                "period_label": str(year),
                "run_tag": run_tag,
                **_summarize_temporal(temporal),
                "gini_mean_distance": inequality["gini_mean_distance"],
                "theil_mean_distance": inequality["theil_mean_distance"],
            }
        )

    out = pd.DataFrame(rows).sort_values("year")
    out.to_csv(STAT_DATA_DIR / "1_access_trend_annual.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '1_access_trend_annual.csv'}")


def build_quarterly_access_trend() -> None:
    rows = []
    for year, run_tag in PARTIAL_START_RUNS.items():
        temporal = _load_temporal_daily(run_tag).copy()
        temporal["quarter"] = temporal["date"].dt.quarter
        grouped = (
            temporal.groupby("quarter")
            .agg(
                mean_n_bikes=("n_bikes", "mean"),
                mean_pct_500m=("pct_within_500m", "mean"),
                mean_distance=("mean_distance", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.merge(_load_quarterly_gini(run_tag), on="quarter", how="left")
        grouped["year"] = year
        grouped["quarter_label"] = grouped["quarter"].apply(lambda q: f"Q{q}")
        grouped["x_position"] = grouped["year"] + (grouped["quarter"] - 2.5) * 0.22
        rows.append(grouped)

    for year, run_tag in FULL_YEAR_RUNS.items():
        temporal = _load_temporal_daily(run_tag).copy()
        temporal["quarter"] = temporal["date"].dt.quarter
        grouped = (
            temporal.groupby("quarter")
            .agg(
                mean_n_bikes=("n_bikes", "mean"),
                mean_pct_500m=("pct_within_500m", "mean"),
                mean_distance=("mean_distance", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.merge(_load_quarterly_gini(run_tag), on="quarter", how="left")
        grouped["year"] = year
        grouped["quarter_label"] = grouped["quarter"].apply(lambda q: f"Q{q}")
        grouped["x_position"] = grouped["year"] + (grouped["quarter"] - 2.5) * 0.22
        rows.append(grouped)

    recent_2026 = _load_temporal_daily(RECENT_WINDOW_RUNS[2026]).copy()
    recent_2026["quarter"] = recent_2026["date"].dt.quarter
    grouped_2026 = (
        recent_2026.groupby("quarter")
        .agg(
            mean_n_bikes=("n_bikes", "mean"),
            mean_pct_500m=("pct_within_500m", "mean"),
            mean_distance=("mean_distance", "mean"),
        )
        .reset_index()
    )
    grouped_2026 = grouped_2026.merge(
        _load_quarterly_gini(RECENT_WINDOW_RUNS[2026]), on="quarter", how="left"
    )
    grouped_2026["year"] = 2026
    grouped_2026["quarter_label"] = grouped_2026["quarter"].apply(lambda q: f"Q{q}")
    grouped_2026["x_position"] = (
        grouped_2026["year"] + (grouped_2026["quarter"] - 2.5) * 0.22
    )
    rows.append(grouped_2026)

    out = pd.concat(rows, ignore_index=True).sort_values(["year", "quarter"])
    out.to_csv(STAT_DATA_DIR / "1_access_trend_quarterly.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '1_access_trend_quarterly.csv'}")


def _write_density_outputs(
    t1_max: float,
    t2_max: float,
) -> None:
    buurten = load_buurten()
    summary_rows = []

    for year, run_tag in DENSITY_TIME_RUNS.items():
        coverage = _load_coverage_snapshots(run_tag)
        coverage["quarter"] = coverage["date"].dt.quarter

        buurt_quarter = coverage.groupby(["quarter", "buurt_idx"], as_index=False).agg(
            mean_distance=("mean_distance", "mean"),
            mean_pct_500m=("pct_within_500m", "mean"),
            total_addresses=("total_addresses", "first"),
        )
        buurt_quarter["bevolkingsdichtheid_inwoners_per_km2"] = buurten.loc[
            buurt_quarter["buurt_idx"], "bevolkingsdichtheid_inwoners_per_km2"
        ].values
        buurt_quarter["density_quartile"] = buurt_quarter[
            "bevolkingsdichtheid_inwoners_per_km2"
        ].apply(lambda value: _density_label(value, t1_max, t2_max))
        buurt_quarter["density_quartile_label"] = buurt_quarter[
            "density_quartile"
        ].apply(
            lambda label: (
                _density_label_with_range(label, t1_max, t2_max)
                if label is not None
                else None
            )
        )
        valid = buurt_quarter.dropna(
            subset=["density_quartile", "mean_distance"]
        ).copy()

        grouped = (
            valid.groupby(
                ["quarter", "density_quartile", "density_quartile_label"],
                observed=True,
            )
            .agg(
                mean_distance=("mean_distance", "mean"),
                mean_pct_500m=("mean_pct_500m", "mean"),
                total_addresses=("total_addresses", "sum"),
                n_buurten=("buurt_idx", "nunique"),
            )
            .reset_index()
        )
        grouped["year"] = year
        grouped["run_tag"] = run_tag
        grouped["quarter_label"] = grouped["quarter"].apply(lambda q: f"Q{q}")
        grouped["x_position"] = grouped["year"] + (grouped["quarter"] - 2.5) * 0.22
        summary_rows.append(grouped)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary.sort_values(["year", "quarter", "density_quartile"])
    summary.to_csv(STAT_DATA_DIR / "2_density_divider.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '2_density_divider.csv'}")


def build_density_gap_data() -> None:
    t1_max, t2_max = _density_thresholds()
    _write_density_outputs(t1_max, t2_max)


def build_boundary_leakage_data() -> None:
    _, latest_stations = _load_latest_station_snapshot()
    _, recent_station_ids = _load_recent_common_docked_station_ids()
    _, houses_rd, house_weights = _load_houses_rd()

    latest_stations = latest_stations[
        latest_stations["station_id_str"].isin(recent_station_ids)
    ]
    latest_stations = (
        latest_stations.set_index("station_id_str")
        .loc[recent_station_ids]
        .reset_index(drop=False)
    )
    if latest_stations.empty:
        raise ValueError("No latest stations match recent docked availability columns")

    station_coords = wgs84_to_rd(latest_stations[["lat", "lon"]].to_numpy())
    station_tree = cKDTree(station_coords)
    nearest_dist, nearest_idx = station_tree.query(houses_rd, k=1)
    candidate_lists = station_tree.query_ball_point(houses_rd, r=COVERAGE_RADIUS_M)
    candidate_counts = np.fromiter(
        (len(candidates) for candidates in candidate_lists), dtype=np.int32
    )

    house_points = gpd.GeoDataFrame(
        {"weight": house_weights},
        geometry=gpd.points_from_xy(houses_rd[:, 0], houses_rd[:, 1]),
        crs="EPSG:28992",
    )
    nearest_station_points = gpd.GeoDataFrame(
        {"station_id": latest_stations.iloc[nearest_idx]["station_id"].to_numpy()},
        geometry=gpd.points_from_xy(
            station_coords[nearest_idx, 0], station_coords[nearest_idx, 1]
        ),
        crs="EPSG:28992",
    )
    all_station_points = gpd.GeoDataFrame(
        {"station_id": latest_stations["station_id"].to_numpy()},
        geometry=gpd.points_from_xy(station_coords[:, 0], station_coords[:, 1]),
        crs="EPSG:28992",
    )

    rows = []
    for area_name in ("pc4", "pc6", "buurten", "wijken", "k20_zones"):
        area = _load_area_layer(area_name).reset_index(drop=True)
        house_join = gpd.sjoin(
            house_points, area[["geometry"]], how="left", predicate="within"
        )
        nearest_station_join = gpd.sjoin(
            nearest_station_points, area[["geometry"]], how="left", predicate="within"
        )
        station_join = gpd.sjoin(
            all_station_points, area[["geometry"]], how="left", predicate="within"
        )
        house_area = house_join["index_right"].to_numpy()
        nearest_station_area = nearest_station_join["index_right"].to_numpy()
        station_area = station_join["index_right"].to_numpy()
        valid = ~pd.isna(house_area) & ~pd.isna(nearest_station_area)
        valid_500 = valid & (nearest_dist <= COVERAGE_RADIUS_M)
        mismatch = house_area[valid] != nearest_station_area[valid]
        mismatch_500 = house_area[valid_500] != nearest_station_area[valid_500]
        covered = candidate_counts > 0
        any_cross = np.zeros(len(candidate_lists), dtype=bool)
        no_same_area = np.zeros(len(candidate_lists), dtype=bool)
        nearest_outside = np.zeros(len(candidate_lists), dtype=bool)

        for house_idx, station_indices in enumerate(candidate_lists):
            if not station_indices or pd.isna(house_area[house_idx]):
                continue
            candidate_areas = station_area[station_indices]
            candidate_areas = candidate_areas[~pd.isna(candidate_areas)]
            if len(candidate_areas) == 0:
                continue
            any_cross[house_idx] = np.any(candidate_areas != house_area[house_idx])
            no_same_area[house_idx] = np.all(candidate_areas != house_area[house_idx])
            if (
                not pd.isna(nearest_station_area[house_idx])
                and nearest_station_area[house_idx] != house_area[house_idx]
            ):
                nearest_outside[house_idx] = True

        same_area_covered = covered & ~no_same_area
        rows.append(
            {
                "area_level": area_name,
                "n_areas": int(len(area)),
                "pct_addresses_house_vs_nearest_station_area_mismatch": float(
                    house_weights[valid][mismatch].sum()
                    / house_weights[valid].sum()
                    * 100
                ),
                "pct_addresses_mismatch_within_500m_nearest_station": float(
                    house_weights[valid_500][mismatch_500].sum()
                    / house_weights[valid_500].sum()
                    * 100
                ),
                "pct_covered_addresses_with_any_cross_boundary_station_500m": float(
                    house_weights[covered & any_cross].sum()
                    / house_weights[covered].sum()
                    * 100
                ),
                "pct_covered_addresses_with_no_same_area_station_500m": float(
                    house_weights[covered & no_same_area].sum()
                    / house_weights[covered].sum()
                    * 100
                ),
                "pct_covered_addresses_with_nearest_station_outside_area_500m": float(
                    house_weights[covered & nearest_outside].sum()
                    / house_weights[covered].sum()
                    * 100
                ),
                "pct_addresses_with_same_area_station_500m": float(
                    house_weights[same_area_covered].sum() / house_weights.sum() * 100
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(STAT_DATA_DIR / "4_area_coverage_by_area.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '4_area_coverage_by_area.csv'}")

    area_labels = {
        "pc4": "PC4",
        "pc6": "PC6",
        "buurten": "CBS buurten",
        "wijken": "CBS wijken",
        "k20_zones": "Service zones (K=20)",
    }
    plot_out = pd.DataFrame(
        {
            "area_level": out["area_level"],
            "area_label": out["area_level"].map(area_labels),
            "pct_covered_by_same_area_station": out[
                "pct_addresses_with_same_area_station_500m"
            ],
            "pct_has_outside_area_option": out[
                "pct_covered_addresses_with_any_cross_boundary_station_500m"
            ],
            "pct_only_outside_area_options": out[
                "pct_covered_addresses_with_no_same_area_station_500m"
            ],
            "pct_nearest_station_outside_area": out[
                "pct_covered_addresses_with_nearest_station_outside_area_500m"
            ],
        }
    )
    plot_out.to_csv(STAT_DATA_DIR / "4_area_coverage_plot.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '4_area_coverage_plot.csv'}")


def build_coverage_geometry_data() -> None:
    _, latest_stations = _load_latest_station_snapshot()
    _, recent_wide = _load_recent_common_docked_window()
    _, houses_rd, house_weights = _load_houses_rd()
    total_addresses = float(house_weights.sum())

    latest_stations = latest_stations[
        latest_stations["station_id_str"].isin(recent_wide.columns)
    ].copy()
    latest_stations = latest_stations.set_index("station_id_str").loc[
        recent_wide.columns
    ]
    station_coords = wgs84_to_rd(latest_stations[["lat", "lon"]].to_numpy())
    station_tree = cKDTree(station_coords)
    nearest_dist, _ = station_tree.query(houses_rd, k=1)
    candidate_lists = station_tree.query_ball_point(houses_rd, r=COVERAGE_RADIUS_M)
    candidate_counts = np.fromiter(
        (len(candidates) for candidates in candidate_lists), dtype=np.int32
    )

    distance_rows = []
    for threshold in range(50, 1251, 50):
        pct = float(
            house_weights[nearest_dist <= threshold].sum() / total_addresses * 100
        )
        distance_rows.append(
            {
                "distance_threshold_m": threshold,
                "pct_addresses_within_threshold": pct,
            }
        )
    distance_out = pd.DataFrame(distance_rows)
    distance_out.to_csv(STAT_DATA_DIR / "3_station_coverage_distance.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '3_station_coverage_distance.csv'}")

    overlap_rows = []
    for label, mask in (
        ("0 stations", candidate_counts == 0),
        ("1 station", candidate_counts == 1),
        ("2-4 stations", (candidate_counts >= 2) & (candidate_counts <= 4)),
        ("5-9 stations", (candidate_counts >= 5) & (candidate_counts <= 9)),
        ("10+ stations", candidate_counts >= 10),
    ):
        overlap_rows.append(
            {
                "band": label,
                "pct_addresses_in_band": float(
                    house_weights[mask].sum() / total_addresses * 100
                ),
            }
        )
    overlap_out = pd.DataFrame(overlap_rows)
    overlap_out.to_csv(STAT_DATA_DIR / "3_station_coverage_500m.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / '3_station_coverage_500m.csv'}")

    empty_share = (recent_wide == 0).mean(axis=0)
    unique_weight = np.zeros(len(recent_wide.columns), dtype=float)
    for house_idx, station_indices in enumerate(candidate_lists):
        if len(station_indices) == 1:
            unique_weight[station_indices[0]] += house_weights[house_idx]

    unique_mask = unique_weight > 0
    unique_weighted_empty = float(
        np.sum(unique_weight * empty_share.to_numpy()) / unique_weight.sum() * 100
    )
    annotation_out = pd.DataFrame(
        [
            {
                "metric": "pct_addresses_zero_station_500m",
                "value": float(
                    house_weights[candidate_counts == 0].sum() / total_addresses * 100
                ),
            },
            {
                "metric": "pct_addresses_single_station_500m",
                "value": float(
                    house_weights[candidate_counts == 1].sum() / total_addresses * 100
                ),
            },
            {
                "metric": "pct_addresses_multi_station_500m",
                "value": float(
                    house_weights[candidate_counts >= 2].sum() / total_addresses * 100
                ),
            },
            {
                "metric": "n_stations_with_unique_coverage",
                "value": int(unique_mask.sum()),
            },
            {
                "metric": "weighted_empty_share_for_unique_coverage_addresses",
                "value": unique_weighted_empty,
            },
        ]
    )
    annotation_out.to_csv(
        STAT_DATA_DIR / "3_station_coverage_annotation_values.csv", index=False
    )
    print(f"Wrote {STAT_DATA_DIR / '3_station_coverage_annotation_values.csv'}")


def run_data_step() -> None:
    _ensure_statistical_analysis_dirs()
    build_annual_access_trend()
    build_quarterly_access_trend()
    build_density_gap_data()
    build_boundary_leakage_data()
    build_coverage_geometry_data()


def _load_output_data(name: str) -> pd.DataFrame:
    return pd.read_csv(STAT_DATA_DIR / name)


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    path = STAT_FIGURES_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {path}")


def plot_annual_access_trend() -> None:
    quarterly_df = _load_output_data("1_access_trend_quarterly.csv")
    quarter_ticks = quarterly_df["x_position"].to_numpy()
    quarter_labels = quarterly_df["quarter_label"].tolist()
    year_centers = (
        quarterly_df.groupby("year", as_index=False)["x_position"]
        .mean()
        .sort_values("year")
    )

    panels = [
        ("mean_n_bikes", "Mean docked bikes available", "{:,.0f}"),
        ("mean_pct_500m", "Addresses within 500m of a bike", "{:.1f}%"),
        ("mean_distance", "Mean nearest-bike distance", "{:,.0f} m"),
        (
            "gini_mean_distance",
            "Neighborhood inequality in mean distance (Gini)",
            "{:.2f}",
        ),
    ]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(12.6, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )
    fig.patch.set_facecolor(BACKGROUND)
    fig.subplots_adjust(top=0.93)

    for ax, (column, title, label_fmt) in zip(axes, panels, strict=True):
        _style_axis(ax)
        quarter_values = quarterly_df[column].to_numpy()
        ax.plot(
            quarter_ticks,
            quarter_values,
            color=BLUE,
            linewidth=2.6,
            zorder=1,
        )
        ax.scatter(
            quarter_ticks,
            quarter_values,
            s=30,
            color=BLUE,
            edgecolor=PANEL_BACKGROUND,
            linewidth=0.8,
            zorder=2,
        )
        final_x = quarter_ticks[-1]
        final_y = quarter_values[-1]

        for year in year_centers["year"].to_numpy()[:-1]:
            boundary = year + 0.5
            ax.axvline(boundary, color=GRID, linewidth=1.0, alpha=0.85, zorder=0)

        if column == "mean_pct_500m":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        elif column == "mean_distance":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,} m")
            )
        elif column == "gini_mean_distance":
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
            )

        ax.text(
            0.0,
            1.03,
            title,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=TEXT,
            ha="left",
        )
        ax.text(
            final_x + 0.08,
            final_y,
            label_fmt.format(final_y),
            color=TEXT,
            fontsize=9,
            va="center",
        )

    axes[-1].set_xticks([])
    axes[-1].set_xlim(quarter_ticks.min() - 0.25, quarter_ticks.max() + 0.3)
    axes[-1].set_xlabel("")
    for x_pos, q_label in zip(quarter_ticks, quarter_labels, strict=True):
        axes[-1].text(
            x_pos,
            -0.07,
            q_label,
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=SUBTLE_TEXT,
        )
    for _, row in year_centers.iterrows():
        axes[-1].text(
            row["x_position"],
            -0.19,
            f"{int(row['year'])}",
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=TEXT,
        )
    for year in year_centers["year"].to_numpy()[:-1]:
        boundary = year + 0.5
        axes[-1].axvline(boundary, color=GRID, linewidth=1.0, alpha=0.85, zorder=0)

    fig.text(
        0.08,
        0.98,
        "Access improved from 2021 to 2026",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _save_figure(fig, "1_access_trend.png")


def _density_range_label(group: str, df: pd.DataFrame) -> str:
    label = df.loc[df["density_quartile"] == group, "density_quartile_label"].iloc[0]
    if "<=" in label:
        return f"<= {label.split('<= ', 1)[1].split('/km2', 1)[0]}"
    if ">" in label:
        return f"> {label.split('> ', 1)[1].split('/km2', 1)[0]}"
    return label.split("(", 1)[1].split("/km2", 1)[0]


def plot_annual_density_gap() -> None:
    df = _load_output_data("2_density_divider.csv")
    quarter_ticks = (
        df[["year", "quarter", "quarter_label", "x_position"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])
    )
    year_centers = (
        quarter_ticks.groupby("year", as_index=False)["x_position"]
        .mean()
        .sort_values("year")
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.2, 7.4),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    fig.patch.set_facecolor(BACKGROUND)
    fig.subplots_adjust(top=0.78, bottom=0.18)

    panels = [
        ("mean_distance", "Mean nearest-bike distance"),
        ("mean_pct_500m", "Addresses within 500m of a bike"),
    ]
    order = [
        "T3 highest density",
        "T2 middle density",
        "T1 lowest density",
    ]

    for ax, (column, title) in zip(axes, panels, strict=True):
        _style_axis(ax)
        for group in order:
            sub = df[df["density_quartile"] == group].sort_values("x_position")
            color = DENSITY_COLORS[group]
            label = _density_range_label(group, df)
            ax.plot(
                sub["x_position"],
                sub[column],
                color=color,
                linewidth=2.2,
                zorder=2,
                label=label,
            )
            ax.scatter(
                sub["x_position"],
                sub[column],
                s=36,
                color=color,
                edgecolor=PANEL_BACKGROUND,
                linewidth=0.8,
                zorder=3,
            )

        for year in year_centers["year"].to_numpy()[:-1]:
            boundary = year + 0.5
            ax.axvline(boundary, color=GRID, linewidth=1.0, alpha=0.85, zorder=0)

        if column == "mean_pct_500m":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        else:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,} m")
            )

        ax.text(
            0.0,
            1.03,
            title,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=TEXT,
            ha="left",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        title="People/km2",
        bbox_to_anchor=(0.72, 0.975),
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor=PANEL_BACKGROUND,
        edgecolor=GRID,
        fontsize=9,
        title_fontsize=9,
    )
    legend.get_title().set_color(TEXT)
    for text in legend.get_texts():
        text.set_color(TEXT)

    axes[-1].set_xticks([])
    axes[-1].set_xlim(
        float(quarter_ticks["x_position"].min()) - 0.15,
        float(quarter_ticks["x_position"].max()) + 0.15,
    )
    axes[-1].set_xlabel("")
    for _, row in quarter_ticks.iterrows():
        axes[-1].text(
            row["x_position"],
            -0.07,
            row["quarter_label"],
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=SUBTLE_TEXT,
        )
    for _, row in year_centers.iterrows():
        axes[-1].text(
            row["x_position"],
            -0.19,
            f"{int(row['year'])}",
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=TEXT,
        )
    for year in year_centers["year"].to_numpy()[:-1]:
        boundary = year + 0.5
        axes[-1].axvline(boundary, color=GRID, linewidth=1.0, alpha=0.85, zorder=0)

    fig.text(
        0.08,
        0.975,
        "Density is a divider in access",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.94,
        "Distance is to the nearest available bike (not just station) from address",
        fontsize=9,
        color=SUBTLE_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.912,
        "Mean distance is address-weighted within each CBS buurt,",
        fontsize=9,
        color=SUBTLE_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.89,
        "then averaged across buurten in the same density group",
        fontsize=9,
        color=SUBTLE_TEXT,
        ha="left",
        va="top",
    )

    _save_figure(fig, "2_density_divider.png")


def plot_boundary_leakage() -> None:
    df = _load_output_data("4_area_coverage_plot.csv")

    fig, ax = plt.subplots(figsize=(11.0, 7.5))
    fig.patch.set_facecolor(BACKGROUND)
    _style_axis(ax)

    x = np.arange(len(df))
    width = 0.27
    ax.bar(
        x - width,
        df["pct_covered_by_same_area_station"],
        width=width,
        color=GOLD,
        label="Covered by same-area station",
    )
    ax.bar(
        x,
        df["pct_has_outside_area_option"],
        width=width,
        color=BLUE,
        label="Has outside-area option",
    )
    ax.bar(
        x + width,
        df["pct_nearest_station_outside_area"],
        width=width,
        color=TEAL,
        label="Nearest option is outside",
    )
    ax.bar(
        x + width,
        df["pct_only_outside_area_options"],
        width=width,
        color=ORANGE,
        label="Only outside-area options",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["area_label"])
    ax.set_ylabel("Share of addresses (%)")
    fig.suptitle(
        "Coverage often depends on stations across area boundaries",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 3, 2]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.22),
    )
    fig.subplots_adjust(top=0.78)
    fig.text(
        0.55,
        0.91,
        "Covered = At least one station within 500m (of address).\n\nGold uses all addresses;\nblue/teal/orange use covered addresses as denominator.",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT,
    )
    _save_figure(fig, "4_area_coverage.png")


def plot_coverage_geometry() -> None:
    distance_df = _load_output_data("3_station_coverage_distance.csv")
    overlap_df = _load_output_data("3_station_coverage_500m.csv")
    annotation_df = _load_output_data("3_station_coverage_annotation_values.csv")
    annotation_values = dict(
        zip(annotation_df["metric"], annotation_df["value"], strict=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor(BACKGROUND)
    axes[0].plot(
        distance_df["distance_threshold_m"],
        distance_df["pct_addresses_within_threshold"],
        color=BLUE,
        marker="o",
        linewidth=2,
    )
    axes[0].set_xlabel("Distance to nearest station (m)")
    axes[0].set_ylabel("Addresses within threshold (%)")
    axes[0].set_title("Coverage rises quickly up to about 500 m")
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[0].grid(color=GRID, alpha=0.6)

    bar_x = np.arange(len(overlap_df))
    axes[1].bar(
        bar_x,
        overlap_df["pct_addresses_in_band"],
        color=[RUST, GOLD, TEAL, BLUE, SLATE],
        edgecolor=GRID,
    )
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels(overlap_df["band"])
    axes[1].set_ylabel("Address share (%)")
    axes[1].set_title("Almost 90% addresses have station(s) within 500m")
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color=GRID, alpha=0.6)
    one_station_pct = float(
        overlap_df.loc[overlap_df["band"] == "1 station", "pct_addresses_in_band"].iloc[
            0
        ]
    )
    axes[1].annotate(
        "Current risk of such station\nbeing empty: "
        f"{annotation_values['weighted_empty_share_for_unique_coverage_addresses']:.1f}%",
        xy=(1, one_station_pct),
        xycoords="data",
        xytext=(0.02, 0.96),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT,
        bbox={
            "boxstyle": "square,pad=0.25",
            "facecolor": PANEL_BACKGROUND,
            "edgecolor": GRID,
            "linewidth": 0.8,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": TEXT,
            "linewidth": 1.0,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )

    _save_figure(fig, "3_station_coverage.png")


def run_figures_step() -> None:
    _ensure_statistical_analysis_dirs()
    _configure_theme()
    plot_annual_access_trend()
    plot_annual_density_gap()
    plot_boundary_leakage()
    plot_coverage_geometry()


def main() -> None:
    run_data_step()
    run_figures_step()


if __name__ == "__main__":
    main()
