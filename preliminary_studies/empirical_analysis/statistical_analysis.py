"""Prepare statistical-analysis tables and thesis figures for key empirical findings.

Usage:
    uv run preliminary_studies/empirical_analysis/statistical_analysis.py
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
STAT_DATA_DIR = STAT_ANALYSIS_DIR / "data"
STAT_FIGURES_DIR = STAT_ANALYSIS_DIR / "figures"
LEGACY_ANALYSIS_DIR = Path(__file__).resolve().parent / "legacy" / "output" / "analysis"
PROVIDER = "donkey_denHaag"
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
    return LEGACY_ANALYSIS_DIR / run_tag / "tables"


def _coverage_dir(run_tag: str) -> Path:
    return LEGACY_ANALYSIS_DIR / run_tag / "buurt_hour_coverage"


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
    df = pd.read_csv(path)
    return {row["metric"]: float(row["value"]) for _, row in df.iterrows()}


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


def _load_latest_station_snapshot() -> tuple[tuple[int, int, int], pd.DataFrame]:
    latest_station = latest_date(discover_station_dates(DATA_DIR, PROVIDER))
    if latest_station is None:
        raise FileNotFoundError(f"No station snapshots found for {PROVIDER}")

    stations = load_station_day(DATA_DIR, PROVIDER, *latest_station)
    if stations is None:
        raise FileNotFoundError(f"Failed to load station snapshot for {latest_station}")

    stations = stations.copy()
    stations["station_id_str"] = stations["station_id"].astype(str)
    return latest_station, stations


def _load_recent_common_docked_station_ids(
    window_days: int = RECENT_WINDOW_DAYS,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    all_dates = discover_docked_dates(DATA_DIR, PROVIDER)
    if len(all_dates) < window_days:
        raise ValueError(
            f"Need at least {window_days} docked days, found {len(all_dates)}"
        )

    selected_dates = all_dates[-window_days:]
    common_cols: set[str] | None = None
    loaded_dates = []

    for year, month, day in selected_dates:
        day_df = load_docked_day(DATA_DIR, PROVIDER, year, month, day)
        if day_df is None:
            continue
        cols = set(day_df.columns.astype(str))
        common_cols = cols if common_cols is None else common_cols & cols
        loaded_dates.append((year, month, day))

    if not loaded_dates or common_cols is None:
        raise FileNotFoundError(
            f"No docked snapshots found for recent window of {PROVIDER}"
        )

    return loaded_dates, sorted(common_cols)


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
    out.to_csv(STAT_DATA_DIR / "annual_access_trend.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / 'annual_access_trend.csv'}")


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
    out.to_csv(STAT_DATA_DIR / "quarterly_access_trend.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / 'quarterly_access_trend.csv'}")


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
    summary.to_csv(STAT_DATA_DIR / "quarterly_density_gap.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / 'quarterly_density_gap.csv'}")


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
    for area_name in ("pc4", "pc6", "buurten", "wijken"):
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
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(STAT_DATA_DIR / "boundary_leakage_by_area.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / 'boundary_leakage_by_area.csv'}")

    area_labels = {
        "pc4": "PC4",
        "pc6": "PC6",
        "buurten": "CBS buurten",
        "wijken": "CBS wijken",
    }
    plot_out = pd.DataFrame(
        {
            "area_level": out["area_level"],
            "area_label": out["area_level"].map(area_labels),
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
    plot_out.to_csv(STAT_DATA_DIR / "boundary_leakage_plot.csv", index=False)
    print(f"Wrote {STAT_DATA_DIR / 'boundary_leakage_plot.csv'}")


def run_data_step() -> None:
    _ensure_statistical_analysis_dirs()
    build_annual_access_trend()
    build_quarterly_access_trend()
    build_density_gap_data()
    build_boundary_leakage_data()


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
    quarterly_df = _load_output_data("quarterly_access_trend.csv")
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
    _save_figure(fig, "annual_access_trend.png")


def _density_range_label(group: str, df: pd.DataFrame) -> str:
    label = df.loc[df["density_quartile"] == group, "density_quartile_label"].iloc[0]
    if "<=" in label:
        return f"<= {label.split('<= ', 1)[1].split('/km2', 1)[0]}"
    if ">" in label:
        return f"> {label.split('> ', 1)[1].split('/km2', 1)[0]}"
    return label.split("(", 1)[1].split("/km2", 1)[0]


def plot_annual_density_gap() -> None:
    df = _load_output_data("quarterly_density_gap.csv")
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
    fig.subplots_adjust(top=0.82, bottom=0.18)

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
        bbox_to_anchor=(0.58, 0.975),
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
        "Density is a persistent divider in access",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )

    _save_figure(fig, "annual_density_gap.png")


def plot_boundary_leakage() -> None:
    df = _load_output_data("boundary_leakage_plot.csv")

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.patch.set_facecolor(BACKGROUND)
    _style_axis(ax)

    x = np.arange(len(df))
    width = 0.34
    ax.bar(
        x - width / 2,
        df["pct_has_outside_area_option"],
        width=width,
        color=BLUE,
        label="Has outside-area option",
    )
    ax.bar(
        x + width / 2,
        df["pct_nearest_station_outside_area"],
        width=width,
        color=TEAL,
        label="Nearest option is outside",
    )
    ax.bar(
        x + width / 2,
        df["pct_only_outside_area_options"],
        width=width,
        color=ORANGE,
        label="Only outside-area options",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["area_label"])
    ax.set_ylabel("Share of covered addresses (%)")
    ax.set_title("Coverage often depends on stations across area boundaries")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        loc="upper right",
    )
    fig.text(
        0.125,
        0.02,
        "Covered addresses have at least one station within 500m.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=SUBTLE_TEXT,
    )
    _save_figure(fig, "boundary_leakage.png")


def run_figures_step() -> None:
    _ensure_statistical_analysis_dirs()
    _configure_theme()
    plot_annual_access_trend()
    plot_annual_density_gap()
    plot_boundary_leakage()


def main() -> None:
    run_data_step()
    run_figures_step()


if __name__ == "__main__":
    main()
