"""Prepare statistical-analysis tables and thesis figures for key empirical findings.

Usage:
    uv run preliminary_studies/empirical_analysis/statistical_analysis.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from internal.coverage_utils import load_buurten
from internal.paths import OUTPUT_DIR, ensure_output_dirs
from matplotlib import ticker

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preliminary_studies.cmdp_adapted_data.cmdp_adapted_story import (  # noqa: E402
    run_cmdp_adapted_data_step,
    run_cmdp_adapted_figures_step,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORY_DIR = OUTPUT_DIR / "statistical_analysis"
STORY_DATA_DIR = STORY_DIR / "data"
STORY_FIGURES_DIR = STORY_DIR / "figures"
LEGACY_ANALYSIS_DIR = Path(__file__).resolve().parent / "legacy" / "output" / "analysis"

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


def _load_spatial_summary(run_tag: str) -> pd.DataFrame:
    path = _analysis_tables_dir(run_tag) / "spatial_buurt_summary.csv"
    return pd.read_csv(
        path,
        usecols=[
            "buurt_idx",
            "buurtnaam",
            "mean_distance",
            "mean_pct_500m",
            "total_addresses",
            "bevolkingsdichtheid_inwoners_per_km2",
            "personenautos_per_huishouden",
            "gemiddeld_inkomen_per_inwoner",
        ],
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


def _ensure_story_dirs() -> None:
    ensure_output_dirs()
    for path in (STORY_DIR, STORY_DATA_DIR, STORY_FIGURES_DIR):
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
    out.to_csv(STORY_DATA_DIR / "annual_access_trend.csv", index=False)
    logger.info("Wrote %s", STORY_DATA_DIR / "annual_access_trend.csv")


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
    out.to_csv(STORY_DATA_DIR / "quarterly_access_trend.csv", index=False)
    logger.info("Wrote %s", STORY_DATA_DIR / "quarterly_access_trend.csv")


def _write_density_outputs(
    t1_max: float,
    t2_max: float,
) -> None:
    buurten = load_buurten()
    summary_rows = []

    for year, run_tag in DENSITY_TIME_RUNS.items():
        coverage = _load_coverage_snapshots(run_tag)
        coverage["quarter"] = coverage["date"].dt.quarter

        buurt_quarter = (
            coverage.groupby(["quarter", "buurt_idx"], as_index=False)
            .agg(
                mean_distance=("mean_distance", "mean"),
                mean_pct_500m=("pct_within_500m", "mean"),
                total_addresses=("total_addresses", "first"),
            )
        )
        buurt_quarter["bevolkingsdichtheid_inwoners_per_km2"] = buurten.loc[
            buurt_quarter["buurt_idx"], "bevolkingsdichtheid_inwoners_per_km2"
        ].values
        buurt_quarter["density_quartile"] = buurt_quarter[
            "bevolkingsdichtheid_inwoners_per_km2"
        ].apply(lambda value: _density_label(value, t1_max, t2_max))
        buurt_quarter["density_quartile_label"] = buurt_quarter["density_quartile"].apply(
            lambda label: (
                _density_label_with_range(label, t1_max, t2_max)
                if label is not None
                else None
            )
        )
        valid = buurt_quarter.dropna(subset=["density_quartile", "mean_distance"]).copy()

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
    summary.to_csv(STORY_DATA_DIR / "quarterly_density_gap.csv", index=False)
    logger.info("Wrote %s", STORY_DATA_DIR / "quarterly_density_gap.csv")


def build_density_story_data() -> None:
    t1_max, t2_max = _density_thresholds()
    _write_density_outputs(t1_max, t2_max)

    metadata = {
        "density_tercile_thresholds": {
            "t1_max": round(t1_max, 6),
            "t2_max": round(t2_max, 6),
        },
        "files": [
            "annual_access_trend.csv",
            "quarterly_access_trend.csv",
            "quarterly_density_gap.csv",
        ],
    }
    with open(STORY_DATA_DIR / "story_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote %s", STORY_DATA_DIR / "story_metadata.json")


def run_data_step() -> None:
    _ensure_story_dirs()
    build_annual_access_trend()
    build_quarterly_access_trend()
    build_density_story_data()


def _load_story_data(name: str) -> pd.DataFrame:
    return pd.read_csv(STORY_DATA_DIR / name)


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
    path = STORY_FIGURES_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_annual_access_trend() -> None:
    quarterly_df = _load_story_data("quarterly_access_trend.csv")
    quarter_ticks = quarterly_df["x_position"].to_numpy()
    quarter_labels = quarterly_df["quarter_label"].tolist()
    year_centers = (
        quarterly_df.groupby("year", as_index=False)["x_position"]
        .mean()
        .sort_values("year")
    )

    panels = [
        ("mean_n_bikes", "Mean docked bikes available", "{:,.0f}"),
        ("mean_pct_500m", "Addresses within 500 m", "{:.1f}%"),
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
        "Access improved sharply from 2021 Q4 to early 2026",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _save_figure(fig, "annual_access_trend.png")


def _direct_label_last_point(ax, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x + 0.1,
        y,
        text.upper(),
        color=color,
        fontsize=9,
        va="center",
        ha="left",
    )


def plot_annual_density_gap() -> None:
    df = _load_story_data("quarterly_density_gap.csv")
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
        figsize=(12.2, 8.6),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    fig.patch.set_facecolor(BACKGROUND)
    fig.subplots_adjust(top=0.86, bottom=0.16)

    panels = [
        ("mean_distance", "Mean nearest-bike distance"),
        ("mean_pct_500m", "Addresses within 500 m"),
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
            label = group.split(" ", 1)[1]
            ax.plot(sub["x_position"], sub[column], color=color, linewidth=2.2, zorder=2)
            ax.scatter(
                sub["x_position"],
                sub[column],
                s=36,
                color=color,
                edgecolor=PANEL_BACKGROUND,
                linewidth=0.8,
                zorder=3,
            )
            _direct_label_last_point(
                ax,
                float(sub["x_position"].iloc[-1]),
                float(sub[column].iloc[-1]),
                label,
                color,
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

    axes[-1].set_xticks([])
    axes[-1].set_xlim(
        float(quarter_ticks["x_position"].min()) - 0.15,
        float(quarter_ticks["x_position"].max()) + 0.9,
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
    fig.text(
        0.08,
        0.948,
        "Density here means CBS buurt population density, measured in inhabitants per km2 and split into fixed low, middle, and high groups.",
        fontsize=10,
        color=SUBTLE_TEXT,
        ha="left",
        va="top",
    )

    _save_figure(fig, "annual_density_gap.png")


def run_figures_step() -> None:
    _ensure_story_dirs()
    _configure_theme()
    plot_annual_access_trend()
    plot_annual_density_gap()


def main() -> None:
    run_data_step()
    run_figures_step()
    run_cmdp_adapted_data_step()
    run_cmdp_adapted_figures_step()


if __name__ == "__main__":
    main()
