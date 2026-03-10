"""Helpers for reading processed docked/dockless/station CSV tables.

Usage:
    Import-only helper module used by empirical_analysis scripts.
"""

import re
from pathlib import Path

import pandas as pd

DATE_PATTERN = re.compile(r"(\d{8})")


def _date_tag(year: int, month: int, day: int) -> str:
    return f"{year}{month:02d}{day:02d}"


def _parse_date_tag(tag: str) -> tuple[int, int, int]:
    return int(tag[:4]), int(tag[4:6]), int(tag[6:8])


def _extract_date_from_name(filename: str) -> tuple[int, int, int] | None:
    match = DATE_PATTERN.search(filename)
    if not match:
        return None
    return _parse_date_tag(match.group(1))


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _discover_dates_from_dir(
    directory: Path, prefixes: tuple[str, ...]
) -> list[tuple[int, int, int]]:
    if not directory.exists():
        return []
    dates = set()
    for path in directory.glob("*.csv"):
        if prefixes and not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        parsed = _extract_date_from_name(path.name)
        if parsed is not None:
            dates.add(parsed)
    return sorted(dates)


def discover_docked_dates(
    data_dir: str | Path, provider: str
) -> list[tuple[int, int, int]]:
    base = Path(data_dir)
    return sorted(
        set(
            _discover_dates_from_dir(base / "docked" / provider, ("docked_",))
            + _discover_dates_from_dir(
                base / "availability" / provider, ("availability_",)
            )
        )
    )


def discover_station_dates(
    data_dir: str | Path, provider: str
) -> list[tuple[int, int, int]]:
    base = Path(data_dir)
    return _discover_dates_from_dir(base / "stations" / provider, ("stations_",))


def latest_date(dates: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    if not dates:
        return None
    return max(dates)


def load_docked_day(
    data_dir: str | Path,
    provider: str,
    year: int,
    month: int,
    day: int,
) -> pd.DataFrame | None:
    tag = _date_tag(year, month, day)
    base = Path(data_dir)
    path = _first_existing(
        [
            base / "docked" / provider / f"docked_{tag}.csv",
            base / "availability" / provider / f"availability_{tag}.csv",
            base / "availability" / provider / f"availability_{provider}_{tag}.csv",
        ]
    )
    if path is None:
        return None
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def load_dockless_day(
    data_dir: str | Path,
    provider: str,
    year: int,
    month: int,
    day: int,
) -> pd.DataFrame | None:
    tag = _date_tag(year, month, day)
    base = Path(data_dir)
    path = _first_existing(
        [
            base / "dockless" / provider / f"dockless_{tag}.csv",
            base / "free_bikes" / provider / f"free_bikes_{tag}.csv",
            base / "free_bikes" / provider / f"free_bikes_{provider}_{tag}.csv",
        ]
    )
    if path is None:
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_station_day(
    data_dir: str | Path,
    provider: str,
    year: int,
    month: int,
    day: int,
) -> pd.DataFrame | None:
    tag = _date_tag(year, month, day)
    path = Path(data_dir) / "stations" / provider / f"stations_{tag}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
