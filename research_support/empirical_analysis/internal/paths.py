"""Shared output paths for empirical analysis artifacts.

Usage:
    Import-only helper module used by empirical_analysis scripts.
"""

from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
PROJECT_ROOT = ROOT_DIR.parents[1]

DATA_DIR = OUTPUT_DIR / "data"
DATA_DOCKED_DIR = DATA_DIR / "docked"
DATA_DOCKLESS_DIR = DATA_DIR / "dockless"
DATA_STATIONS_DIR = DATA_DIR / "stations"
MAPS_DIR = OUTPUT_DIR / "maps"
GEODATA_DIR = OUTPUT_DIR / "geodata"
INDEX_DIR = OUTPUT_DIR / "index"

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def ensure_output_dirs() -> None:
    """Create the output directory tree if it does not exist."""
    for path in (
        DATA_DOCKED_DIR,
        DATA_DOCKLESS_DIR,
        DATA_STATIONS_DIR,
        MAPS_DIR,
        GEODATA_DIR,
        INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def analysis_run_tag(start_date: date | None, end_date: date | None) -> str:
    if start_date is not None and end_date is not None:
        return f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    if start_date is not None:
        return f"from_{start_date.strftime('%Y%m%d')}"
    if end_date is not None:
        return f"to_{end_date.strftime('%Y%m%d')}"
    return "all_dates"


def provider_docked_data_dir(provider: str) -> Path:
    return DATA_DOCKED_DIR / provider


def provider_dockless_data_dir(provider: str) -> Path:
    return DATA_DOCKLESS_DIR / provider


def provider_stations_data_dir(provider: str) -> Path:
    return DATA_STATIONS_DIR / provider
