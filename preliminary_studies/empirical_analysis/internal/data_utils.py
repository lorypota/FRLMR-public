"""
Donkey Bike-Sharing Data Utilities
===================================

Usage:
    Import-only helper module used by empirical_analysis scripts.

Shared functions for extracting and parsing Donkey GBFS data from tar.gz
snapshots. The data files use Python dict syntax (not JSON), so we parse
with ast.literal_eval().

Data structure:
    {base_dir}/{YYYY}/{MM}/{DD}/{HH}/{provider}_fietsData_{YYYYMMDDHHMM}.tar.gz
    Each archive contains: station_information, station_status, free_bike_status,
    system_information, system_pricing_plans, system_hours, system_regions,
    gbfs_versions, vehicle_types
"""

import ast
import csv
import json
import os
import re
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .paths import DEFAULT_DATA_ROOT as REPO_DEFAULT_DATA_ROOT
from .paths import INDEX_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_ROOT = os.environ.get("DONKEY_DATA_ROOT", str(REPO_DEFAULT_DATA_ROOT))

PROVIDER = "donkey_denHaag"

DEN_HAAG_BBOX = {
    "lat_min": 52.03,
    "lat_max": 52.12,
    "lon_min": 4.22,
    "lon_max": 4.38,
}
DEN_HAAG_CENTER = (52.075, 4.30)

AMSTERDAM_PROVIDER = "donkey_am"
AMSTERDAM_BBOX = {
    "lat_min": 52.29,
    "lat_max": 52.41,
    "lon_min": 4.77,
    "lon_max": 5.01,
}
AMSTERDAM_CENTER = (52.3527, 4.8928)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_QUALITY_EVENTS_PATH = INDEX_DIR / "data_quality_events.csv"
DATA_QUALITY_EVENT_FIELDS = [
    "provider",
    "date",
    "timestamp",
    "tar_path",
    "member_name",
    "issue_type",
    "missing_member",
    "consumer",
    "action_taken",
    "note",
]
RAW_TAR_PATH_PATTERN = re.compile(
    r"(?P<year>\d{4})[\\/](?P<month>\d{2})[\\/](?P<day>\d{2})[\\/](?P<hour>\d{2})[\\/](?P<filename>[^\\/]+\.tar\.gz)$"
)

# =============================================================================
# PARSING
# =============================================================================


def parse_gbfs_file(raw_bytes: bytes) -> dict | None:
    """Parse a GBFS data file from raw bytes.

    The archived GBFS payloads are not fully consistent. Some snapshots use
    Python dict syntax (True/False/None, single quotes), while others use
    standard JSON (true/false/null, double quotes).

    Tries UTF-8 first, falls back to Latin-1 for files with accented characters
    (e.g. station names like 'Jozef Israëlslaan').

    Some files are truncated (e.g. ending with "'version': '2.3" instead of
    "'version': '2.3'}"). We attempt to recover by appending closing characters.

    Returns None if parsing fails (e.g. corrupt files like system_pricing_plans).
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(encoding).lstrip("\ufeff").rstrip("\x00").strip()
        except UnicodeDecodeError:
            continue
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            # Try to recover truncated files by appending closing chars
            for suffix in ("'}", "'}}", "'}}}", "}", "}}", "}}}"):
                try:
                    return json.loads(text + suffix)
                except json.JSONDecodeError:
                    pass
                try:
                    return ast.literal_eval(text + suffix)
                except (SyntaxError, ValueError):
                    continue
            return None
    return None


# =============================================================================
# TAR EXTRACTION
# =============================================================================


def extract_file_from_tar(tar_path: str | Path, member_name: str) -> dict | None:
    """Extract and parse a single named member from a tar.gz archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            f = tf.extractfile(member_name)
        except KeyError:
            return None
        if f is None:
            return None
        return parse_gbfs_file(f.read())


def tar_member_exists(tar_path: str | Path, member_name: str) -> bool:
    """Return True when the tar archive contains the given member."""
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.getmember(member_name)
        except KeyError:
            return False
    return True


def format_data_quality_tar_path(tar_path: str | Path) -> str:
    """Return a machine-independent tar path rooted at the YYYY/MM/DD/HH tree."""
    path_str = str(Path(tar_path))
    match = RAW_TAR_PATH_PATTERN.search(path_str)
    if match:
        return (
            f"\\{match.group('year')}\\{match.group('month')}\\"
            f"{match.group('day')}\\{match.group('hour')}\\{match.group('filename')}"
        )
    return Path(tar_path).name


def append_data_quality_event(
    *,
    provider: str,
    tar_path: str | Path,
    member_name: str,
    issue_type: str,
    consumer: str,
    action_taken: str,
    note: str = "",
) -> None:
    """Append one raw-data quality event to the empirical-analysis quality log."""
    tar_path = Path(tar_path)
    timestamp = parse_timestamp_from_filename(tar_path.name, provider=provider)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "provider": provider,
        "date": timestamp.strftime("%Y-%m-%d"),
        "timestamp": timestamp.isoformat(),
        "tar_path": format_data_quality_tar_path(tar_path),
        "member_name": member_name,
        "issue_type": issue_type,
        # Keep legacy column populated for compatibility with older readers.
        "missing_member": member_name if issue_type == "missing_member" else "",
        "consumer": consumer,
        "action_taken": action_taken,
        "note": note,
    }

    if DATA_QUALITY_EVENTS_PATH.exists():
        with DATA_QUALITY_EVENTS_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fields != DATA_QUALITY_EVENT_FIELDS:
            migrated_rows = []
            for row in existing_rows:
                migrated = {field: row.get(field, "") for field in DATA_QUALITY_EVENT_FIELDS}
                if not migrated["member_name"]:
                    migrated["member_name"] = row.get("missing_member", "")
                if not migrated["issue_type"]:
                    migrated["issue_type"] = (
                        "missing_member" if row.get("missing_member", "") else ""
                    )
                migrated_rows.append(migrated)
            with DATA_QUALITY_EVENTS_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=DATA_QUALITY_EVENT_FIELDS)
                writer.writeheader()
                writer.writerows(migrated_rows)

    write_header = not DATA_QUALITY_EVENTS_PATH.exists()

    with DATA_QUALITY_EVENTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATA_QUALITY_EVENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(event)


def extract_all_from_tar(tar_path: str | Path) -> dict[str, dict | None]:
    """Extract and parse all members from a tar.gz archive.

    Returns a dict mapping member name to parsed data (or None if parsing failed).
    """
    result = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            f = tf.extractfile(member)
            if f is not None:
                result[member.name] = parse_gbfs_file(f.read())
            else:
                result[member.name] = None
    return result


# =============================================================================
# FILE LISTING
# =============================================================================


def list_tar_files(
    data_root: str,
    year: int,
    month: int,
    day: int,
    hour: int | None = None,
    provider: str | None = None,
) -> list[Path]:
    """List tar.gz files for a given provider/day (or specific hour).

    Path: {data_root}/{YYYY}/{MM}/{DD}/{HH}/
    Returns sorted list of Path objects.
    """
    if provider is None:
        provider = PROVIDER
    base = Path(data_root) / str(year) / f"{month:02d}" / f"{day:02d}"
    prefix = f"{provider}_fietsData_"

    hours = [hour] if hour is not None else range(24)

    files = []
    for h in hours:
        hour_dir = base / f"{h:02d}"
        if not hour_dir.exists():
            continue
        for f in hour_dir.iterdir():
            if f.name.startswith(prefix) and f.name.endswith(".tar.gz"):
                files.append(f)

    files.sort()
    return files


def parse_timestamp_from_filename(
    filename: str,
    provider: str | None = None,
) -> datetime:
    """Extract datetime from a filename like donkey_denHaag_fietsData_202501010000.tar.gz."""
    if provider is None:
        provider = PROVIDER
    prefix = f"{provider}_fietsData_"
    suffix = ".tar.gz"
    name = filename
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return datetime.strptime(name, "%Y%m%d%H%M")


# =============================================================================
# CONVENIENCE EXTRACTORS
# =============================================================================


def get_station_info(tar_path: str | Path) -> list[dict] | None:
    """Extract station_information and return the list of station dicts."""
    data = extract_file_from_tar(tar_path, "station_information")
    if data is None:
        return None
    return data["data"]["stations"]


def get_station_status(tar_path: str | Path) -> list[dict] | None:
    """Extract station_status and return the list of station status dicts."""
    data = extract_file_from_tar(tar_path, "station_status")
    if data is None:
        return None
    return data["data"]["stations"]


def get_free_bike_status(tar_path: str | Path) -> list[dict] | None:
    """Extract free_bike_status and return the list of bike dicts."""
    data = extract_file_from_tar(tar_path, "free_bike_status")
    if data is None:
        return None
    payload = data.get("data", {})
    if not isinstance(payload, dict):
        return None
    nested_payload = payload.get("data")
    if isinstance(nested_payload, dict):
        payload = nested_payload
    bikes = payload.get("bikes")
    if isinstance(bikes, list):
        return bikes
    vehicles = payload.get("vehicles")
    if isinstance(vehicles, list):
        return vehicles
    return None


# =============================================================================
# FILTERING
# =============================================================================


def filter_by_bbox(items: list[dict], bbox: dict | None = None) -> list[dict]:
    """Filter any list of dicts with 'lat'/'lon' keys by bounding box."""
    if bbox is None:
        bbox = DEN_HAAG_BBOX
    filtered = []
    for s in items:
        try:
            lat = float(s["lat"])
            lon = float(s["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]:
            item = dict(s)
            item["lat"] = lat
            item["lon"] = lon
            filtered.append(item)
    return filtered


def filter_den_haag_stations(
    stations: list[dict], bbox: dict | None = None
) -> list[dict]:
    """Filter stations to those within the Den Haag bounding box."""
    return filter_by_bbox(stations, bbox)


def get_den_haag_station_ids(tar_path: str | Path) -> set[str]:
    """Get the set of station IDs within the Den Haag bounding box."""
    stations = get_station_info(tar_path)
    if stations is None:
        return set()
    filtered = filter_den_haag_stations(stations)
    return {s["station_id"] for s in filtered}


# =============================================================================
# DATAFRAME LOADING
# =============================================================================


def load_day_availability(
    data_root: str,
    year: int = 2025,
    month: int = 1,
    day: int = 1,
    cache_dir: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    """Load all snapshots for a provider/day into a DataFrame.

    Returns DataFrame with:
        - Index: datetime timestamps (one per minute)
        - Columns: station_id strings
        - Values: num_bikes_available (int)

    Also saves a parallel ``docks_<date>.csv`` table with
    ``num_docks_available`` values when *cache_dir* is set.

    If cache_dir is set, saves/loads a derived table for speed.
    """
    if provider is None:
        provider = PROVIDER
    if cache_dir:
        date_tag = f"{year}{month:02d}{day:02d}"
        preferred_cache_path = os.path.join(cache_dir, f"docked_{date_tag}.csv")
        docks_cache_path = os.path.join(cache_dir, f"docks_{date_tag}.csv")
        legacy_tag = "" if provider == PROVIDER else f"{provider}_"
        legacy_paths = [
            os.path.join(cache_dir, f"availability_{date_tag}.csv"),
            os.path.join(cache_dir, f"availability_{legacy_tag}{date_tag}.csv"),
        ]
        if os.path.exists(preferred_cache_path):
            print(f"  Loading saved docked-bike table: {preferred_cache_path}")
            try:
                return pd.read_csv(
                    preferred_cache_path, index_col="timestamp", parse_dates=True
                )
            except pd.errors.EmptyDataError:
                print(
                    f"  [WARN] Saved docked-bike table is empty, rebuilding: "
                    f"{preferred_cache_path}"
                )
        for legacy_path in legacy_paths:
            if legacy_path != preferred_cache_path and os.path.exists(legacy_path):
                print(f"  Loading legacy docked-bike table: {legacy_path}")
                try:
                    return pd.read_csv(
                        legacy_path, index_col="timestamp", parse_dates=True
                    )
                except pd.errors.EmptyDataError:
                    print(
                        f"  [WARN] Legacy docked-bike table is empty, rebuilding: "
                        f"{legacy_path}"
                    )
    else:
        preferred_cache_path = None
        docks_cache_path = None

    tar_files = list_tar_files(data_root, year, month, day, provider=provider)
    print(f"  Found {len(tar_files)} snapshots to process...")

    bikes_records = []
    docks_records = []
    missing_status_count = 0
    unreadable_status_count = 0
    for i, tar_path in enumerate(tar_files):
        if i % 100 == 0:
            print(f"  Processing snapshot {i + 1}/{len(tar_files)}...")
        ts = parse_timestamp_from_filename(tar_path.name, provider=provider)
        statuses = get_station_status(tar_path)
        if statuses is None:
            if tar_member_exists(tar_path, "station_status"):
                unreadable_status_count += 1
                append_data_quality_event(
                    provider=provider,
                    tar_path=tar_path,
                    member_name="station_status",
                    issue_type="unreadable_member",
                    consumer="docked_table",
                    action_taken="skipped_snapshot",
                    note="parse failure while building docked table",
                )
                print(
                    f"    [WARN] station_status unreadable in {tar_path.name}; "
                    "skipping snapshot"
                )
            else:
                missing_status_count += 1
                print(
                    f"    [WARN] station_status missing in {tar_path.name}; "
                    "skipping snapshot"
                )
                append_data_quality_event(
                    provider=provider,
                    tar_path=tar_path,
                    member_name="station_status",
                    issue_type="missing_member",
                    consumer="docked_table",
                    action_taken="skipped_snapshot",
                )
            continue
        bikes_row = {"timestamp": ts}
        docks_row = {"timestamp": ts}
        for s in statuses:
            bikes_row[s["station_id"]] = s["num_bikes_available"]
            docks_row[s["station_id"]] = s.get("num_docks_available", 0)
        bikes_records.append(bikes_row)
        docks_records.append(docks_row)

    if bikes_records:
        df = pd.DataFrame(bikes_records).set_index("timestamp").sort_index()
    else:
        df = pd.DataFrame(columns=["timestamp"]).set_index("timestamp")

    if docks_records:
        docks_df = pd.DataFrame(docks_records).set_index("timestamp").sort_index()
    else:
        docks_df = pd.DataFrame(columns=["timestamp"]).set_index("timestamp")

    if missing_status_count:
        print(
            f"  Skipped {missing_status_count} snapshot(s) with missing station_status"
        )
    if unreadable_status_count:
        print(
            f"  Skipped {unreadable_status_count} snapshot(s) with unreadable "
            "station_status"
        )

    if preferred_cache_path:
        os.makedirs(os.path.dirname(preferred_cache_path), exist_ok=True)
        df.to_csv(preferred_cache_path)
        print(f"  Saved docked-bike table: {preferred_cache_path}")
    if docks_cache_path:
        docks_df.to_csv(docks_cache_path)
        print(f"  Saved docks-available table: {docks_cache_path}")

    return df


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================


def discover_available_dates(
    base_dir: str | None = None,
    provider: str | None = None,
) -> list[tuple[int, int, int]]:
    """Scan base_dir for all dates that have data for the given provider.

    Returns sorted list of (year, month, day) tuples.
    """
    if base_dir is None:
        base_dir = DEFAULT_DATA_ROOT
    if provider is None:
        provider = PROVIDER
    prefix = f"{provider}_fietsData_"
    dates = []
    base = Path(base_dir)
    if not base.exists():
        print(f"Data root does not exist: {base}")
        return dates
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        if year < 2020 or year > 2030:
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            try:
                month = int(month_dir.name)
            except ValueError:
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = int(day_dir.name)
                except ValueError:
                    continue
                has_data = False
                for hour_dir in day_dir.iterdir():
                    if not hour_dir.is_dir():
                        continue
                    for f in hour_dir.iterdir():
                        if f.name.startswith(prefix) and f.name.endswith(".tar.gz"):
                            has_data = True
                            break
                    if has_data:
                        break
                if has_data:
                    dates.append((year, month, day))
    return dates


def discover_available_hours(
    base_dir: str,
    year: int,
    month: int,
    day: int,
    provider: str | None = None,
) -> list[int]:
    """Return sorted list of hours that have data for the given provider/date."""
    if provider is None:
        provider = PROVIDER
    prefix = f"{provider}_fietsData_"
    base = Path(base_dir) / str(year) / f"{month:02d}" / f"{day:02d}"
    hours = []
    if not base.exists():
        return hours
    for hour_dir in sorted(base.iterdir()):
        if not hour_dir.is_dir():
            continue
        try:
            h = int(hour_dir.name)
        except ValueError:
            continue
        has_files = any(
            f.name.startswith(prefix) and f.name.endswith(".tar.gz")
            for f in hour_dir.iterdir()
        )
        if has_files:
            hours.append(h)
    return hours


def load_day_free_bikes(
    data_root: str,
    year: int = 2025,
    month: int = 1,
    day: int = 1,
    bbox: dict | None = None,
    cache_dir: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    """Load all free (dockless) bike positions for a day into a DataFrame.

    Returns DataFrame with:
        - Index: datetime timestamps
        - Columns: bike_id, lat, lon
    Each row is one bike observation at one timestamp.

    If cache_dir is set, saves/loads a derived table for speed.
    """
    if provider is None:
        provider = PROVIDER
    if bbox is None:
        bbox = DEN_HAAG_BBOX

    if cache_dir:
        date_tag = f"{year}{month:02d}{day:02d}"
        preferred_cache_path = os.path.join(cache_dir, f"dockless_{date_tag}.csv")
        legacy_tag = "" if provider == PROVIDER else f"{provider}_"
        legacy_paths = [
            os.path.join(cache_dir, f"free_bikes_{date_tag}.csv"),
            os.path.join(cache_dir, f"free_bikes_{legacy_tag}{date_tag}.csv"),
        ]
        if os.path.exists(preferred_cache_path):
            print(f"  Loading saved dockless-bike table: {preferred_cache_path}")
            try:
                return pd.read_csv(preferred_cache_path, parse_dates=["timestamp"])
            except pd.errors.EmptyDataError:
                print(
                    f"  [WARN] Saved dockless-bike table is empty, rebuilding: "
                    f"{preferred_cache_path}"
                )
        for legacy_path in legacy_paths:
            if legacy_path != preferred_cache_path and os.path.exists(legacy_path):
                print(f"  Loading legacy dockless-bike table: {legacy_path}")
                try:
                    return pd.read_csv(legacy_path, parse_dates=["timestamp"])
                except pd.errors.EmptyDataError:
                    print(
                        f"  [WARN] Legacy dockless-bike table is empty, rebuilding: "
                        f"{legacy_path}"
                    )
    else:
        preferred_cache_path = None

    tar_files = list_tar_files(data_root, year, month, day, provider=provider)
    print(f"  Found {len(tar_files)} snapshots to process...")

    records = []
    for i, tar_path in enumerate(tar_files):
        if i % 100 == 0:
            print(f"  Processing snapshot {i + 1}/{len(tar_files)}...")
        ts = parse_timestamp_from_filename(tar_path.name, provider=provider)
        bikes = get_free_bike_status(tar_path)
        if bikes is None:
            continue
        for b in filter_by_bbox(bikes, bbox):
            bike_id = b.get("bike_id", b.get("vehicle_id"))
            if bike_id is None:
                continue
            records.append(
                {
                    "timestamp": ts,
                    "bike_id": bike_id,
                    "lat": b["lat"],
                    "lon": b["lon"],
                    "is_reserved": b.get("is_reserved", False),
                    "is_disabled": b.get("is_disabled", False),
                    "last_reported": b.get("last_reported"),
                    "station_id": b.get("station_id", b.get("home_station_id", "")),
                    "vehicle_type_id": b.get("vehicle_type_id", ""),
                }
            )

    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "bike_id",
                "lat",
                "lon",
                "is_reserved",
                "is_disabled",
                "last_reported",
                "station_id",
                "vehicle_type_id",
            ]
        )

    if preferred_cache_path:
        os.makedirs(os.path.dirname(preferred_cache_path), exist_ok=True)
        df.to_csv(preferred_cache_path, index=False)
        print(f"  Saved dockless-bike table: {preferred_cache_path}")

    return df
