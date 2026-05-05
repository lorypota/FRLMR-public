"""Build docked/dockless/station CSV tables from raw tar snapshots.

Usage:
    uv run preliminary_studies/empirical_analysis/build_data_tables.py --data-root /full/path/to/snapshots

    uv run preliminary_studies/empirical_analysis/build_data_tables.py `
    --data-root preliminary_studies/empirical_analysis/output/raw_staging `
    --providers donkey_denHaag ns_ov_fiets
"""

import argparse
import csv
from pathlib import Path

from internal.artifact_index import rebuild_artifact_index
from internal.data_utils import (
    AMSTERDAM_PROVIDER,
    DEFAULT_DATA_ROOT,
    DEN_HAAG_BBOX,
    PROVIDER,
    append_data_quality_event,
    discover_available_dates,
    extract_file_from_tar,
    get_station_info,
    get_station_status,
    list_tar_files,
    load_day_availability,
    load_day_free_bikes,
    tar_archive_is_readable,
    tar_member_exists,
)
from internal.paths import (
    ensure_output_dirs,
    provider_docked_data_dir,
    provider_dockless_data_dir,
    provider_stations_data_dir,
)

DEFAULT_PROVIDERS = [PROVIDER, AMSTERDAM_PROVIDER]


def build_station_table(
    data_root: Path,
    provider: str,
    year: int,
    month: int,
    day: int,
    stations_dir: Path,
) -> bool:
    """Extract station metadata from first snapshot of the day and save CSV."""
    files = list_tar_files(
        data_root=str(data_root),
        year=year,
        month=month,
        day=day,
        provider=provider,
    )
    if not files:
        return False

    station_info = None
    status = []
    selected_file = None
    for tar_path in files:
        if not tar_archive_is_readable(tar_path):
            append_data_quality_event(
                provider=provider,
                tar_path=tar_path,
                member_name="",
                issue_type="unreadable_archive",
                consumer="station_table",
                action_taken="used_fallback_snapshot",
                note="tar archive could not be opened",
            )
            print(
                f"    [WARN] archive unreadable in {tar_path.name}; "
                "trying next snapshot"
            )
            continue
        station_info = get_station_info(tar_path)
        if station_info:
            selected_file = tar_path
            status = get_station_status(tar_path) or []
            if selected_file != files[0]:
                print(
                    f"    [WARN] Using later snapshot for station metadata: "
                    f"{selected_file.name}"
                )
            break
        if tar_member_exists(tar_path, "station_information"):
            append_data_quality_event(
                provider=provider,
                tar_path=tar_path,
                member_name="station_information",
                issue_type="unreadable_member",
                consumer="station_table",
                action_taken="used_fallback_snapshot",
                note="parse failure while building station metadata",
            )
            print(
                f"    [WARN] station_information unreadable in {tar_path.name}; "
                "trying next snapshot"
            )
            continue
        print(
            f"    [WARN] station_information missing in {tar_path.name}; "
            "trying next snapshot"
        )
        append_data_quality_event(
            provider=provider,
            tar_path=tar_path,
            member_name="station_information",
            issue_type="missing_member",
            consumer="station_table",
            action_taken="used_fallback_snapshot",
        )

    if not station_info or selected_file is None:
        return False

    status_map = {s["station_id"]: s for s in status}

    rows = []
    for station in station_info:
        station_id = station.get("station_id")
        if station_id is None:
            continue
        capacity = station.get("capacity")
        if capacity is None:
            st = status_map.get(station_id, {})
            capacity = st.get("num_bikes_available", 0) + st.get(
                "num_docks_available", 0
            )
        rows.append(
            {
                "station_id": station_id,
                "name": station.get("name", ""),
                "lat": station.get("lat"),
                "lon": station.get("lon"),
                "capacity": int(capacity) if capacity is not None else 0,
                "is_virtual_station": station.get("is_virtual_station", False),
                "region_id": station.get("region_id", ""),
            }
        )

    if not rows:
        return False

    rows.sort(key=lambda r: str(r["station_id"]))
    stations_dir.mkdir(parents=True, exist_ok=True)
    out_path = stations_dir / f"stations_{year}{month:02d}{day:02d}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "name",
                "lat",
                "lon",
                "capacity",
                "is_virtual_station",
                "region_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw snapshots and save docked/dockless/station CSV tables."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=(
            "Full or relative path to raw snapshot root "
            "(format: YYYY/MM/DD/HH/provider_fietsData_*.tar.gz)."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=DEFAULT_PROVIDERS,
        help=f"Providers to process. Default: {' '.join(DEFAULT_PROVIDERS)}",
    )
    parser.add_argument(
        "--skip-dockless",
        action="store_true",
        help="Skip dockless table generation for all providers.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip rebuilding output/index/artifacts.{csv,json} at the end.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    ensure_output_dirs()

    total_docked_days = 0
    total_dockless_days = 0
    total_station_days = 0

    print(f"Data root: {data_root}")
    print(f"Providers: {args.providers}")

    for provider in args.providers:
        dates = discover_available_dates(base_dir=str(data_root), provider=provider)
        print(f"\nProvider {provider}: {len(dates)} day(s) found")
        if not dates:
            continue

        docked_dir = provider_docked_data_dir(provider)
        stations_dir = provider_stations_data_dir(provider)
        docked_dir.mkdir(parents=True, exist_ok=True)
        stations_dir.mkdir(parents=True, exist_ok=True)

        for year, month, day in dates:
            date_str = f"{year}-{month:02d}-{day:02d}"
            print(f"  Building station metadata table for {date_str}...")
            ok = build_station_table(
                data_root=data_root,
                provider=provider,
                year=year,
                month=month,
                day=day,
                stations_dir=stations_dir,
            )
            if ok:
                total_station_days += 1
            else:
                print(f"    [WARN] Station metadata missing for {date_str}")

            print(f"  Building docked table for {date_str}...")
            load_day_availability(
                data_root=str(data_root),
                year=year,
                month=month,
                day=day,
                cache_dir=str(docked_dir),
                provider=provider,
            )
            total_docked_days += 1

            if not args.skip_dockless:
                first_file = list_tar_files(
                    str(data_root), year, month, day, provider=provider
                )
                try:
                    has_free = (
                        first_file
                        and extract_file_from_tar(first_file[0], "free_bike_status")
                        is not None
                    )
                except KeyError:
                    has_free = False
                if has_free:
                    dockless_dir = provider_dockless_data_dir(provider)
                    dockless_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  Building dockless table for {date_str}...")
                    load_day_free_bikes(
                        data_root=str(data_root),
                        year=year,
                        month=month,
                        day=day,
                        bbox=DEN_HAAG_BBOX,
                        cache_dir=str(dockless_dir),
                        provider=provider,
                    )
                    total_dockless_days += 1

    if not args.skip_index:
        csv_path, json_path, n_rows = rebuild_artifact_index()
        print(f"\nArtifact index updated with {n_rows} rows")
        print(f"  {csv_path}")
        print(f"  {json_path}")

    print("\nDone")
    print(f"  Station day tables: {total_station_days}")
    print(f"  Docked day tables: {total_docked_days}")
    if args.skip_dockless:
        print("  Dockless day tables: skipped")
    else:
        print(f"  Dockless day tables: {total_dockless_days}")


if __name__ == "__main__":
    main()
