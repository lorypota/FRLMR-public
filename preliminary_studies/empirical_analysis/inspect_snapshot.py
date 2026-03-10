"""
Inspect a Single Donkey Snapshot
=========================================

Run:
    uv run python preliminary_studies/empirical_analysis/inspect_snapshot.py
    uv run python preliminary_studies/empirical_analysis/inspect_snapshot.py --year 2025 --month 1 --day 1 --hour 0

Extracts one donkey_denHaag tar.gz file and prints the contents of each
data file. Produces a structured summary answering:
- Does Donkey have station_status, free_bike_status, or both?
- What fields are in station_information? (need lat, lon, station_id, name, capacity)
- What fields are in station_status? (need num_bikes_available, num_docks_available)
- How many stations/vehicles are there in total?

Output: terminal only (no files saved)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internal.data_utils import DEFAULT_DATA_ROOT, extract_all_from_tar, list_tar_files

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Explore a single Donkey snapshot")
    parser.add_argument(
        "--data-root", default=DEFAULT_DATA_ROOT, help="Root directory of the data"
    )
    parser.add_argument("--year", type=int, default=2025, help="Year of snapshot")
    parser.add_argument("--month", type=int, default=1, help="Month of snapshot")
    parser.add_argument("--day", type=int, default=1, help="Day of snapshot")
    parser.add_argument("--hour", type=int, default=0, help="Hour of snapshot")
    args = parser.parse_args()

    # Find the first snapshot for the given hour
    files = list_tar_files(args.data_root, args.year, args.month, args.day, args.hour)
    if not files:
        print(
            f"No donkey_denHaag files found for {args.year}-{args.month:02d}-{args.day:02d} hour {args.hour:02d}"
        )
        return

    tar_path = files[0]
    print(f"Extracting: {tar_path.name}")
    print(f"Full path: {tar_path}")
    print()

    # Extract all members
    all_data = extract_all_from_tar(tar_path)

    # Print each member
    for name, data in all_data.items():
        print("=" * 70)
        print(f"  {name}")
        print("=" * 70)
        if data is None:
            print("  [PARSE FAILED - file may be corrupt]")
        else:
            # Pretty-print with truncation for large outputs
            formatted = json.dumps(data, indent=2, default=str)
            lines = formatted.split("\n")
            if len(lines) > 50:
                for line in lines[:30]:
                    print(f"  {line}")
                print(f"  ... ({len(lines) - 50} lines omitted) ...")
                for line in lines[-20:]:
                    print(f"  {line}")
            else:
                for line in lines:
                    print(f"  {line}")
        print()

    # =================================================================
    # STRUCTURED SUMMARY
    # =================================================================
    print()
    print("=" * 70)
    print("  DATA SUMMARY")
    print("=" * 70)

    # Check which data files exist
    has_station_info = all_data.get("station_information") is not None
    has_station_status = all_data.get("station_status") is not None
    has_free_bike = all_data.get("free_bike_status") is not None
    has_vehicle_status = "vehicle_status" in all_data

    print(f"\n  Has station_information:  {has_station_info}")
    print(f"  Has station_status:       {has_station_status}")
    print(f"  Has free_bike_status:     {has_free_bike}")
    print(f"  Has vehicle_status:       {has_vehicle_status}")
    print(f"\n  All members in archive:   {list(all_data.keys())}")

    # Station information details
    if has_station_info:
        stations = all_data["station_information"]["data"]["stations"]
        print("\n  --- station_information ---")
        print(f"  Number of stations: {len(stations)}")
        print(f"  Fields: {list(stations[0].keys())}")
        required_fields = ["lat", "lon", "station_id", "name", "capacity"]
        for field in required_fields:
            present = field in stations[0]
            example = stations[0].get(field, "N/A")
            print(
                f"    '{field}': {'PRESENT' if present else 'MISSING'} (example: {example})"
            )

        # Capacity distribution
        capacities = [s["capacity"] for s in stations]
        from collections import Counter

        cap_counts = Counter(capacities)
        print("\n  Capacity distribution:")
        for cap in sorted(cap_counts.keys()):
            print(f"    capacity={cap}: {cap_counts[cap]} stations")

    # Station status details
    if has_station_status:
        statuses = all_data["station_status"]["data"]["stations"]
        print("\n  --- station_status ---")
        print(f"  Number of station status records: {len(statuses)}")
        print(f"  Fields: {list(statuses[0].keys())}")
        required_fields = ["num_bikes_available", "num_docks_available", "station_id"]
        for field in required_fields:
            present = field in statuses[0]
            example = statuses[0].get(field, "N/A")
            print(
                f"    '{field}': {'PRESENT' if present else 'MISSING'} (example: {example})"
            )

        # Availability distribution
        avail = [s["num_bikes_available"] for s in statuses]
        print("\n  Availability stats:")
        print(f"    Total bikes across all stations: {sum(avail)}")
        print(f"    Stations with 0 bikes: {sum(1 for a in avail if a == 0)}")
        print(f"    Min availability: {min(avail)}")
        print(f"    Max availability: {max(avail)}")

    # Free bike status details
    if has_free_bike:
        bikes = all_data["free_bike_status"]["data"]["bikes"]
        print("\n  --- free_bike_status ---")
        print(f"  Number of free-floating bikes: {len(bikes)}")
        print(f"  Fields: {list(bikes[0].keys())}")

    # Cross-reference station IDs
    if has_station_info and has_station_status:
        info_ids = {
            s["station_id"] for s in all_data["station_information"]["data"]["stations"]
        }
        status_ids = {
            s["station_id"] for s in all_data["station_status"]["data"]["stations"]
        }
        print("\n  --- Cross-reference ---")
        print(f"  station_information IDs: {len(info_ids)}")
        print(f"  station_status IDs:      {len(status_ids)}")
        print(f"  IDs match exactly:       {info_ids == status_ids}")

    print()


if __name__ == "__main__":
    main()
