"""
Check Temporal Coverage
================================

Run:
    uv run python preliminary_studies/empirical_analysis/check_temporal_coverage.py
    uv run python preliminary_studies/empirical_analysis/check_temporal_coverage.py --data-root /full/path/to/snapshots

Analyzes the directory structure of the Donkey data to understand:
- What date range is available
- How many snapshots per hour/day
- Whether there are any gaps (missing minutes)

Output: terminal only (no files saved)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internal.data_utils import DEFAULT_DATA_ROOT, PROVIDER, parse_timestamp_from_filename

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Check temporal coverage of Donkey data"
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        return

    print(f"Data root: {data_root}")
    print()

    # Walk the directory structure: {data_root}/{YYYY}/{MM}/{DD}/{HH}/
    total_donkey_snapshots = 0
    total_all_files = 0
    all_providers = set()
    days_found = []
    missing_minutes = []

    # List year directories
    year_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    for year_dir in year_dirs:
        year_name = year_dir.name
        months = sorted([d for d in year_dir.iterdir() if d.is_dir()])
        print(f"Year {year_name}/ ({len(months)} months)")

        for month_dir in months:
            month = month_dir.name
            days = sorted([d for d in month_dir.iterdir() if d.is_dir()])
            print(f"  {month}/ ({len(days)} days)")

            for day_dir in days:
                day = day_dir.name
                hours = sorted([d for d in day_dir.iterdir() if d.is_dir()])
                day_donkey_count = 0
                day_all_count = 0
                day_missing = []

                for hour_dir in hours:
                    hour = hour_dir.name
                    all_files = list(hour_dir.iterdir())
                    donkey_files = [
                        f
                        for f in all_files
                        if f.name.startswith(f"{PROVIDER}_fietsData_")
                        and f.name.endswith(".tar.gz")
                    ]

                    # Track providers
                    for f in all_files:
                        if f.name.endswith(".tar.gz"):
                            provider = f.name.rsplit("_fietsData_", 1)[0]
                            all_providers.add(provider)

                    day_donkey_count += len(donkey_files)
                    day_all_count += len(all_files)

                    # Check for missing minutes
                    if donkey_files:
                        minutes_present = set()
                        for f in donkey_files:
                            ts = parse_timestamp_from_filename(f.name)
                            minutes_present.add(ts.minute)
                        expected_minutes = set(range(60))
                        missing = expected_minutes - minutes_present
                        if missing:
                            for m in sorted(missing):
                                day_missing.append(f"{hour}:{m:02d}")

                total_donkey_snapshots += day_donkey_count
                total_all_files += day_all_count
                days_found.append(f"{year_name}-{month}-{day}")

                status = (
                    "OK" if not day_missing else f"GAPS: {len(day_missing)} missing"
                )
                print(
                    f"    {day}/ - {day_donkey_count} donkey snapshots, {day_all_count} total files [{status}]"
                )

                if day_missing and len(day_missing) <= 10:
                    print(f"         Missing: {day_missing}")
                elif day_missing:
                    print(
                        f"         Missing: {day_missing[:5]} ... (+{len(day_missing) - 5} more)"
                    )

                missing_minutes.extend(day_missing)

    # =================================================================
    # COVERAGE REPORT
    # =================================================================
    if not days_found:
        print("No data found.")
        return

    print()
    print("=" * 60)
    print("  COVERAGE REPORT")
    print("=" * 60)
    print(f"\n  Date range: {days_found[0]} to {days_found[-1]}")
    print(f"  Total days: {len(days_found)}")
    print(f"  Providers found: {sorted(all_providers)}")
    print(f"\n  Total {PROVIDER} snapshots: {total_donkey_snapshots}")
    print(f"  Total files (all providers): {total_all_files}")
    print()


if __name__ == "__main__":
    main()
