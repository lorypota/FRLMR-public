"""Stage selected providers and snapshots from a GBFS network share.

Usage:
    uv run preliminary_studies/empirical_analysis/stage_gbfs_subset.py `
        --source-root "\\\\tsn.tno.nl\\RA-Data\\SV\\sv-057767\\Feeds\\OpenOV\\GBFS" `
        --start 2026-02-01 --end 2026-02-07 `
        --providers donkey_denHaag ns_ov_fiets

Copies raw tar.gz files from a slow network share into a local folder,
preserving the original YYYY/MM/DD/HH layout so the result can be passed
directly to build_data_tables.py as --data-root.
"""

import argparse
import shutil
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

DEFAULT_STAGE_ROOT = Path(__file__).resolve().parent / "output" / "raw_staging"


def discover_day_files(
    source_root: Path,
    d: date,
    providers: list[str],
) -> tuple[Path, dict[str, list[Path]]]:
    """Find tar files for the requested providers on a given day."""
    day_dir = source_root / str(d.year) / f"{d.month:02d}" / f"{d.day:02d}"
    if not day_dir.exists():
        return day_dir, {}

    prefixes = tuple(f"{p}_fietsData_" for p in providers)
    provider_files: dict[str, list[Path]] = defaultdict(list)
    for hour_dir in sorted(day_dir.iterdir()):
        if not hour_dir.is_dir():
            continue
        for tar_path in sorted(hour_dir.glob("*_fietsData_*.tar.gz")):
            if not tar_path.name.startswith(prefixes):
                continue
            provider = tar_path.name.split("_fietsData_")[0]
            provider_files[provider].append(tar_path)
    return day_dir, dict(sorted(provider_files.items()))


def select_files(paths: list[Path], mode: str) -> list[Path]:
    if mode == "first":
        return paths[:1]
    if mode == "all":
        return paths
    if mode != "first-per-hour":
        raise ValueError(f"Unsupported mode: {mode}")
    selected: list[Path] = []
    seen_hours: set[str] = set()
    for path in paths:
        hour = path.parent.name
        if hour not in seen_hours:
            selected.append(path)
            seen_hours.add(hour)
    return selected


def copy_preserving_tree(
    paths: list[Path],
    source_root: Path,
    dest_root: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Copy files, skipping those already staged. Returns (copied, skipped, bytes)."""
    copied = 0
    skipped = 0
    copied_bytes = 0
    for src in paths:
        dest = dest_root / src.relative_to(source_root)
        if dest.exists():
            skipped += 1
            continue
        copied += 1
        copied_bytes += src.stat().st_size
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    return copied, skipped, copied_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage selected GBFS providers from a network share into a local folder."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root of the raw GBFS date tree.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date inclusive (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        required=True,
        help="Providers to stage (e.g. donkey_denHaag ns_ov_fiets).",
    )
    parser.add_argument(
        "--mode",
        choices=["first", "first-per-hour", "all"],
        default="all",
        help=(
            "How many snapshots to copy per provider per day. "
            "'all' copies every minute (~1400/day). "
            "'first-per-hour' copies 24. "
            "'first' copies 1. Default: all."
        ),
    )
    parser.add_argument(
        "--dest-root",
        default=str(DEFAULT_STAGE_ROOT),
        help="Local destination root for staged raw files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without transferring files.",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit(f"--end ({end}) is before --start ({start})")

    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    num_days = (end - start).days + 1

    if args.dry_run:
        print("Dry run — no files will be copied.")
    print(
        f"Staging {num_days} day(s) from {start} to {end}, "
        f"providers: {', '.join(args.providers)}, mode: {args.mode}\n"
    )

    grand_copied = 0
    grand_skipped = 0
    grand_bytes = 0
    days_missing = []

    d = start
    while d <= end:
        day_dir, provider_files = discover_day_files(
            source_root=source_root,
            d=d,
            providers=args.providers,
        )

        if not provider_files:
            days_missing.append(d)
            d += timedelta(days=1)
            continue

        day_copied = 0
        day_skipped = 0
        day_bytes = 0
        for provider in args.providers:
            paths = provider_files.get(provider, [])
            if not paths:
                continue
            selected = select_files(paths, args.mode)
            copied, skipped, copied_bytes = copy_preserving_tree(
                paths=selected,
                source_root=source_root,
                dest_root=dest_root,
                dry_run=args.dry_run,
            )
            day_copied += copied
            day_skipped += skipped
            day_bytes += copied_bytes

        status = f"{day_copied} new"
        if day_skipped:
            status += f", {day_skipped} skipped"
        print(f"  {d}: {status} ({day_bytes / (1024 * 1024):.1f} MB)")

        grand_copied += day_copied
        grand_skipped += day_skipped
        grand_bytes += day_bytes
        d += timedelta(days=1)

    print()
    if days_missing:
        print(f"Days with no data: {len(days_missing)}")
        if len(days_missing) <= 5:
            for md in days_missing:
                print(f"  {md}")
        else:
            print(f"  {days_missing[0]} ... {days_missing[-1]}")
        print()

    action = "Would stage" if args.dry_run else "Staged"
    print(f"{action} {grand_copied} new file(s), {grand_bytes / (1024 * 1024):.1f} MB")
    if grand_skipped:
        print(f"Skipped {grand_skipped} already-staged file(s)")
    print(f"Destination: {dest_root}")


if __name__ == "__main__":
    main()
