"""Build an index of saved artifacts under empirical_analysis/output.

Run:
    uv run python preliminary_studies/empirical_analysis/artifact_index.py
"""

import argparse
import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from internal.paths import OUTPUT_DIR, ensure_output_dirs

DATE_PATTERN = re.compile(r"(\d{8})")


def _extract_date_from_name(filename: str) -> str:
    match = DATE_PATTERN.search(filename)
    if not match:
        return ""
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def _to_iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_artifacts(output_dir: Path) -> list[dict]:
    rows = []

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.parts[-2:] in (("index", "artifacts.csv"), ("index", "artifacts.json")):
            continue

        rel = path.relative_to(output_dir).as_posix()
        parts = rel.split("/")

        artifact_type = "other"
        provider = ""
        city = ""
        date = _extract_date_from_name(path.name)

        if parts[0] in {"data", "cache"} and len(parts) >= 3:
            provider = parts[2]
            if parts[1] in {"docked", "availability"}:
                artifact_type = "docked_data"
            elif parts[1] in {"dockless", "free_bikes"}:
                artifact_type = "dockless_data"
            elif parts[1] == "stations":
                artifact_type = "stations_data"
        elif parts[0] == "maps":
            artifact_type = "map"
            lower_name = path.name.lower()
            if "den_haag" in lower_name:
                city = "den_haag"
            elif "amsterdam" in lower_name:
                city = "amsterdam"
        elif parts[0] == "geodata":
            artifact_type = "geodata"
            lower_name = path.name.lower()
            if "den_haag" in lower_name:
                city = "den_haag"
            elif "amsterdam" in lower_name:
                city = "amsterdam"

        stat = path.stat()
        rows.append(
            {
                "artifact_type": artifact_type,
                "provider": provider,
                "city": city,
                "date": date,
                "path": rel,
                "size_bytes": stat.st_size,
                "modified_at_utc": _to_iso_utc(stat.st_mtime),
            }
        )

    return rows


def rebuild_artifact_index(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, int]:
    """Scan output_dir and write both CSV and JSON indexes."""
    ensure_output_dirs()
    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    rows = _iter_artifacts(output_dir)
    rows.sort(
        key=lambda r: (
            r["artifact_type"],
            r["provider"],
            r["city"],
            r["date"],
            r["path"],
        )
    )

    csv_path = index_dir / "artifacts.csv"
    json_path = index_dir / "artifacts.json"

    fieldnames = [
        "artifact_type",
        "provider",
        "city",
        "date",
        "path",
        "size_bytes",
        "modified_at_utc",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build artifact index for empirical analysis"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    csv_path, json_path, n_rows = rebuild_artifact_index(Path(args.output_dir))
    print(f"Wrote {n_rows} artifact rows")
    print(f"CSV index: {csv_path}")
    print(f"JSON index: {json_path}")


if __name__ == "__main__":
    main()
