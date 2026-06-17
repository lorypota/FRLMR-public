"""Write the LaTeX zone-attributes table (SI) from the service-zone outputs.

Reads the assignment and density-profile CSVs and prints a ready-to-paste
LaTeX table, rows ordered by category then service-pressure score.

Usage:
    uv run research_support/service_zone_calculation/generate_zone_table.py
"""

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TAG = "z20_cat5"


def load_rows() -> list[dict]:
    with (OUTPUT_DIR / f"service_zone_assignments_{TAG}.csv").open(
        encoding="utf-8"
    ) as f:
        assignments = list(csv.DictReader(f))
    with (OUTPUT_DIR / f"service_zone_density_profile_{TAG}.csv").open(
        encoding="utf-8"
    ) as f:
        profile = list(csv.DictReader(f))

    stations: dict[int, int] = defaultdict(int)
    capacity: dict[int, int] = defaultdict(int)
    for row in assignments:
        zone = int(row["service_zone"])
        stations[zone] += 1
        capacity[zone] += int(float(row["capacity"] or 0))

    rows = [
        {
            "zone": int(r["service_zone"]),
            "cat": int(float(r["service_category"])),
            "stations": stations[int(r["service_zone"])],
            "capacity": capacity[int(r["service_zone"])],
            "dep": float(r["departure_score"]),
            "dens": float(r["density_score"]),
            "act": float(r["activity_score"]),
            "score": float(r["service_pressure_score"]),
        }
        for r in profile
    ]
    rows.sort(key=lambda x: (x["cat"], x["score"]))
    return rows


def main() -> None:
    rows = load_rows()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Service zones by category, with station count, aggregate capacity, the three normalized scores ($\tilde{\lambda}_d$ departures, $\tilde{\rho}$ address density, $\tilde{\beta}$ non-residential activity), and the service-pressure score}",
        r"\label{tab:si:zones}",
        r"\begin{tabular}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"Zone & Category & Stations & Capacity & $\tilde{\lambda}_d$ & "
        r"$\tilde{\rho}$ & $\tilde{\beta}$ & Score \\",
        r"\midrule",
    ]
    prev_cat = None
    for x in rows:
        if prev_cat is not None and x["cat"] != prev_cat:
            lines.append(r"\midrule")
        lines.append(
            f"{x['zone']} & {x['cat']} & {x['stations']} & {x['capacity']} & "
            f"{x['dep']:.3f} & {x['dens']:.3f} & {x['act']:.3f} & {x['score']:.3f} \\\\"
        )
        prev_cat = x["cat"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
