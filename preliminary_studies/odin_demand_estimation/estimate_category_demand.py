"""Estimate ODiN category-period demand rates for a Den Haag CMDP.

The script reads ODiN from PostgreSQL, maps PC4 origins/destinations to the
existing service categories, and writes category-period arrival/departure rates
that can be used as Skellam parameters.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - gives a clear CLI error
    raise SystemExit("Missing PostgreSQL dependency.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "pc4_period_demand_rates.csv"

SUPPORTED_YEARS = (2022, 2023, 2024)
REQUIRED_COLUMNS = {
    "opid",
    "verplid",
    "verpl",
    "vertpc",
    "aankpc",
    "vertgem",
    "aankgem",
    "afstv",
    "khvm",
    "vertuur",
    "factorv",
}

COLUMN_CANDIDATES = {
    "opid": ("opid",),
    "verplid": ("verplid",),
    "verpl": ("verpl",),
    "vertpc": ("vertpc", "vertpc_pram"),
    "aankpc": ("aankpc", "aankpc_pram"),
    "vertgem": ("vertgem", "vertgem_dans24"),
    "aankgem": ("aankgem", "aankgem_dans24"),
    "afstv": ("afstv",),
    "khvm": ("khvm",),
    "vertuur": ("vertuur",),
    "factorv": ("factorv",),
}

SCENARIOS = {
    "bike_suitable_all_modes": None,
    "bike_suitable_car_driver": 1,
    "bike_suitable_current_bike": 5,
}

PERIOD_HOURS = {
    "morning": 12,
    "evening": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ODiN demand rates by service category and period."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024],
        help="ODiN years to query. Supported: 2022, 2023, 2024.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for light smoke tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_local_env_file() -> None:
    """Load KEY=VALUE pairs from a local .env file without overwriting env vars."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def db_config_from_env() -> dict[str, str | int]:
    load_local_env_file()
    required = [
        "ODIN_DB_HOST",
        "ODIN_DB_PORT",
        "ODIN_DB_NAME",
        "ODIN_DB_USER",
        "ODIN_DB_PASSWORD",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required database environment variables: {joined}")

    return {
        "host": os.environ["ODIN_DB_HOST"],
        "port": int(os.environ["ODIN_DB_PORT"]),
        "dbname": os.environ["ODIN_DB_NAME"],
        "user": os.environ["ODIN_DB_USER"],
        "password": os.environ["ODIN_DB_PASSWORD"],
    }


def normalize_pc4(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        pc4 = str(int(float(value))).zfill(4)
    except (TypeError, ValueError):
        pc4 = str(value).strip()
    if pc4 in {"", "0", "0000", "9999", "-2"}:
        return None
    return pc4.zfill(4)


def get_table_columns(conn: psycopg.Connection, year: int) -> set[str]:
    query = """
        SELECT lower(column_name) AS column_name
        FROM information_schema.columns
        WHERE table_schema = 'odin'
          AND table_name = %(table_name)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, {"table_name": f"odin{year}"})
        return {row["column_name"] for row in cur.fetchall()}


def resolve_columns(conn: psycopg.Connection, year: int) -> dict[str, str]:
    available = get_table_columns(conn, year)
    resolved = {}
    missing = []
    for logical_name, candidates in COLUMN_CANDIDATES.items():
        match = next(
            (candidate for candidate in candidates if candidate in available), None
        )
        if match is None:
            missing.append(f"{logical_name} ({', '.join(candidates)})")
        else:
            resolved[logical_name] = match

    if missing:
        joined = "; ".join(missing)
        raise SystemExit(f"odin.odin{year} is missing required columns: {joined}")
    return resolved


def fetch_year(conn: psycopg.Connection, year: int, limit: int | None) -> pd.DataFrame:
    if year not in SUPPORTED_YEARS:
        supported = ", ".join(str(y) for y in SUPPORTED_YEARS)
        raise SystemExit(f"Unsupported year {year}. Supported years: {supported}")

    columns = resolve_columns(conn, year)
    limit_clause = "" if limit is None else "LIMIT %(limit)s"
    query = f"""
        SELECT
            {columns["opid"]} AS opid,
            {columns["verplid"]} AS verplid,
            {columns["verpl"]} AS verpl,
            {columns["vertpc"]} AS vertpc,
            {columns["aankpc"]} AS aankpc,
            {columns["vertgem"]} AS vertgem,
            {columns["aankgem"]} AS aankgem,
            {columns["afstv"]} AS afstv,
            {columns["khvm"]} AS khvm,
            {columns["vertuur"]} AS vertuur,
            {columns["factorv"]} AS factorv
        FROM odin.odin{year}
        WHERE {columns["verpl"]} = 1
          AND {columns["afstv"]} BETWEEN 5 AND 100
          AND ({columns["vertgem"]} = 518 OR {columns["aankgem"]} = 518)
        {limit_clause}
    """
    params = {"limit": limit} if limit is not None else None
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(f"ODiN query result is missing required columns: {joined}")

    df["year"] = year
    return df


def period_from_hour(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    hour = int(value)
    if 0 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 23:
        return "evening"
    return None


def prepare_trips(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    trips = df.copy()
    trips["origin_pc4"] = trips["vertpc"].map(normalize_pc4)
    trips["destination_pc4"] = trips["aankpc"].map(normalize_pc4)
    trips["period"] = trips["vertuur"].map(period_from_hour)
    trips["factorv"] = pd.to_numeric(trips["factorv"], errors="coerce")
    trips["khvm"] = pd.to_numeric(trips["khvm"], errors="coerce")
    trips = trips.dropna(subset=["period", "factorv"])
    return trips


def aggregate_direction(
    trips: pd.DataFrame,
    pc4_col: str,
    value_name: str,
) -> pd.DataFrame:
    valid = trips.dropna(subset=[pc4_col])
    if valid.empty:
        return pd.DataFrame(
            columns=["pc4", "period", value_name, "n_trips", "n_persons"]
        )

    grouped = (
        valid.groupby([pc4_col, "period"], as_index=False)
        .agg(
            **{
                value_name: ("factorv", "sum"),
                "n_trips": ("verplid", "nunique"),
                "n_persons": ("opid", "nunique"),
            }
        )
        .rename(columns={pc4_col: "pc4"})
    )
    return grouped


def complete_pc4_period_grid(df: pd.DataFrame, pc4_values: list[str]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [pc4_values, ["morning", "evening"]],
        names=["pc4", "period"],
    ).to_frame(index=False)
    return grid.merge(df, on=["pc4", "period"], how="left")


def aggregate_scenarios(trips: pd.DataFrame) -> pd.DataFrame:
    pc4_values = sorted(
        set(trips["origin_pc4"].dropna()) | set(trips["destination_pc4"].dropna())
    )
    outputs = []
    for scenario, khvm_code in SCENARIOS.items():
        subset = trips if khvm_code is None else trips[trips["khvm"] == khvm_code]

        departures = aggregate_direction(subset, "origin_pc4", "weighted_departures")
        arrivals = aggregate_direction(subset, "destination_pc4", "weighted_arrivals")

        merged = complete_pc4_period_grid(
            departures.merge(
                arrivals,
                on=["pc4", "period"],
                how="outer",
                suffixes=("_departures", "_arrivals"),
            ),
            pc4_values,
        )

        merged["scenario"] = scenario
        outputs.append(merged)

    result = pd.concat(outputs, ignore_index=True)

    for col in ["weighted_departures", "weighted_arrivals"]:
        result[col] = result[col].fillna(0.0)
    for col in [
        "n_trips_departures",
        "n_persons_departures",
        "n_trips_arrivals",
        "n_persons_arrivals",
    ]:
        result[col] = result[col].fillna(0).astype(int)

    result["n_trips"] = result[["n_trips_departures", "n_trips_arrivals"]].max(axis=1)
    result["n_persons"] = result[["n_persons_departures", "n_persons_arrivals"]].max(
        axis=1
    )
    result["lambda_departures_per_hour"] = result.apply(
        lambda row: row["weighted_departures"] / 365 / PERIOD_HOURS[row["period"]],
        axis=1,
    )
    result["lambda_arrivals_per_hour"] = result.apply(
        lambda row: row["weighted_arrivals"] / 365 / PERIOD_HOURS[row["period"]],
        axis=1,
    )
    result["low_unique_person_count"] = result["n_persons"] < 50

    return result[
        [
            "scenario",
            "pc4",
            "period",
            "weighted_departures",
            "weighted_arrivals",
            "lambda_departures_per_hour",
            "lambda_arrivals_per_hour",
            "n_trips",
            "n_persons",
            "low_unique_person_count",
        ]
    ]


def main() -> None:
    args = parse_args()
    db_config = db_config_from_env()

    yearly_outputs = []
    with psycopg.connect(**db_config) as conn:
        for year in args.years:
            raw = fetch_year(conn, year, args.limit)
            trips = prepare_trips(raw)
            output = aggregate_scenarios(trips)
            output.insert(0, "year", year)
            yearly_outputs.append(output)

    result = pd.concat(yearly_outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Wrote {len(result)} rows to {args.output}")
    flagged = int(result["low_unique_person_count"].sum())
    if flagged:
        print(f"Flagged {flagged} rows with fewer than 50 unique respondents.")


if __name__ == "__main__":
    main()
