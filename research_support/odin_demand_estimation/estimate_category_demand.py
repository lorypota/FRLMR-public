"""Estimate ODiN demand rates for the Den Haag CMDP.

ODiN locations are represented through PC4 origin and destination codes. This
script keeps PC4 as the intermediate geography, maps PC4s to the generated
service zones, and writes demand tables that can feed the Skellam setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - gives a clear CLI error
    raise SystemExit("Missing PostgreSQL dependency.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "output"
PC4_GEOJSON = (
    REPO_ROOT
    / "preliminary_studies"
    / "empirical_analysis"
    / "output"
    / "geodata"
    / "pc4_den_haag.geojson"
)
SERVICE_ZONES_GEOJSON = (
    REPO_ROOT
    / "preliminary_studies"
    / "service_zone_calculation"
    / "output"
    / "service_zone_boundaries_k20.geojson"
)

SUPPORTED_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
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

PERIOD_HOURS = {
    "morning": 12,
    "evening": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ODiN demand rates for Den Haag."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(SUPPORTED_YEARS),
        help="ODiN years to query. Supported: 2018 through 2023.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per year for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--pc4-geojson",
        type=Path,
        default=PC4_GEOJSON,
        help="PC4 polygon GeoJSON used for service-zone mapping.",
    )
    parser.add_argument(
        "--service-zones-geojson",
        type=Path,
        default=SERVICE_ZONES_GEOJSON,
        help="Service-zone polygon GeoJSON.",
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


def get_table_columns(conn: psycopg.Connection, year: int) -> dict[str, str]:
    """Return lowercase column lookup to actual database column name."""
    query = sql.SQL("SELECT * FROM {}.{} LIMIT 0").format(
        sql.Identifier("odin"),
        sql.Identifier(f"odin{year}"),
    )
    with conn.cursor() as cur:
        try:
            cur.execute(query)
        except psycopg.errors.UndefinedTable as exc:
            raise SystemExit(
                f"Table odin.odin{year} does not exist in this database. "
            ) from exc
        return {column.name.lower(): column.name for column in cur.description}


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
            resolved[logical_name] = available[match]

    if missing:
        joined = "; ".join(missing)
        available_cols = ", ".join(sorted(available.values()))
        raise SystemExit(
            f"odin.odin{year} is missing required columns: {joined}\n"
            f"Available columns: {available_cols}"
        )
    return resolved


def fetch_year(conn: psycopg.Connection, year: int, limit: int | None) -> pd.DataFrame:
    if year not in SUPPORTED_YEARS:
        supported = ", ".join(str(y) for y in SUPPORTED_YEARS)
        raise SystemExit(f"Unsupported year {year}. Supported years: {supported}")

    columns = resolve_columns(conn, year)
    query = sql.SQL(
        """
        SELECT
            {opid} AS opid,
            {verplid} AS verplid,
            {verpl} AS verpl,
            {vertpc} AS vertpc,
            {aankpc} AS aankpc,
            {vertgem} AS vertgem,
            {aankgem} AS aankgem,
            {afstv} AS afstv,
            {khvm} AS khvm,
            {vertuur} AS vertuur,
            {factorv} AS factorv
        FROM {schema}.{table}
        WHERE {verpl} = 1
          AND {afstv} BETWEEN 5 AND 100
          AND ({vertgem} = 518 OR {aankgem} = 518)
        """
    ).format(
        schema=sql.Identifier("odin"),
        table=sql.Identifier(f"odin{year}"),
        opid=sql.Identifier(columns["opid"]),
        verplid=sql.Identifier(columns["verplid"]),
        verpl=sql.Identifier(columns["verpl"]),
        vertpc=sql.Identifier(columns["vertpc"]),
        aankpc=sql.Identifier(columns["aankpc"]),
        vertgem=sql.Identifier(columns["vertgem"]),
        aankgem=sql.Identifier(columns["aankgem"]),
        afstv=sql.Identifier(columns["afstv"]),
        khvm=sql.Identifier(columns["khvm"]),
        vertuur=sql.Identifier(columns["vertuur"]),
        factorv=sql.Identifier(columns["factorv"]),
    )
    params = None
    if limit is not None:
        query += sql.SQL(" LIMIT {limit}").format(limit=sql.Placeholder("limit"))
        params = {"limit": limit}
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

    df["year"] = str(year)
    df["source_year"] = year
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
    trips["respondent_id"] = (
        trips["source_year"].astype(str) + ":" + trips["opid"].astype(str)
    )
    trips["trip_id"] = (
        trips["source_year"].astype(str) + ":" + trips["verplid"].astype(str)
    )
    return trips.dropna(subset=["period", "factorv"])


def build_pooled_trips(trips: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    if trips.empty or len(years) <= 1:
        return pd.DataFrame(columns=trips.columns)

    pooled = trips.copy()
    pooled["year"] = f"pooled_{min(years)}_{max(years)}"
    pooled["factorv"] = pooled["factorv"] / len(years)
    return pooled


def load_pc4_service_zone_mapping(
    pc4_geojson: Path,
    service_zones_geojson: Path,
) -> pd.DataFrame:
    pc4 = gpd.read_file(pc4_geojson)[["postcode", "geometry"]].copy()
    zones = gpd.read_file(service_zones_geojson)[
        ["service_zone", "service_category", "geometry"]
    ].copy()

    pc4["pc4"] = pc4["postcode"].map(normalize_pc4)
    pc4 = pc4.dropna(subset=["pc4"])
    pc4 = pc4.to_crs(28992)
    zones = zones.to_crs(28992)

    intersections = gpd.overlay(pc4[["pc4", "geometry"]], zones, how="intersection")
    intersections["overlap_area_m2"] = intersections.geometry.area
    dominant = intersections.sort_values(
        "overlap_area_m2", ascending=False
    ).drop_duplicates("pc4")
    return pd.DataFrame(
        dominant[["pc4", "service_zone", "service_category", "overlap_area_m2"]]
    )


def attach_spatial_mapping(trips: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped = trips.merge(
        mapping.add_prefix("origin_"),
        left_on="origin_pc4",
        right_on="origin_pc4",
        how="left",
    )
    mapped = mapped.merge(
        mapping.add_prefix("destination_"),
        left_on="destination_pc4",
        right_on="destination_pc4",
        how="left",
    )
    return mapped


def add_lambdas(
    df: pd.DataFrame,
    departure_col: str = "weighted_departures",
    arrival_col: str = "weighted_arrivals",
) -> pd.DataFrame:
    result = df.copy()
    result["lambda_departures_per_hour"] = result.apply(
        lambda row: row[departure_col] / 365 / PERIOD_HOURS[row["period"]],
        axis=1,
    )
    result["lambda_arrivals_per_hour"] = result.apply(
        lambda row: row[arrival_col] / 365 / PERIOD_HOURS[row["period"]],
        axis=1,
    )
    return result


def aggregate_direction(
    trips: pd.DataFrame,
    group_cols: list[str],
    value_name: str,
) -> pd.DataFrame:
    valid = trips.dropna(subset=group_cols)
    if valid.empty:
        return pd.DataFrame(columns=[*group_cols, value_name, "n_trips", "n_persons"])

    return valid.groupby(group_cols, as_index=False).agg(
        **{
            value_name: ("factorv", "sum"),
            f"{value_name}_trips": ("trip_id", "nunique"),
            f"{value_name}_persons": ("respondent_id", "nunique"),
        }
    )


def aggregate_period_rates(
    trips: pd.DataFrame,
    location_col: str,
    output_location_col: str,
) -> pd.DataFrame:
    departures = aggregate_direction(
        trips,
        ["year", location_col, "period"],
        "weighted_departures",
    ).rename(columns={location_col: output_location_col})
    arrivals = aggregate_direction(
        trips,
        ["year", location_col.replace("origin", "destination"), "period"],
        "weighted_arrivals",
    ).rename(
        columns={location_col.replace("origin", "destination"): output_location_col}
    )

    merged = departures.merge(
        arrivals,
        on=["year", output_location_col, "period"],
        how="outer",
    )
    for col in ["weighted_departures", "weighted_arrivals"]:
        merged[col] = merged[col].fillna(0.0)
    for col in [
        "weighted_departures_trips",
        "weighted_departures_persons",
        "weighted_arrivals_trips",
        "weighted_arrivals_persons",
    ]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["n_trips"] = merged[
        ["weighted_departures_trips", "weighted_arrivals_trips"]
    ].max(axis=1)
    merged["n_persons"] = merged[
        ["weighted_departures_persons", "weighted_arrivals_persons"]
    ].max(axis=1)
    merged["low_unique_person_count"] = merged["n_persons"] < 50
    merged = add_lambdas(merged)

    return merged[
        [
            "year",
            output_location_col,
            "period",
            "weighted_departures",
            "weighted_arrivals",
            "lambda_departures_per_hour",
            "lambda_arrivals_per_hour",
            "n_trips",
            "n_persons",
            "low_unique_person_count",
        ]
    ].sort_values(["year", output_location_col, "period"])


def aggregate_od(
    trips: pd.DataFrame,
    group_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_cols = extra_cols or []
    valid = trips.dropna(subset=group_cols)
    if valid.empty:
        return pd.DataFrame()

    result = valid.groupby(["year", *group_cols, "period"], as_index=False).agg(
        weighted_trips=("factorv", "sum"),
        n_trips=("trip_id", "nunique"),
        n_persons=("respondent_id", "nunique"),
    )
    result["lambda_trips_per_hour"] = result.apply(
        lambda row: row["weighted_trips"] / 365 / PERIOD_HOURS[row["period"]],
        axis=1,
    )
    result["low_unique_person_count"] = result["n_persons"] < 50
    return result[
        [
            "year",
            *group_cols,
            "period",
            "weighted_trips",
            "lambda_trips_per_hour",
            "n_trips",
            "n_persons",
            "low_unique_person_count",
            *extra_cols,
        ]
    ].sort_values(["year", *group_cols, "period"])


def write_outputs(trips: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    outputs = {
        "pc4_period_demand_rates.csv": aggregate_period_rates(
            trips,
            "origin_pc4",
            "pc4",
        ),
        "service_zone_period_demand_rates.csv": aggregate_period_rates(
            trips,
            "origin_service_zone",
            "service_zone",
        ),
        "category_period_demand_rates.csv": aggregate_period_rates(
            trips,
            "origin_service_category",
            "service_category",
        ),
        "pc4_od_demand_rates.csv": aggregate_od(
            trips,
            ["origin_pc4", "destination_pc4"],
        ),
        "service_zone_od_demand_rates.csv": aggregate_od(
            trips.dropna(subset=["origin_service_zone", "destination_service_zone"]),
            [
                "origin_service_zone",
                "destination_service_zone",
                "origin_service_category",
                "destination_service_category",
            ],
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)
    return outputs


def print_summary(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    print(f"Wrote outputs to {output_dir}")
    for filename, df in outputs.items():
        flagged = int(df["low_unique_person_count"].sum()) if not df.empty else 0
        print(f"{filename}: {len(df)} rows, {flagged} low-count rows")


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years))
    db_config = db_config_from_env()

    yearly_trips = []
    with psycopg.connect(**db_config) as conn:
        for year in years:
            raw = fetch_year(conn, year, args.limit)
            yearly_trips.append(prepare_trips(raw))

    trips = pd.concat(yearly_trips, ignore_index=True)
    pooled = build_pooled_trips(trips, years)
    if not pooled.empty:
        trips = pd.concat([trips, pooled], ignore_index=True)

    mapping = load_pc4_service_zone_mapping(
        args.pc4_geojson, args.service_zones_geojson
    )
    trips = attach_spatial_mapping(trips, mapping)

    outputs = write_outputs(trips, args.output_dir)
    print_summary(outputs, args.output_dir)


if __name__ == "__main__":
    main()
