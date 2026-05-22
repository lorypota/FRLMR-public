from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_scenario
from common.network import generate_network

SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_RATES_PATH = (
    SCRIPT_DIR.parent
    / "research_support"
    / "odin_demand_estimation"
    / "output"
    / "category_period_demand_rates.csv"
)
STATION_ASSIGNMENTS_PATH = (
    SCRIPT_DIR.parent
    / "research_support"
    / "service_zone_calculation"
    / "output"
    / "service_zone_assignments_k20.csv"
)
DOCKED_DATA_DIR = (
    SCRIPT_DIR.parent
    / "research_support"
    / "empirical_analysis"
    / "output"
    / "data"
    / "docked"
    / "donkey_denHaag"
)
PERIODS = ("morning", "evening")
YEAR_GROUP = "pooled_2018_2023"
R_MAX_VALUES = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.10, 1.0]
DEMAND_SCALES = [0.01]


def _read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{label} CSV is empty: {path}")
    return rows


def load_category_period_demand_rates() -> dict[int, dict[str, dict[str, Any]]]:
    rows = _read_csv_rows(DEMAND_RATES_PATH, "Demand-rate")
    rates: dict[int, dict[str, dict[str, Any]]] = {cat: {} for cat in range(5)}
    for row in rows:
        if row["year"] != YEAR_GROUP:
            continue
        category = int(float(row["service_category"]))
        if category not in rates:
            raise ValueError(f"Unsupported service_category value: {category}")
        period = row["period"].strip().lower()
        if period not in PERIODS:
            continue
        rates[category][period] = {
            "lambda_departures_per_hour": float(row["lambda_departures_per_hour"]),
            "lambda_arrivals_per_hour": float(row["lambda_arrivals_per_hour"]),
            "weighted_departures": float(row["weighted_departures"]),
            "weighted_arrivals": float(row["weighted_arrivals"]),
            "n_trips": int(row["n_trips"]),
            "n_persons": int(row["n_persons"]),
            "low_unique_person_count": row["low_unique_person_count"].lower() == "true",
        }

    missing = [
        f"cat{cat}/{period}"
        for cat in range(5)
        for period in PERIODS
        if period not in rates[cat]
    ]
    if missing:
        raise ValueError(
            "Demand-rate CSV is missing required category-period rows: "
            + ", ".join(missing)
        )
    return rates


def _load_latest_zone_initial_bikes(
    station_to_zone: dict[str, int],
) -> tuple[dict[int, int], str, str]:
    docked_files = sorted(DOCKED_DATA_DIR.glob("docked_*.csv"))
    if not docked_files:
        raise FileNotFoundError(f"No docked-bike CSV files found in {DOCKED_DATA_DIR}")

    docked_path = docked_files[-1]
    rows = _read_csv_rows(docked_path, "Docked-bike")
    latest_row = rows[-1]
    zone_bikes = {zone: 0 for zone in set(station_to_zone.values())}
    for station_id, zone in station_to_zone.items():
        value = latest_row.get(station_id)
        if value not in (None, ""):
            zone_bikes[zone] += int(float(value))
    return zone_bikes, str(docked_path), latest_row["timestamp"]


def load_zone_inventory_units() -> list[dict[str, Any]]:
    rows = _read_csv_rows(STATION_ASSIGNMENTS_PATH, "Station-assignment")
    zones: dict[int, dict[str, Any]] = {}
    station_to_zone = {}
    for row in rows:
        zone = int(float(row["service_zone"]))
        category = int(float(row["service_category"]))
        if category not in range(5):
            raise ValueError(f"Unsupported service_category value: {category}")
        zone_info = zones.setdefault(
            zone,
            {
                "service_zone": zone,
                "service_category": category,
                "station_count": 0,
                "capacity": 0,
            },
        )
        if zone_info["service_category"] != category:
            raise ValueError(f"Service zone {zone} maps to multiple categories")
        zone_info["station_count"] += 1
        zone_info["capacity"] += int(float(row.get("capacity") or 0.0))
        station_to_zone[row["station_id_raw"]] = zone

    missing = [
        cat
        for cat in range(5)
        if not any(zone["service_category"] == cat for zone in zones.values())
    ]
    if missing:
        raise ValueError(
            "Station-assignment CSV is missing zones for categories: "
            + ", ".join(str(cat) for cat in missing)
        )

    zone_initial_bikes, initial_bikes_path, initial_bikes_timestamp = (
        _load_latest_zone_initial_bikes(station_to_zone)
    )
    zone_records = []
    for zone in zones.values():
        initial_bikes = zone_initial_bikes.get(zone["service_zone"], 0)
        zone_records.append(
            {
                **zone,
                "initial_bikes": min(initial_bikes, zone["capacity"]),
                "raw_initial_bikes": initial_bikes,
                "initial_bikes_path": initial_bikes_path,
                "initial_bikes_timestamp": initial_bikes_timestamp,
            }
        )
    return sorted(
        zone_records,
        key=lambda zone: (zone["service_category"], zone["service_zone"]),
    )


def build_den_haag_scenario(demand_scale: float = 1.0) -> dict[str, Any]:
    """Build the 5-category Den Haag CMDP scenario.

    Each generated node represents one service zone with aggregate
    station capacity. ODiN demand remains category-period demand and is split
    equally over the service zones in each category.
    """
    if demand_scale <= 0:
        raise ValueError("demand_scale must be positive")

    scenario = get_scenario(5)
    category_rates = load_category_period_demand_rates()
    zone_records = load_zone_inventory_units()
    reference_node_list = list(scenario["node_list"])

    scenario["node_list"] = [
        sum(1 for zone in zone_records if zone["service_category"] == cat)
        for cat in scenario["active_cats"]
    ]
    scenario["boundaries"] = np.cumsum([0] + scenario["node_list"])

    zone_demand_params = []
    demand_params = []
    raw_category_demand_params = []
    for cat_idx, cat in enumerate(scenario["active_cats"]):
        cat_zones = [zone for zone in zone_records if zone["service_category"] == cat]
        zone_count = scenario["node_list"][cat_idx]
        cat_params = []
        raw_cat_params = []
        for period in PERIODS:
            row = category_rates[cat][period]
            raw_lambda_a = row["lambda_arrivals_per_hour"]
            raw_lambda_d = row["lambda_departures_per_hour"]
            lambda_a = raw_lambda_a / zone_count * demand_scale
            lambda_d = raw_lambda_d / zone_count * demand_scale
            cat_params.append((lambda_a, lambda_d))
            raw_cat_params.append((raw_lambda_a, raw_lambda_d))
        demand_params.append(cat_params)
        raw_category_demand_params.append(raw_cat_params)
        zone_demand_params.extend([cat_params for _zone in cat_zones])

    scenario["demand_params"] = demand_params
    scenario["raw_category_demand_params"] = raw_category_demand_params
    scenario["zone_demand_params"] = zone_demand_params
    scenario["demand_rates"] = category_rates
    scenario["demand_year_group"] = YEAR_GROUP
    scenario["demand_rates_path"] = str(DEMAND_RATES_PATH)
    scenario["station_assignments_path"] = str(STATION_ASSIGNMENTS_PATH)
    scenario["model_unit"] = "service_zone"
    scenario["demand_allocation"] = "category_period_equal_split_over_service_zones"
    scenario["zone_records"] = zone_records
    scenario["zone_capacities"] = [zone["capacity"] for zone in zone_records]
    scenario["zone_initial_bikes"] = [zone["initial_bikes"] for zone in zone_records]
    scenario["zone_raw_initial_bikes"] = [
        zone["raw_initial_bikes"] for zone in zone_records
    ]
    scenario["initial_bikes_path"] = zone_records[0]["initial_bikes_path"]
    scenario["initial_bikes_timestamp"] = zone_records[0]["initial_bikes_timestamp"]
    scenario["station_counts_by_category"] = {
        cat: sum(
            zone["station_count"]
            for zone in zone_records
            if zone["service_category"] == cat
        )
        for cat in scenario["active_cats"]
    }
    scenario["station_capacity_sums_by_category"] = {
        cat: sum(
            zone["capacity"] for zone in zone_records if zone["service_category"] == cat
        )
        for cat in scenario["active_cats"]
    }
    scenario["reference_node_list"] = reference_node_list
    scenario["reference_station_counts_by_category"] = dict(
        zip(scenario["active_cats"], reference_node_list, strict=True)
    )
    scenario["demand_scale"] = demand_scale
    scenario["boundaries"] = np.asarray(scenario["boundaries"])
    return scenario


def build_den_haag_network(scenario: dict[str, Any]):
    graph = generate_network(scenario["node_list"])
    for node, zone in enumerate(scenario["zone_records"]):
        graph.nodes[node]["station"] = zone["service_category"]
        graph.nodes[node]["service_zone"] = zone["service_zone"]
        graph.nodes[node]["capacity"] = zone["capacity"]
        graph.nodes[node]["initial_bikes"] = zone["initial_bikes"]
        graph.nodes[node]["raw_initial_bikes"] = zone["raw_initial_bikes"]
        graph.nodes[node]["station_count"] = zone["station_count"]
        graph.nodes[node]["bikes"] = zone["initial_bikes"]
    return graph
