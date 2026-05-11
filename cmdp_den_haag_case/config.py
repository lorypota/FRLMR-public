from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_scenario

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
PERIODS = ("morning", "evening")
YEAR_GROUP = "pooled_2018_2023"
R_MAX_VALUES = [0.05, 0.0625, 0.075, 0.0875, 0.10, 0.125, 0.15, 0.20, 1.0]
DEMAND_SCALES = [0.005, 0.01, 0.02]


def load_category_period_demand_rates() -> dict[int, dict[str, dict[str, Any]]]:
    if not DEMAND_RATES_PATH.exists():
        raise FileNotFoundError(f"Demand-rate CSV not found: {DEMAND_RATES_PATH}")

    with DEMAND_RATES_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Demand-rate CSV is empty: {DEMAND_RATES_PATH}")

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


def load_station_assignment_summary() -> dict[str, Any]:
    if not STATION_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Station-assignment CSV not found: {STATION_ASSIGNMENTS_PATH}"
        )

    with STATION_ASSIGNMENTS_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Station-assignment CSV is empty: {STATION_ASSIGNMENTS_PATH}")

    station_counts = {cat: 0 for cat in range(5)}
    capacity_sums = {cat: 0.0 for cat in range(5)}
    for row in rows:
        category = int(float(row["service_category"]))
        if category not in station_counts:
            raise ValueError(f"Unsupported service_category value: {category}")
        station_counts[category] += 1
        capacity_sums[category] += float(row.get("capacity") or 0.0)

    missing = [cat for cat, count in station_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Station-assignment CSV is missing stations for categories: "
            + ", ".join(str(cat) for cat in missing)
        )

    return {
        "station_counts_by_category": station_counts,
        "station_capacity_sums_by_category": capacity_sums,
    }


def build_den_haag_scenario(
    demand_scale: float = 1.0,
) -> dict[str, Any]:
    """Build the 5-category Den Haag CMDP scenario.

    ODiN rates are category-level potential movement demand. The station
    simulator consumes station-level Skellam parameters, so each category-period
    lambda is divided by the real Donkey station count in that category.
    """
    if demand_scale <= 0:
        raise ValueError("demand_scale must be positive")

    scenario = get_scenario(5)
    rates = load_category_period_demand_rates()
    station_summary = load_station_assignment_summary()
    station_counts = station_summary["station_counts_by_category"]
    reference_node_list = list(scenario["node_list"])

    scenario["node_list"] = [station_counts[cat] for cat in scenario["active_cats"]]
    scenario["boundaries"] = np.cumsum([0] + scenario["node_list"])

    demand_params = []
    raw_category_demand_params = []
    for cat_idx, cat in enumerate(scenario["active_cats"]):
        station_count = scenario["node_list"][cat_idx]
        cat_params = []
        raw_cat_params = []
        for period in PERIODS:
            row = rates[cat][period]
            raw_lambda_a = row["lambda_arrivals_per_hour"]
            raw_lambda_d = row["lambda_departures_per_hour"]
            lambda_a = raw_lambda_a / station_count * demand_scale
            lambda_d = raw_lambda_d / station_count * demand_scale
            cat_params.append((lambda_a, lambda_d))
            raw_cat_params.append((raw_lambda_a, raw_lambda_d))
        demand_params.append(cat_params)
        raw_category_demand_params.append(raw_cat_params)

    scenario["demand_params"] = demand_params
    scenario["raw_category_demand_params"] = raw_category_demand_params
    scenario["demand_rates"] = rates
    scenario["demand_year_group"] = YEAR_GROUP
    scenario["demand_rates_path"] = str(DEMAND_RATES_PATH)
    scenario["station_assignments_path"] = str(STATION_ASSIGNMENTS_PATH)
    scenario["station_counts_by_category"] = station_counts
    scenario["station_capacity_sums_by_category"] = station_summary[
        "station_capacity_sums_by_category"
    ]
    scenario["reference_node_list"] = reference_node_list
    scenario["reference_station_counts_by_category"] = dict(
        zip(scenario["active_cats"], reference_node_list, strict=True)
    )
    scenario["demand_scale"] = demand_scale
    scenario["boundaries"] = np.asarray(scenario["boundaries"])
    return scenario
