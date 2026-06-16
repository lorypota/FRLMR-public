"""
Shared configuration.

Contains all scenario definitions (demand parameters, network topology,
station reward parameters) and helper functions used across training,
evaluation, and analysis scripts.
"""

from typing import Any, TypedDict

import numpy as np


class ScenarioDef(TypedDict):
    """Static scenario definition (before expansion in get_scenario)."""

    demand_params: list[list[tuple[float, float]]]
    node_list: list[int]
    active_cats: list[int]
    station_params: dict[int, tuple[float, float, float, float, float, float]]


# Resource limits app-reken12
MAX_MEMORY_MB = 24576
CPU_CORES = "0-19"

# Fairness penalty (chi) and rebalancing cost (phi) per station category
CHI = {0: 1, 1: 0.5, 2: 0.4, 3: -0.5, 4: -1}
PHI = {0: 1, 1: 0.8, 2: 0.4, 3: 0.3, 4: 0.1}

# =============================================================================
# SCENARIO CONFIGURATIONS
#
# Each scenario defines:
#   demand_params:   list of (λ_a, λ_d) tuples per time slot, per category
#   node_list:       number of stations per category (peripheral -> central)
#   active_cats:     which of the 5 category indices are used
#   station_params:  per-category reward params as compact tuples:
#                    (chi, phi, evening_target, evening_threshold,
#                               morning_target, morning_threshold)
# =============================================================================

SCENARIOS: dict[int, ScenarioDef] = {
    2: {
        "demand_params": [
            [(0.3, 2), (1.5, 0.3)],  # cat 0 (remote)
            [(13.8, 3.6), (6.6, 13.8)],  # cat 4 (central)
        ],
        "node_list": [60, 10],
        "active_cats": [0, 4],
        "station_params": {
            0: (1, 1, 22, 0.4, 2, 8),
            4: (-1, 0.1, 0, 61, 88, 1),
        },
    },
    3: {
        "demand_params": [
            [(0.3, 2), (1.5, 0.3)],  # cat 0
            [(3.3, 1.5), (1.5, 3.3)],  # cat 2
            [(13.8, 9), (12, 13.8)],  # cat 4
        ],
        "node_list": [60, 30, 10],
        "active_cats": [0, 2, 4],
        "station_params": {
            0: (1, 1, 22, 0.4, 2, 8),
            2: (0.4, 0.4, 3, 12, 25, 2),
            4: (-1, 0.1, 5, 30, 36, 7),
        },
    },
    4: {
        "demand_params": [
            [(0.3, 2), (1.5, 0.3)],  # cat 0
            [(0.45, 3), (2.25, 0.45)],  # cat 1
            [(9.2, 2.4), (4.4, 9.2)],  # cat 3
            [(13.8, 7), (9, 13.8)],  # cat 4
        ],
        "node_list": [60, 40, 20, 10],
        "active_cats": [0, 1, 3, 4],
        "station_params": {
            0: (1, 1, 22, 0.4, 2, 8),
            1: (0.5, 0.8, 32, 0.3, 1, 11),
            3: (-0.5, 0.3, 0.3, 41, 60, 1),
            4: (-1, 0.1, 2, 42, 64, 3),
        },
    },
    5: {
        "demand_params": [
            [(0.3, 2), (1.5, 0.3)],  # cat 0
            [(0.45, 3), (2.25, 0.45)],  # cat 1
            [(3.3, 1.5), (1.5, 3.3)],  # cat 2
            [(9.2, 5.1), (6.6, 9.2)],  # cat 3
            [(13.8, 7), (10, 13.8)],  # cat 4
        ],
        "node_list": [60, 40, 30, 20, 10],
        "active_cats": [0, 1, 2, 3, 4],
        "station_params": {
            0: (1, 1, 22, 0.4, 2, 8),
            1: (0.5, 0.8, 32, 0.3, 1, 11),
            2: (0.4, 0.4, 3, 12, 25, 2),
            3: (-0.5, 0.3, 3, 26, 39, 3),
            4: (-1, 0.1, 2, 42, 54, 3),
        },
    },
}

# Training duration per category (number of repeats)
TRAIN_UNTIL = {0: 50, 1: 50, 2: 37, 3: 55, 4: 110}

# Default hyperparameters
GAMMA = 20  # Rebalancing cost coefficient
NUM_TRAIN_DAYS = 1000
NUM_EVAL_DAYS = 101  # Days to run evaluation (first is skipped)
TIME_SLOTS = [(0, 12), (12, 24)]


def build_station_params(raw):
    """Convert compact tuples to the dict format expected by FairEnv.

    Args:
        raw: Dict mapping category -> (chi, phi, eve_target, eve_thresh, morn_target, morn_thresh)

    Returns:
        Dict mapping category -> dict with named keys.
    """
    return {
        cat: {
            "chi": chi,
            "phi": phi,
            "evening_target": eve_t,
            "evening_threshold": eve_th,
            "morning_target": morn_t,
            "morning_threshold": morn_th,
        }
        for cat, (chi, phi, eve_t, eve_th, morn_t, morn_th) in raw.items()
    }


def get_scenario(num_categories):
    """Get a scenario config with station_params already expanded.

    Args:
        num_categories: Number of area categories (2, 3, 4, or 5).

    Returns:
        Dict with keys: demand_params, node_list, active_cats, station_params (expanded).

    Raises:
        ValueError: If num_categories not in [2, 3, 4, 5].
    """
    if num_categories not in SCENARIOS:
        raise ValueError(
            f"Invalid number of categories: {num_categories} (choose from 2, 3, 4 or 5)."
        )
    scenario: dict[str, Any] = dict(SCENARIOS[num_categories])
    scenario["station_params"] = build_station_params(scenario["station_params"])
    scenario["boundaries"] = np.cumsum([0] + scenario["node_list"])
    return scenario
