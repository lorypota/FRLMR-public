"""Justify the service-zone category count using demand-profile spread.

This script does two things in one place:
1. Cluster the 20 service zones for candidate K values.
2. Show the detailed arrival, departure, and net-flow spread for one selected K.

Lower within-category spread means one shared Q-table is a better approximation
for the zones in that category.

Usage:
    uv run research_support/service_zone_calculation/justify_category_count.py
    uv run research_support/service_zone_calculation/justify_category_count.py --categories 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DEMAND_CSV = (
    SCRIPT_DIR.parent
    / "ODiN_demand_estimation"
    / "output"
    / "service_zone_period_demand_rates.csv"
)
YEAR_GROUP = "pooled_2018_2023"
RANDOM_SEED = 7
CLUSTER_FEATURES = [
    "morning_net_share",
    "evening_net_share",
    "morning_arrivals",
    "morning_departures",
    "evening_arrivals",
    "evening_departures",
]


def load_zone_demand() -> pd.DataFrame:
    demand = pd.read_csv(DEMAND_CSV)
    demand = demand[demand["year"] == YEAR_GROUP]
    frame = demand.pivot(
        index="service_zone",
        columns="period",
        values=["lambda_arrivals_per_hour", "lambda_departures_per_hour"],
    )
    frame.columns = [
        f"{period}_{'arrivals' if value.endswith('arrivals_per_hour') else 'departures'}"
        for value, period in frame.columns
    ]
    frame = frame.reset_index()
    for period in ("morning", "evening"):
        departures = frame[f"{period}_departures"]
        frame[f"{period}_net_pct"] = (
            (frame[f"{period}_arrivals"] - departures) / departures * 100
        ).where(departures > 0, 0.0)
        frame[f"{period}_net_share"] = frame[f"{period}_net_pct"] / 100
    return frame


def demand_cluster_labels(features: pd.DataFrame, categories: int) -> np.ndarray:
    x = StandardScaler().fit_transform(features[CLUSTER_FEATURES].to_numpy(float))
    labels = KMeans(
        n_clusters=categories,
        n_init=50,
        random_state=RANDOM_SEED,
    ).fit_predict(x)
    order = features.groupby(labels)["morning_net_share"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[label] for label in labels], dtype=int)


def summarize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby("service_category").agg(
        n_zones=("service_zone", "size"),
        zones=("service_zone", lambda s: ",".join(str(int(v)) for v in sorted(s))),
        morning_net_mean_pct=("morning_net_pct", "mean"),
        morning_net_min_pct=("morning_net_pct", "min"),
        morning_net_max_pct=("morning_net_pct", "max"),
        evening_net_mean_pct=("evening_net_pct", "mean"),
        evening_net_min_pct=("evening_net_pct", "min"),
        evening_net_max_pct=("evening_net_pct", "max"),
        morning_arrivals_min=("morning_arrivals", "min"),
        morning_arrivals_max=("morning_arrivals", "max"),
        morning_departures_min=("morning_departures", "min"),
        morning_departures_max=("morning_departures", "max"),
        evening_arrivals_min=("evening_arrivals", "min"),
        evening_arrivals_max=("evening_arrivals", "max"),
        evening_departures_min=("evening_departures", "min"),
        evening_departures_max=("evening_departures", "max"),
    )
    for period in ("morning", "evening"):
        for direction in ("arrivals", "departures"):
            summary[f"{period}_{direction}_spread"] = (
                summary[f"{period}_{direction}_max"]
                - summary[f"{period}_{direction}_min"]
            )
    summary["morning_net_spread_pp"] = (
        summary["morning_net_max_pct"] - summary["morning_net_min_pct"]
    )
    summary["evening_net_spread_pp"] = (
        summary["evening_net_max_pct"] - summary["evening_net_min_pct"]
    )
    return summary.reset_index()


def compare_category_counts(
    features: pd.DataFrame, min_k: int, max_k: int
) -> pd.DataFrame:
    rows = []
    for categories in range(min_k, max_k + 1):
        labels = demand_cluster_labels(features, categories)
        frame = features.assign(service_category=labels)
        summary = summarize_categories(frame)
        rows.append(
            {
                "K": categories,
                "smallest_category_size": int(summary["n_zones"].min()),
                "singleton_categories": int((summary["n_zones"] == 1).sum()),
                "max_arrival_diff": round(
                    float(
                        max(
                            summary["morning_arrivals_spread"].max(),
                            summary["evening_arrivals_spread"].max(),
                        )
                    ),
                    1,
                ),
                "max_departure_diff": round(
                    float(
                        max(
                            summary["morning_departures_spread"].max(),
                            summary["evening_departures_spread"].max(),
                        )
                    ),
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_split(features: pd.DataFrame, categories: int) -> pd.DataFrame:
    labels = demand_cluster_labels(features, categories)
    return (
        features.assign(service_category=labels)
        .sort_values(["service_category", "service_zone"])
        .reset_index(drop=True)
    )


def format_category_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": summary["service_category"],
            "n_zones": summary["n_zones"],
            "zones": summary["zones"],
            "morning_net_mean_pct": summary["morning_net_mean_pct"],
            "morning_net_range_pct": summary.apply(
                lambda row: (
                    f"{row['morning_net_min_pct']:.1f} "
                    f"to {row['morning_net_max_pct']:.1f}"
                ),
                axis=1,
            ),
            "evening_net_mean_pct": summary["evening_net_mean_pct"],
            "evening_net_range_pct": summary.apply(
                lambda row: (
                    f"{row['evening_net_min_pct']:.1f} "
                    f"to {row['evening_net_max_pct']:.1f}"
                ),
                axis=1,
            ),
            "morning_arrivals_range": summary.apply(
                lambda row: (
                    f"{row['morning_arrivals_min']:.1f} "
                    f"to {row['morning_arrivals_max']:.1f}"
                ),
                axis=1,
            ),
            "morning_departures_range": summary.apply(
                lambda row: (
                    f"{row['morning_departures_min']:.1f} "
                    f"to {row['morning_departures_max']:.1f}"
                ),
                axis=1,
            ),
            "evening_arrivals_range": summary.apply(
                lambda row: (
                    f"{row['evening_arrivals_min']:.1f} "
                    f"to {row['evening_arrivals_max']:.1f}"
                ),
                axis=1,
            ),
            "evening_departures_range": summary.apply(
                lambda row: (
                    f"{row['evening_departures_min']:.1f} "
                    f"to {row['evening_departures_max']:.1f}"
                ),
                axis=1,
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--categories",
        type=int,
        default=5,
        help="Candidate K to describe after the K comparison.",
    )
    args = parser.parse_args()

    features = load_zone_demand()

    comparison = compare_category_counts(features, args.min_k, args.max_k)

    print(
        "smallest_category_size = fewest zones in any category. "
        "This shows how much replication the shared Q-table gets."
    )
    print(
        "max_arrival_diff and max_departure_diff = largest difference between "
        "the highest and lowest hourly rate inside one category."
    )
    print(comparison.to_string(index=False))

    tag = f"candidate_k{args.categories}"
    split = candidate_split(features, args.categories)

    summary = summarize_categories(split)
    display = format_category_summary(summary)
    print(f"\n{tag} ({YEAR_GROUP}):")
    print(display.round(1).to_string(index=False))
    print("\nArrival and departure ranges are hourly rates. Lower is more similar.")


if __name__ == "__main__":
    main()
