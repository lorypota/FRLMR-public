"""Shared coverage computation utilities for statistical analysis.

Provides functions for loading house/buurt data, computing nearest-bike
distances, and aggregating coverage metrics by area.

Usage:
    Import-only helper module used by analysis_*.py scripts.
"""

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .paths import GEODATA_DIR

logger = logging.getLogger(__name__)

PROVIDER = "donkey_denHaag"

CBS_MISSING_VALUES = {-99995, -99997}

DISTANCE_BANDS = [(0, 100), (100, 250), (250, 500), (500, float("inf"))]
DISTANCE_BAND_LABELS = ["0-100m", "100-250m", "250-500m", "500m+"]

_transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False)


def wgs84_to_rd(lat_lon: np.ndarray) -> np.ndarray:
    """Convert (N, 2) array of [lat, lon] WGS84 to EPSG:28992 [x, y] meters."""
    x, y = _transformer_to_rd.transform(lat_lon[:, 0], lat_lon[:, 1])
    return np.column_stack([x, y])


def load_houses(geodata_dir=GEODATA_DIR):
    """Load houses from houses_den_haag.json.

    Returns (N, 3) array: [lat, lon, address_count] in WGS84.
    """
    path = geodata_dir / "houses_den_haag.json"
    with open(path) as f:
        data = json.load(f)

    rows = []
    for cell_values in data["cells"].values():
        for i in range(0, len(cell_values), 3):
            lat = cell_values[i] / 1_000_000
            lon = cell_values[i + 1] / 1_000_000
            count = cell_values[i + 2]
            rows.append((lat, lon, count))

    houses = np.array(rows, dtype=np.float64)
    logger.info(
        "Loaded %d house locations (%d total addresses)",
        len(houses),
        int(houses[:, 2].sum()),
    )
    return houses


def load_buurten(geodata_dir=GEODATA_DIR):
    """Load buurten GeoJSON, replace CBS missing values with NaN, reproject to RD.

    Returns GeoDataFrame in EPSG:28992 with a reset integer index.
    """
    path = geodata_dir / "buurten_den_haag.geojson"
    gdf = gpd.read_file(path)

    numeric_cols = gdf.select_dtypes(include="number").columns
    for col in numeric_cols:
        gdf[col] = gdf[col].replace(CBS_MISSING_VALUES, np.nan)

    gdf = gdf.to_crs("EPSG:28992")
    gdf = gdf.reset_index(drop=True)
    return gdf


def assign_houses_to_buurten(houses_rd, buurten_gdf):
    """Assign each house to a buurt via spatial join.

    Args:
        houses_rd: (N, 2) array of house coordinates in EPSG:28992.
        buurten_gdf: GeoDataFrame of buurten in EPSG:28992.

    Returns:
        (N,) integer array of buurt indices. -1 if outside all buurten.
    """
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(houses_rd[:, 0], houses_rd[:, 1]),
        crs="EPSG:28992",
    )
    joined = gpd.sjoin(points, buurten_gdf, how="left", predicate="within")

    result = np.full(len(houses_rd), -1, dtype=np.int64)
    mask = joined["index_right"].notna()
    result[joined.index[mask]] = joined["index_right"][mask].astype(int).values
    return result


def get_docked_bike_positions(docked_row, stations_df):
    """Expand docked station counts into (M, 2) array of [lat, lon].

    Args:
        docked_row: Series with station_id columns and bike count values.
            Can be None if no docked data.
        stations_df: DataFrame with station_id, lat, lon columns.
            Can be None if no station data.

    Returns:
        (M, 2) ndarray of bike positions in WGS84 [lat, lon].
    """
    if docked_row is None or stations_df is None:
        return np.empty((0, 2))

    parts = []

    station_lookup = stations_df.set_index("station_id")[["lat", "lon"]]
    for sid, count in docked_row.items():
        if pd.isna(count):
            continue
        count = int(count)
        if count <= 0:
            continue
        # Station IDs may be int in stations_df but str in docked columns
        for key in (sid, str(sid), int(sid) if str(sid).isdigit() else None):
            if key is not None and key in station_lookup.index:
                row = station_lookup.loc[key]
                parts.append(np.tile([row["lat"], row["lon"]], (count, 1)))
                break

    if not parts:
        return np.empty((0, 2))
    return np.vstack(parts)


def compute_nearest_distances(houses_rd, bikes_rd):
    """Compute nearest-bike distance for each house.

    Args:
        houses_rd: (N, 2) array of house positions in EPSG:28992.
        bikes_rd: (M, 2) array of bike positions in EPSG:28992.

    Returns:
        (N,) array of distances in meters. inf if no bikes.
    """
    if len(bikes_rd) == 0:
        return np.full(len(houses_rd), np.inf)

    tree = cKDTree(bikes_rd)
    distances, _ = tree.query(houses_rd, k=1)
    return distances


def compute_coverage_metrics(distances, weights):
    """Compute address-weighted coverage metrics from per-house distances.

    Args:
        distances: (N,) array of distances in meters.
        weights: (N,) array of address counts per house.

    Returns:
        dict with: mean_distance, median_distance, pct_within_100m,
        pct_within_250m, pct_within_500m, band_proportions.
    """
    total_weight = weights.sum()
    if total_weight == 0:
        return {
            "mean_distance": np.nan,
            "median_distance": np.nan,
            "pct_within_100m": 0.0,
            "pct_within_250m": 0.0,
            "pct_within_500m": 0.0,
            "band_proportions": {label: 0.0 for label in DISTANCE_BAND_LABELS},
        }

    mean_dist = np.average(distances, weights=weights)

    sorted_idx = np.argsort(distances)
    sorted_d = distances[sorted_idx]
    sorted_w = weights[sorted_idx]
    cum_w = np.cumsum(sorted_w)
    median_idx = np.searchsorted(cum_w, total_weight / 2)
    median_dist = sorted_d[min(median_idx, len(sorted_d) - 1)]

    pct_100 = weights[distances <= 100].sum() / total_weight * 100
    pct_250 = weights[distances <= 250].sum() / total_weight * 100
    pct_500 = weights[distances <= 500].sum() / total_weight * 100

    band_props = {}
    for (lo, hi), label in zip(DISTANCE_BANDS, DISTANCE_BAND_LABELS, strict=True):
        mask = (distances > lo) & (distances <= hi) if lo > 0 else (distances <= hi)
        band_props[label] = weights[mask].sum() / total_weight * 100

    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "pct_within_100m": pct_100,
        "pct_within_250m": pct_250,
        "pct_within_500m": pct_500,
        "band_proportions": band_props,
    }


def aggregate_by_buurt(distances, weights, buurt_indices, n_buurten):
    """Aggregate per-house distances to per-buurt coverage metrics.

    Returns DataFrame indexed by buurt index with coverage columns.
    """
    records = []
    for bi in range(n_buurten):
        mask = buurt_indices == bi
        if not mask.any():
            records.append(
                {
                    "buurt_idx": bi,
                    "n_houses": 0,
                    "total_addresses": 0,
                    "mean_distance": np.nan,
                    "pct_within_100m": np.nan,
                    "pct_within_250m": np.nan,
                    "pct_within_500m": np.nan,
                }
            )
            continue

        d = distances[mask]
        w = weights[mask]
        metrics = compute_coverage_metrics(d, w)
        records.append(
            {
                "buurt_idx": bi,
                "n_houses": int(mask.sum()),
                "total_addresses": int(w.sum()),
                "mean_distance": metrics["mean_distance"],
                "pct_within_100m": metrics["pct_within_100m"],
                "pct_within_250m": metrics["pct_within_250m"],
                "pct_within_500m": metrics["pct_within_500m"],
            }
        )

    return pd.DataFrame(records).set_index("buurt_idx")


def subsample_hourly(df, timestamp_col="timestamp"):
    """Pick one row per hour, closest to the :00 mark.

    For dockless data (long format with multiple bikes per timestamp),
    returns all rows for the selected timestamps.
    """
    ts = pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df["_hour"] = ts.dt.hour
    df["_minute"] = ts.dt.minute

    unique_ts = df.drop_duplicates(subset=[timestamp_col])
    unique_ts = unique_ts.copy()
    unique_ts["_hour"] = pd.to_datetime(unique_ts[timestamp_col]).dt.hour
    unique_ts["_minute"] = pd.to_datetime(unique_ts[timestamp_col]).dt.minute
    best = unique_ts.loc[unique_ts.groupby("_hour")["_minute"].idxmin()]
    selected_timestamps = set(best[timestamp_col])

    result = df[df[timestamp_col].isin(selected_timestamps)].copy()
    result.drop(columns=["_hour", "_minute"], inplace=True)
    return result


def subsample_hourly_wide(df):
    """Pick one row per hour for wide-format DataFrames (docked).

    Index must be a DatetimeIndex.
    """
    hours = df.index.hour
    minutes = df.index.minute

    best_indices = []
    for h in range(24):
        mask = hours == h
        if not mask.any():
            continue
        candidates = df.index[mask]
        mins = minutes[mask]
        best_indices.append(candidates[mins.argmin()])

    return df.loc[best_indices]
