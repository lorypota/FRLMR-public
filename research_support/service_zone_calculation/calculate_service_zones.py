"""Calculate empirical service zones for Den Haag docked-bike stations."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram

from research_support.empirical_analysis.internal.coverage_utils import (
    load_houses,
    wgs84_to_rd,
)
from research_support.empirical_analysis.internal.data_utils import DEN_HAAG_BBOX
from research_support.empirical_analysis.internal.paths import DATA_DIR, PROJECT_ROOT
from research_support.empirical_analysis.internal.processed_data_utils import (
    discover_docked_dates,
    discover_station_dates,
    latest_date,
    load_docked_day,
    load_station_day,
)

logger = logging.getLogger(__name__)

PROVIDERS = ("donkey_denHaag",)
RECENT_WINDOW_DAYS = 15
COVERAGE_RADIUS_M = 500
SERVICE_ZONE_COUNT = 20
SERVICE_CATEGORY_COUNT = 5
RANDOM_SEED = 7
SERVICE_ZONE_TAG = f"z{SERVICE_ZONE_COUNT}_cat{SERVICE_CATEGORY_COUNT}"

SERVICE_ZONE_DIR = PROJECT_ROOT / "research_support" / "service_zone_calculation"
OUTPUT_DIR = SERVICE_ZONE_DIR / "output"
FIGURES_DIR = SERVICE_ZONE_DIR / "figures"
BAG_ACTIVITY_CACHE = OUTPUT_DIR / "bag_non_residential_activity_den_haag.csv"
BAG_VERBLIJFSOBJECTEN_URL = (
    "https://api.pdok.nl/kadaster/bag/ogc/v2/collections/verblijfsobject/items"
)
BAG_REQUEST_LIMIT = 1000
BAG_ACTIVITY_FUNCTIONS = {
    "bijeenkomstfunctie",
    "gezondheidszorgfunctie",
    "kantoorfunctie",
    "logiesfunctie",
    "onderwijsfunctie",
    "sportfunctie",
    "winkelfunctie",
}
ODIN_DEMAND_PATH = (
    PROJECT_ROOT
    / "research_support"
    / "odin_demand_estimation"
    / "output"
    / "service_zone_period_demand_rates.csv"
)
ODIN_POOLED_YEAR = "pooled_2018_2023"
SERVICE_PRESSURE_WEIGHTS = {
    "departure_score": 1 / 3,
    "density_score": 1 / 3,
    "activity_score": 1 / 3,
}

BACKGROUND = "#f6f0e8"
PANEL_BACKGROUND = "#fbf8f2"
TEXT = "#23313b"
SUBTLE_TEXT = "#66727c"
GRID = "#d9d0c3"
BLUE = "#2f6690"
TEAL = "#2a9d8f"
GOLD = "#d8a34f"
RUST = "#b75d3a"
SLATE = "#8f98a3"

SERVICE_CATEGORY_COLORS = [RUST, GOLD, TEAL, BLUE, SLATE]


def ensure_output_dirs() -> None:
    for path in (
        SERVICE_ZONE_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_plot_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": SUBTLE_TEXT,
            "ytick.color": SUBTLE_TEXT,
            "text.color": TEXT,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _write_csv(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def _write_plot_geodata(gdf: gpd.GeoDataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    gdf.to_file(path, driver="GeoJSON")
    logger.info("Wrote %s", path)


def _filter_den_haag_stations(stations: pd.DataFrame) -> pd.DataFrame:
    return stations[
        stations["lat"].between(DEN_HAAG_BBOX["lat_min"], DEN_HAAG_BBOX["lat_max"])
        & stations["lon"].between(DEN_HAAG_BBOX["lon_min"], DEN_HAAG_BBOX["lon_max"])
    ].copy()


def _load_latest_station_snapshot() -> tuple[
    dict[str, tuple[int, int, int]], pd.DataFrame
]:
    latest_station_dates: dict[str, tuple[int, int, int]] = {}
    station_frames: list[pd.DataFrame] = []

    for provider in PROVIDERS:
        latest_station = latest_date(discover_station_dates(DATA_DIR, provider))
        if latest_station is None:
            raise FileNotFoundError(f"No station snapshots found for {provider}")

        stations = load_station_day(DATA_DIR, provider, *latest_station)
        if stations is None:
            raise FileNotFoundError(
                f"Failed to load station snapshot for {provider} on {latest_station}"
            )

        stations = _filter_den_haag_stations(stations)
        stations["provider"] = provider
        stations["station_id_raw"] = stations["station_id"].astype(str)
        stations["station_id_str"] = (
            stations["provider"] + "::" + stations["station_id_raw"]
        )
        latest_station_dates[provider] = latest_station
        station_frames.append(stations)

    return latest_station_dates, pd.concat(station_frames, ignore_index=True)


def _load_recent_common_docked_window(
    allowed_station_ids_by_provider: dict[str, set[str]],
    window_days: int = RECENT_WINDOW_DAYS,
) -> tuple[list[tuple[int, int, int]], pd.DataFrame]:
    provider_dates = [
        set(discover_docked_dates(DATA_DIR, provider)) for provider in PROVIDERS
    ]
    common_dates = sorted(set.intersection(*provider_dates))
    if len(common_dates) < window_days:
        raise ValueError(
            f"Need at least {window_days} common docked days, found {len(common_dates)}"
        )

    selected_dates = common_dates[-window_days:]
    provider_wide_frames: list[pd.DataFrame] = []

    for provider in PROVIDERS:
        frames: list[pd.DataFrame] = []
        common_cols: set[str] | None = None
        allowed_station_ids = allowed_station_ids_by_provider[provider]

        for year, month, day in selected_dates:
            day_df = load_docked_day(DATA_DIR, provider, year, month, day)
            if day_df is None:
                continue
            day_df = day_df.copy()
            day_df.columns = day_df.columns.astype(str)
            cols = set(day_df.columns) & allowed_station_ids
            common_cols = cols if common_cols is None else (common_cols & cols)
            frames.append(day_df)

        if not frames or common_cols is None:
            raise FileNotFoundError(
                f"No docked snapshots found for recent window of {provider}"
            )

        ordered_cols = sorted(common_cols)
        provider_wide = pd.concat(
            [frame[ordered_cols] for frame in frames]
        ).sort_index()
        provider_wide = provider_wide.rename(
            columns={col: f"{provider}::{col}" for col in provider_wide.columns}
        )
        provider_wide_frames.append(provider_wide)

    wide = pd.concat(provider_wide_frames, axis=1).sort_index()
    return selected_dates, wide


def _load_houses_rd() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    houses = load_houses()
    houses_rd = wgs84_to_rd(houses[:, :2])
    house_weights = houses[:, 2]
    return houses, houses_rd, house_weights


def _weighted_kmeans_labels(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    n_init: int = 20,
    seed: int = RANDOM_SEED,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-6)

    best_centroids: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_inertia = np.inf

    for attempt in range(n_init):
        rng = np.random.default_rng(seed + attempt)
        centroids = np.empty((k, points.shape[1]), dtype=float)

        first_idx = rng.choice(len(points), p=weights / weights.sum())
        centroids[0] = points[first_idx]
        min_dist_sq = np.sum((points - centroids[0]) ** 2, axis=1)

        for idx in range(1, k):
            probs = weights * np.maximum(min_dist_sq, 1e-12)
            probs = probs / probs.sum()
            chosen_idx = rng.choice(len(points), p=probs)
            centroids[idx] = points[chosen_idx]
            dist_sq = np.sum((points - centroids[idx]) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        labels = np.zeros(len(points), dtype=int)
        for _ in range(max_iter):
            dist_sq_matrix = np.sum(
                (points[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(dist_sq_matrix, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(k):
                member_mask = labels == idx
                if not np.any(member_mask):
                    refill_idx = rng.choice(len(points), p=weights / weights.sum())
                    centroids[idx] = points[refill_idx]
                    continue
                centroids[idx] = np.average(
                    points[member_mask],
                    axis=0,
                    weights=weights[member_mask],
                )

        final_dist_sq = np.sum((points - centroids[labels]) ** 2, axis=1)
        inertia = float(np.sum(weights * final_dist_sq))
        if inertia < best_inertia:
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_inertia = inertia

    if best_centroids is None or best_labels is None:
        raise RuntimeError(f"Weighted K-means failed for k={k}")

    return best_centroids, best_labels, best_inertia


def _minmax_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    if np.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _load_zone_demand_profile(demand_path: Path = ODIN_DEMAND_PATH) -> pd.DataFrame:
    if not demand_path.exists():
        raise FileNotFoundError(
            "Demand-aware service categories need ODiN service-zone demand rates. "
            f"Missing: {demand_path}"
        )

    demand = pd.read_csv(demand_path)
    required_cols = {
        "year",
        "service_zone",
        "lambda_departures_per_hour",
        "lambda_arrivals_per_hour",
    }
    missing_cols = sorted(required_cols - set(demand.columns))
    if missing_cols:
        raise ValueError(
            f"{demand_path} is missing required columns: {', '.join(missing_cols)}"
        )

    selected = demand[demand["year"].astype(str) == ODIN_POOLED_YEAR].copy()
    if selected.empty:
        raise ValueError(f"{demand_path} has no rows for year={ODIN_POOLED_YEAR}")

    selected["service_zone"] = selected["service_zone"].astype(int)
    return (
        selected.groupby("service_zone", as_index=False)
        .agg(
            zone_departures_per_hour=("lambda_departures_per_hour", "sum"),
            zone_arrivals_per_hour=("lambda_arrivals_per_hour", "sum"),
        )
        .sort_values("service_zone")
    )


def _activity_functions(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    return [
        value.strip()
        for value in str(raw_value).split(",")
        if value.strip() in BAG_ACTIVITY_FUNCTIONS
    ]


def _download_bag_activity_points() -> pd.DataFrame:
    bbox = ",".join(
        map(
            str,
            [
                DEN_HAAG_BBOX["lon_min"],
                DEN_HAAG_BBOX["lat_min"],
                DEN_HAAG_BBOX["lon_max"],
                DEN_HAAG_BBOX["lat_max"],
            ],
        )
    )
    query = urllib.parse.urlencode(
        {"f": "json", "limit": BAG_REQUEST_LIMIT, "bbox": bbox}
    )
    next_url: str | None = f"{BAG_VERBLIJFSOBJECTEN_URL}?{query}"
    rows: list[dict[str, object]] = []
    page_count = 0

    while next_url is not None:
        request = urllib.request.Request(
            next_url,
            headers={"User-Agent": "FRLSR service-zone calculation"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)

        page_count += 1
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            if properties.get("woonplaats_naam") != "'s-Gravenhage":
                continue
            if properties.get("status") != "Verblijfsobject in gebruik":
                continue
            functions = _activity_functions(properties.get("gebruiksdoel"))
            if not functions:
                continue

            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point":
                continue
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) < 2:
                continue

            rows.append(
                {
                    "identificatie": properties.get("identificatie"),
                    "lat": float(coordinates[1]),
                    "lon": float(coordinates[0]),
                    "gebruiksdoel": properties.get("gebruiksdoel"),
                    "activity_function_count": len(functions),
                    "oppervlakte_m2": pd.to_numeric(
                        properties.get("oppervlakte"), errors="coerce"
                    ),
                }
            )

        next_url = next(
            (
                link.get("href")
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )

    logger.info(
        "Downloaded %d BAG non-residential activity objects across %d page(s)",
        len(rows),
        page_count,
    )
    return pd.DataFrame(rows)


def _load_bag_activity_points(city_boundary) -> gpd.GeoDataFrame:
    if BAG_ACTIVITY_CACHE.exists():
        activity_df = pd.read_csv(BAG_ACTIVITY_CACHE)
        logger.info("Loaded BAG activity cache from %s", BAG_ACTIVITY_CACHE)
    else:
        activity_df = _download_bag_activity_points()
        activity_df.to_csv(BAG_ACTIVITY_CACHE, index=False)
        logger.info("Wrote %s", BAG_ACTIVITY_CACHE)

    if activity_df.empty:
        return gpd.GeoDataFrame(
            activity_df,
            geometry=[],
            crs="EPSG:28992",
        )

    activity_points = gpd.GeoDataFrame(
        activity_df,
        geometry=gpd.points_from_xy(activity_df["lon"], activity_df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:28992")
    return activity_points[activity_points.within(city_boundary)].copy()


def _summarize_bag_activity_by_zone(
    activity_points: gpd.GeoDataFrame,
    zone_polygon_gdf: gpd.GeoDataFrame,
    zone_density_df: pd.DataFrame,
) -> pd.DataFrame:
    zone_activity = zone_density_df[["service_zone", "zone_area_km2"]].copy()
    if activity_points.empty:
        zone_activity["bag_activity_count"] = 0
        zone_activity["bag_activity_function_count"] = 0
        zone_activity["bag_activity_area_m2"] = 0.0
    else:
        joined = gpd.sjoin(
            activity_points,
            zone_polygon_gdf[["service_zone", "geometry"]],
            how="left",
            predicate="within",
        )
        summary = (
            joined.dropna(subset=["service_zone"])
            .assign(service_zone=lambda df: df["service_zone"].astype(int))
            .groupby("service_zone", as_index=False)
            .agg(
                bag_activity_count=("identificatie", "count"),
                bag_activity_function_count=("activity_function_count", "sum"),
                bag_activity_area_m2=("oppervlakte_m2", "sum"),
            )
        )
        zone_activity = zone_activity.merge(
            summary,
            on="service_zone",
            how="left",
            validate="one_to_one",
        )
        fill_cols = [
            "bag_activity_count",
            "bag_activity_function_count",
            "bag_activity_area_m2",
        ]
        zone_activity[fill_cols] = zone_activity[fill_cols].fillna(0)

    zone_activity["bag_activity_density_per_km2"] = np.where(
        zone_activity["zone_area_km2"] > 0,
        zone_activity["bag_activity_count"] / zone_activity["zone_area_km2"],
        np.nan,
    )
    return zone_activity.drop(columns=["zone_area_km2"])


def _assign_service_pressure_categories(
    zone_profile_df: pd.DataFrame,
    category_count: int,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    scored = zone_profile_df.copy()
    scored["departure_score"] = _minmax_score(scored["zone_departures_per_hour"])
    scored["density_score"] = _minmax_score(scored["address_density_per_km2"])
    scored["activity_score"] = _minmax_score(scored["bag_activity_density_per_km2"])
    scored["service_pressure_score"] = sum(
        scored[column] * weight for column, weight in SERVICE_PRESSURE_WEIGHTS.items()
    )
    scored["density_rank"] = (
        scored["address_density_per_km2"].rank(method="first").astype(int) - 1
    )

    ordered_pressure = scored.sort_values("service_pressure_score").reset_index(
        drop=True
    )
    effective_category_count = min(category_count, len(ordered_pressure))
    ordered_pressure["service_pressure_rank"] = np.arange(
        len(ordered_pressure), dtype=int
    )
    ordered_pressure["service_category"] = (
        ordered_pressure["service_pressure_rank"]
        * effective_category_count
        // len(ordered_pressure)
    ).astype(int)
    ordered_pressure["zones_in_category"] = ordered_pressure.groupby(
        "service_category"
    )["service_zone"].transform("size")
    category_lookup = dict(
        zip(
            ordered_pressure["service_zone"],
            ordered_pressure["service_category"],
            strict=True,
        )
    )
    rank_lookup = dict(
        zip(
            ordered_pressure["service_zone"],
            ordered_pressure["service_pressure_rank"],
            strict=True,
        )
    )
    return ordered_pressure, category_lookup, rank_lookup


def _build_service_zone_polygons(
    centroids: np.ndarray,
    city_boundary,
) -> gpd.GeoDataFrame:
    centroid_points = [Point(xy) for xy in centroids]
    voronoi_cells = voronoi_diagram(
        MultiPoint(centroid_points),
        envelope=city_boundary.envelope,
        edges=False,
    )
    zone_polygon_rows = []
    for cell in voronoi_cells.geoms:
        clipped = cell.intersection(city_boundary)
        if clipped.is_empty:
            continue
        zone_id = None
        for idx, centroid_point in enumerate(centroid_points):
            if clipped.buffer(1e-6).covers(centroid_point):
                zone_id = idx
                break
        if zone_id is None:
            zone_id = int(np.argmin([clipped.distance(pt) for pt in centroid_points]))
        zone_polygon_rows.append({"service_zone": int(zone_id), "geometry": clipped})

    zones = (
        gpd.GeoDataFrame(zone_polygon_rows, geometry="geometry", crs="EPSG:28992")
        .sort_values("service_zone")
        .drop_duplicates(subset="service_zone")
    )
    return zones  # ty: ignore[invalid-return-type]


def _load_area_layer(name: str) -> gpd.GeoDataFrame:
    geodata_dir = (
        PROJECT_ROOT / "research_support" / "empirical_analysis" / "output" / "geodata"
    )
    if name == "pc4":
        paths = [geodata_dir / "pc4_den_haag.geojson"]
    elif name == "pc6":
        paths = sorted((geodata_dir / "pc6_den_haag").glob("*.geojson"))
    elif name == "buurten":
        paths = [geodata_dir / "buurten_den_haag.geojson"]
    elif name == "wijken":
        paths = [geodata_dir / "wijken_den_haag.geojson"]
    else:
        raise ValueError(f"Unsupported area layer: {name}")

    gdfs = [gpd.read_file(path) for path in paths]
    area = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(area, geometry="geometry", crs=gdfs[0].crs).to_crs(
        "EPSG:28992"
    )


def calculate_service_zones() -> None:
    ensure_output_dirs()

    _latest_station_dates, latest_stations = _load_latest_station_snapshot()
    allowed_station_ids_by_provider: dict[str, set[str]] = {
        str(provider): set(group["station_id_raw"])
        for provider, group in latest_stations.groupby("provider")
    }
    _recent_dates, recent_wide = _load_recent_common_docked_window(
        allowed_station_ids_by_provider
    )
    _houses, houses_rd, house_weights = _load_houses_rd()
    latest_stations = latest_stations[
        latest_stations["station_id_str"].isin(recent_wide.columns)
    ].copy()
    latest_stations = latest_stations.set_index("station_id_str").loc[
        recent_wide.columns
    ]
    station_coords = wgs84_to_rd(latest_stations[["lat", "lon"]].to_numpy())
    station_tree = KDTree(station_coords)
    _nearest_dist, nearest_idx = station_tree.query(houses_rd, k=1)
    candidate_counts = np.fromiter(
        (
            len(candidates)
            for candidates in station_tree.query_ball_point(
                houses_rd,
                r=COVERAGE_RADIUS_M,
            )
        ),
        dtype=np.int32,
    )
    reachable_mask = candidate_counts > 0

    assigned_addresses = np.bincount(
        nearest_idx[reachable_mask],
        weights=house_weights[reachable_mask],
        minlength=len(latest_stations),
    )
    service_zone_weights = np.maximum(assigned_addresses, 1.0)

    house_points = gpd.GeoDataFrame(
        {"weight": house_weights},
        geometry=gpd.points_from_xy(houses_rd[:, 0], houses_rd[:, 1]),
        crs="EPSG:28992",
    )

    city_boundary = _load_area_layer("buurten").geometry.union_all()
    zone_centroids, zone_labels, _inertia = _weighted_kmeans_labels(
        station_coords,
        service_zone_weights,
        SERVICE_ZONE_COUNT,
        seed=RANDOM_SEED,
    )
    zone_polygon_gdf = _build_service_zone_polygons(zone_centroids, city_boundary)

    service_zone_assignments = latest_stations.reset_index(drop=False).copy()
    if "index" in service_zone_assignments.columns:
        service_zone_assignments = service_zone_assignments.rename(
            columns={"index": "station_id_str"}
        )
    service_zone_assignments["service_zone"] = zone_labels
    zone_polygon_gdf["zone_center_x"] = [
        float(zone_centroids[idx, 0]) for idx in zone_polygon_gdf["service_zone"]
    ]
    zone_polygon_gdf["zone_center_y"] = [
        float(zone_centroids[idx, 1]) for idx in zone_polygon_gdf["service_zone"]
    ]

    house_zone_join = gpd.sjoin(
        house_points,
        zone_polygon_gdf[["service_zone", "geometry"]],
        how="left",
        predicate="within",
    )
    zone_density_rows = []
    for _, zone_row in zone_polygon_gdf.iterrows():
        zone_id = int(zone_row["service_zone"])
        zone_mask = house_zone_join["service_zone"] == zone_id
        total_zone_addresses = float(house_zone_join.loc[zone_mask, "weight"].sum())
        area_km2 = float(zone_row.geometry.area / 1_000_000)
        address_density = total_zone_addresses / area_km2 if area_km2 > 0 else np.nan
        zone_density_rows.append(
            {
                "service_zone": zone_id,
                "zone_area_km2": area_km2,
                "addresses_in_zone": total_zone_addresses,
                "address_density_per_km2": address_density,
            }
        )

    zone_density_df = pd.DataFrame(zone_density_rows).sort_values("service_zone")
    bag_activity_points = _load_bag_activity_points(city_boundary)
    zone_activity_df = _summarize_bag_activity_by_zone(
        bag_activity_points,
        zone_polygon_gdf,
        zone_density_df,
    )
    zone_density_df = zone_density_df.merge(
        zone_activity_df,
        on="service_zone",
        how="left",
        validate="one_to_one",
    )
    zone_demand_df = _load_zone_demand_profile()
    zone_density_df = zone_density_df.merge(
        zone_demand_df,
        on="service_zone",
        how="left",
        validate="one_to_one",
    )
    if zone_density_df["zone_departures_per_hour"].isna().any():
        missing_zones = sorted(
            zone_density_df.loc[
                zone_density_df["zone_departures_per_hour"].isna(), "service_zone"
            ].astype(int)
        )
        raise ValueError(
            "ODiN demand rates are missing generated service zones: "
            + ", ".join(map(str, missing_zones))
        )

    ordered_pressure, category_lookup, rank_lookup = (
        _assign_service_pressure_categories(
            zone_density_df,
            SERVICE_CATEGORY_COUNT,
        )
    )
    score_cols = [
        "service_zone",
        "departure_score",
        "density_score",
        "activity_score",
        "service_pressure_score",
        "density_rank",
    ]
    zone_density_df = zone_density_df.drop(columns=["density_rank"], errors="ignore")
    zone_density_df = zone_density_df.merge(
        ordered_pressure[score_cols],
        on="service_zone",
        how="left",
        validate="one_to_one",
    )
    zone_density_df["service_category"] = zone_density_df["service_zone"].map(
        category_lookup
    )
    zone_density_df["service_pressure_rank"] = zone_density_df["service_zone"].map(
        rank_lookup
    )
    zone_density_df["zones_in_category"] = zone_density_df["service_zone"].map(
        dict(
            zip(
                ordered_pressure["service_zone"],
                ordered_pressure["zones_in_category"],
                strict=True,
            )
        )
    )
    zone_density_df["category_label"] = zone_density_df["service_category"].apply(
        lambda category: f"Cat {int(category)}"
    )
    _write_csv(
        zone_density_df,
        f"service_zone_density_profile_{SERVICE_ZONE_TAG}.csv",
    )

    zone_polygon_gdf = zone_polygon_gdf.merge(
        zone_density_df[
            [
                "service_zone",
                "address_density_per_km2",
                "zone_departures_per_hour",
                "bag_activity_density_per_km2",
                "service_pressure_score",
                "service_category",
            ]
        ],
        on="service_zone",
        how="left",
    )
    service_zone_assignments["service_category"] = service_zone_assignments[
        "service_zone"
    ].map(category_lookup)
    _write_csv(
        service_zone_assignments[
            [
                "station_id_str",
                "provider",
                "station_id",
                "station_id_raw",
                "name",
                "lat",
                "lon",
                "capacity",
                "service_zone",
                "service_category",
            ]
        ],
        f"service_zone_assignments_{SERVICE_ZONE_TAG}.csv",
    )
    _write_plot_geodata(
        zone_polygon_gdf.to_crs("EPSG:4326"),
        f"service_zone_boundaries_{SERVICE_ZONE_TAG}.geojson",
    )


def plot_service_zone_map() -> None:
    ensure_output_dirs()
    configure_plot_theme()

    zone_assign_df = pd.read_csv(
        OUTPUT_DIR / f"service_zone_assignments_{SERVICE_ZONE_TAG}.csv"
    )
    zone_density_df = pd.read_csv(
        OUTPUT_DIR / f"service_zone_density_profile_{SERVICE_ZONE_TAG}.csv"
    )
    zone_boundary_gdf = gpd.read_file(
        OUTPUT_DIR / f"service_zone_boundaries_{SERVICE_ZONE_TAG}.geojson"
    )

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    unique_categories = sorted(
        zone_density_df["service_category"].dropna().astype(int).unique()
    )
    for category in unique_categories:
        color = SERVICE_CATEGORY_COLORS[category]
        boundary_mask = zone_boundary_gdf["service_category"] == category
        if boundary_mask.any():
            zone_boundary_gdf.loc[boundary_mask].plot(
                ax=ax,
                color=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=1.0,
            )
        zone_mask = zone_assign_df["service_category"] == category
        ax.scatter(
            zone_assign_df.loc[zone_mask, "lon"],
            zone_assign_df.loc[zone_mask, "lat"],
            s=24,
            color=color,
            label=f"Cat {category}",
            alpha=0.85,
        )
    ax.set_title(
        f"Chosen {SERVICE_ZONE_COUNT} service zones, mapped into {SERVICE_CATEGORY_COUNT} categories"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, alpha=0.4)
    _save_figure(fig, f"service_zone_map_{SERVICE_ZONE_TAG}.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    calculate_service_zones()
    plot_service_zone_map()


if __name__ == "__main__":
    main()
