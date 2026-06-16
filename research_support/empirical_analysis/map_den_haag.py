"""
Interactive Den Haag area map with time slider

Usage:
    uv run research_support/empirical_analysis/map_den_haag.py

Creates an interactive Folium map showing Den Haag area polygons and bike
availability over time with multiple visualization modes. The map supports PC4,
PC6, CBS buurten, and CBS wijken.

Output:
    output/maps/den_haag.html
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from textwrap import dedent
from typing import cast

import folium
import geopandas as gpd
import pandas as pd
from branca.element import Figure
from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from internal.area_visualizations import (
    DEFAULT_VISUALIZATION_MODE,
    build_visualization_js,
    build_visualization_options_html,
)
from internal.area_visualizations.hotspot import build_hourly_hotspot_data
from internal.artifact_index import rebuild_artifact_index
from internal.data_utils import (
    DEN_HAAG_BBOX,
    DEN_HAAG_CENTER,
    PROVIDER,
    filter_by_bbox,
)
from internal.paths import (
    DATA_DIR,
    GEODATA_DIR,
    MAPS_DIR,
    PROJECT_ROOT,
    ensure_output_dirs,
)
from internal.processed_data_utils import (
    load_docked_day,
    load_station_day,
)

HOUSE_POINTS_CACHE = str(GEODATA_DIR / "houses_den_haag.json")
HOUSE_POINTS_CACHE_VERSION = 2

BAG_WFS_URL = "https://service.pdok.nl/kadaster/bag/wfs/v2_0"
BAG_PAGE_SIZE = 1000
HOUSE_CELL_SIZE_DEGREES = 0.0025
HOUSE_DOWNLOAD_TILE_COUNT = 6
HOUSE_BBOX_BUFFER = 0.005  # ~500m buffer to catch edge houses
PDOK_API_PAGE_SIZE = 500
PDOK_JAARCODE = 2024
SERVICE_ZONE_BOUNDARIES_PATH = (
    PROJECT_ROOT
    / "research_support"
    / "service_zone_calculation"
    / "output"
    / "service_zone_boundaries_z20_cat5.geojson"
)

AREA_LEVEL_CONFIG = {
    "pc4": {
        "label": "PC4",
        "endpoint_url": "https://api.pdok.nl/cbs/postcode4/ogc/v1/collections/postcode4/items",
        "property": "postcode",
        "cache_path": str(GEODATA_DIR / "pc4_den_haag.geojson"),
        "query_params": {"jaarcode": PDOK_JAARCODE},
    },
    "pc6": {
        "label": "PC6",
        "endpoint_url": "https://api.pdok.nl/cbs/postcode6/ogc/v1/collections/postcode6/items",
        "property": "postcode6",
        "cache_path": str(GEODATA_DIR / "pc6_den_haag"),
        "query_params": {"jaarcode": PDOK_JAARCODE},
    },
    "buurt": {
        "label": "CBS buurten",
        "endpoint_url": (
            "https://api.pdok.nl/cbs/wijken-en-buurten-2024/"
            "ogc/v1/collections/buurten/items"
        ),
        "property": "buurtcode",
        "cache_path": str(GEODATA_DIR / "buurten_den_haag.geojson"),
    },
    "wijk": {
        "label": "CBS wijken",
        "endpoint_url": (
            "https://api.pdok.nl/cbs/wijken-en-buurten-2024/"
            "ogc/v1/collections/wijken/items"
        ),
        "property": "wijkcode",
        "cache_path": str(GEODATA_DIR / "wijken_den_haag.geojson"),
    },
    "service_zone": {
        "label": "Calculated CMDP service zones",
        "property": "service_zone",
        "source_path": str(SERVICE_ZONE_BOUNDARIES_PATH),
    },
}
AREA_LEVELS = tuple(AREA_LEVEL_CONFIG)
DEFAULT_AREA_LEVEL = "pc4"
POSTCODE_LEVEL_CONFIG = AREA_LEVEL_CONFIG
POSTCODE_LEVELS = AREA_LEVELS
DEFAULT_POSTCODE_LEVEL = DEFAULT_AREA_LEVEL

LIGHT_MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
DARK_MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
MAP_TILE_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)
DEFAULT_ZOOM = 13
ALL_PROVIDERS_VALUE = "__all__"
MAP_PROVIDER_INFO = {
    "donkey_denHaag": {"label": "Donkey Republic"},
    "ns_ov_fiets": {"label": "NS OV-fiets"},
}


def _build_bag_bbox_28992(bbox: dict[str, float]) -> str:
    """Return a BAG WFS bbox string in the default service CRS."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    min_x, min_y = transformer.transform(bbox["lon_min"], bbox["lat_min"])
    max_x, max_y = transformer.transform(bbox["lon_max"], bbox["lat_max"])
    return f"{min_x:.3f},{min_y:.3f},{max_x:.3f},{max_y:.3f},EPSG:28992"


def download_house_points(bbox: dict[str, float], cache_path: str) -> dict:
    """Download and cache Den Haag residential BAG points in a compact grid.

    Each coordinate stores [lat_i, lon_i, address_count] triples so we know
    how many addresses (apartments) share the same building location.
    """
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("version") == HOUSE_POINTS_CACHE_VERSION:
            print(f"  Loading cached house points from {cache_path}")
            return cached
        print("  House-point cache version changed, rebuilding...")

    print("  Downloading BAG residential points from PDOK...")
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    # Count how many addresses share each coordinate (apartments in same building)
    coord_counts: dict[tuple[int, int], int] = {}
    fetched_count = 0

    # Expand bbox slightly to catch edge houses
    dl_bbox = {
        "lat_min": bbox["lat_min"] - HOUSE_BBOX_BUFFER,
        "lat_max": bbox["lat_max"] + HOUSE_BBOX_BUFFER,
        "lon_min": bbox["lon_min"] - HOUSE_BBOX_BUFFER,
        "lon_max": bbox["lon_max"] + HOUSE_BBOX_BUFFER,
    }

    tile_total = HOUSE_DOWNLOAD_TILE_COUNT * HOUSE_DOWNLOAD_TILE_COUNT
    pages_fetched = 0
    lat_step = (dl_bbox["lat_max"] - dl_bbox["lat_min"]) / HOUSE_DOWNLOAD_TILE_COUNT
    lon_step = (dl_bbox["lon_max"] - dl_bbox["lon_min"]) / HOUSE_DOWNLOAD_TILE_COUNT

    for tile_y in range(HOUSE_DOWNLOAD_TILE_COUNT):
        for tile_x in range(HOUSE_DOWNLOAD_TILE_COUNT):
            tile_bbox = {
                "lat_min": dl_bbox["lat_min"] + (tile_y * lat_step),
                "lat_max": dl_bbox["lat_min"] + ((tile_y + 1) * lat_step),
                "lon_min": dl_bbox["lon_min"] + (tile_x * lon_step),
                "lon_max": dl_bbox["lon_min"] + ((tile_x + 1) * lon_step),
            }
            tile_bbox_28992 = _build_bag_bbox_28992(tile_bbox)
            page_index = 0

            while True:
                params = urllib.parse.urlencode(
                    {
                        "service": "WFS",
                        "version": "2.0.0",
                        "request": "GetFeature",
                        "typeNames": "bag:verblijfsobject",
                        "count": BAG_PAGE_SIZE,
                        "startIndex": page_index * BAG_PAGE_SIZE,
                        "bbox": tile_bbox_28992,
                        "outputFormat": "application/json; subtype=geojson",
                    }
                )
                url = f"{BAG_WFS_URL}?{params}"
                with urllib.request.urlopen(url, timeout=60) as resp:
                    payload = json.load(resp)

                features = payload.get("features", [])
                if not features:
                    break

                fetched_count += len(features)
                pages_fetched += 1
                for feature in features:
                    props = feature.get("properties") or {}
                    if "woonfunctie" not in str(props.get("gebruiksdoel", "")).lower():
                        continue
                    if "in gebruik" not in str(props.get("status", "")).lower():
                        continue

                    geometry = feature.get("geometry") or {}
                    coords = geometry.get("coordinates") or []
                    if geometry.get("type") != "Point" or len(coords) < 2:
                        continue

                    lon, lat = transformer.transform(coords[0], coords[1])
                    if not (
                        dl_bbox["lat_min"] <= lat <= dl_bbox["lat_max"]
                        and dl_bbox["lon_min"] <= lon <= dl_bbox["lon_max"]
                    ):
                        continue

                    lat_i = int(round(lat * 1_000_000))
                    lon_i = int(round(lon * 1_000_000))
                    coord_key = (lat_i, lon_i)
                    coord_counts[coord_key] = coord_counts.get(coord_key, 0) + 1

                page_index += 1
                if pages_fetched == 1 or pages_fetched % 25 == 0:
                    tile_number = (tile_y * HOUSE_DOWNLOAD_TILE_COUNT) + tile_x + 1
                    print(
                        f"    Tiles done: {tile_number}/{tile_total} | "
                        f"pages fetched: {pages_fetched} | "
                        f"raw points: {fetched_count} | "
                        f"unique locations: {len(coord_counts)}"
                    )
                if len(features) < BAG_PAGE_SIZE:
                    break

    # Build cells: each cell stores [lat_i, lon_i, count, lat_i, lon_i, count, ...]
    cells: dict[str, list[int]] = {}
    for (lat_i, lon_i), count in coord_counts.items():
        lon = lon_i / 1_000_000
        lat = lat_i / 1_000_000
        cell_x = int(lon // HOUSE_CELL_SIZE_DEGREES)
        cell_y = int(lat // HOUSE_CELL_SIZE_DEGREES)
        cell_key = f"{cell_x}:{cell_y}"
        if cell_key not in cells:
            cells[cell_key] = []
        cells[cell_key].extend((lat_i, lon_i, count))

    kept_count = len(coord_counts)
    total_addresses = sum(coord_counts.values())
    payload = {
        "version": HOUSE_POINTS_CACHE_VERSION,
        "cellSize": HOUSE_CELL_SIZE_DEGREES,
        "count": kept_count,
        "totalAddresses": total_addresses,
        "cells": cells,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp_path, cache_path)
    print(
        f"  Cached {kept_count} unique locations "
        f"({total_addresses} total addresses) to {cache_path}"
    )
    return payload


def _build_area_api_url(
    endpoint_url: str,
    bbox: dict[str, float],
    extra_params: dict[str, int | str] | None = None,
) -> str:
    """Build the first PDOK area-items URL for one boundary source."""
    params = {
        "f": "json",
        "limit": PDOK_API_PAGE_SIZE,
        "bbox": (
            f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"
        ),
    }
    if extra_params:
        params.update(extra_params)
    return f"{endpoint_url}?{urllib.parse.urlencode(params)}"


def _iter_geojson_cache_parts(cache_path: str) -> list[Path]:
    """Return one or more GeoJSON cache files for a cache target."""
    cache = Path(cache_path)
    if cache.is_file():
        return [cache]
    if cache.is_dir():
        return sorted(cache.glob("*.geojson"))
    return []


def _load_geojson_cache(cache_path: str, area_label: str) -> gpd.GeoDataFrame | None:
    """Load cached GeoJSON from a single file or a directory of split parts."""
    parts = _iter_geojson_cache_parts(cache_path)
    if not parts:
        return None

    if len(parts) == 1:
        print(f"  Loading cached {area_label} polygons from {parts[0]}")
        return gpd.read_file(parts[0])

    print(
        f"  Loading cached {area_label} polygons from {len(parts)} files in {cache_path}"
    )
    frames = [gpd.read_file(part) for part in parts]
    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    return merged


def _write_geojson_cache(gdf: gpd.GeoDataFrame, cache_path: str) -> None:
    """Write GeoJSON cache as one file or split parts, based on cache_path."""
    cache = Path(cache_path)
    if cache.suffix.lower() == ".geojson":
        cache.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache, driver="GeoJSON")
        print(f"  Cached to {cache}")
        return

    cache.mkdir(parents=True, exist_ok=True)
    for old_part in cache.glob("*.geojson"):
        old_part.unlink()

    n_rows = len(gdf)
    part_count = 2
    chunk_size = max(1, (n_rows + part_count - 1) // part_count)
    for part_index, start in enumerate(range(0, n_rows, chunk_size), start=1):
        stop = min(start + chunk_size, n_rows)
        part_path = cache / f"part_{part_index:02d}.geojson"
        gdf.iloc[start:stop].to_file(part_path, driver="GeoJSON")

    print(f"  Cached to {cache} ({len(list(cache.glob('*.geojson')))} parts)")


def _build_runtime_geojson_resource(cache_path: str) -> str | list[str]:
    """Return one runtime path or a list of split-part paths from maps/ to geodata/."""
    parts = _iter_geojson_cache_parts(cache_path)
    if not parts:
        rel_path = os.path.relpath(cache_path, MAPS_DIR).replace("\\", "/")
        return rel_path
    rel_paths = [os.path.relpath(part, MAPS_DIR).replace("\\", "/") for part in parts]
    if len(rel_paths) == 1:
        return rel_paths[0]
    return rel_paths


def _geojson_resource_path(cfg: dict) -> str:
    """Return the GeoJSON path used by the runtime map."""
    if cfg.get("source_path"):
        return cfg["source_path"]
    return cfg["cache_path"]


def _load_local_geojson_source(
    *, source_path: str, area_label: str
) -> gpd.GeoDataFrame:
    """Load a generated local GeoJSON file without copying it into geodata."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(
            f"{area_label} local source not found: {source}. "
            "Run calculate_service_zones.py first."
        )

    print(f"  Loading generated {area_label} polygons from {source}")
    return gpd.read_file(source)


def load_area_polygons(*, bbox: dict[str, float], cfg: dict) -> gpd.GeoDataFrame:
    """Load one area subdivision, either from PDOK or from a generated local file."""
    if cfg.get("source_path"):
        return _load_local_geojson_source(
            source_path=cfg["source_path"],
            area_label=cfg["label"],
        )

    return download_postcode_polygons(
        bbox=bbox,
        cache_path=cfg["cache_path"],
        endpoint_url=cfg["endpoint_url"],
        area_property=cfg["property"],
        area_label=cfg["label"],
        query_params=cfg.get("query_params"),
    )


def download_postcode_polygons(
    *,
    bbox: dict[str, float],
    cache_path: str,
    endpoint_url: str,
    area_property: str,
    area_label: str,
    query_params: dict[str, int | str] | None = None,
) -> gpd.GeoDataFrame:
    """Download postcode polygons from PDOK CBS API, with local caching."""
    cached_gdf = _load_geojson_cache(cache_path, area_label)
    if cached_gdf is not None:
        return cached_gdf

    print(f"  Downloading {area_label} polygons from PDOK...")
    features = []
    page_count = 0
    next_url = _build_area_api_url(endpoint_url, bbox, query_params)
    while next_url:
        with urllib.request.urlopen(next_url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        page_features = payload.get("features", [])
        if not page_features:
            break
        features.extend(page_features)
        page_count += 1
        if page_count == 1 or page_count % 5 == 0:
            print(
                f"    Pages fetched: {page_count} | "
                f"{area_label} areas so far: {len(features)}"
            )

        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if area_property not in gdf.columns:
        raise KeyError(f"{area_label} response missing '{area_property}' column")
    print(f"  Downloaded {len(gdf)} {area_label} areas")

    _write_geojson_cache(gdf, cache_path)
    return gdf


def process_date(
    data_dir: str,
    year: int,
    month: int,
    day: int,
    pc4_gdf: gpd.GeoDataFrame,
) -> dict | None:
    """Process one date's processed tables."""
    date_str = f"{year}-{month:02d}-{day:02d}"
    stations_df = load_station_day(data_dir, PROVIDER, year, month, day)
    if stations_df is None or stations_df.empty:
        print(f"  {date_str}: station metadata not found, skipping")
        return None

    required_station_cols = {"station_id", "name", "lat", "lon", "capacity"}
    if not required_station_cols.issubset(stations_df.columns):
        print(f"  {date_str}: station metadata columns missing, skipping")
        return None

    all_stations = stations_df[list(required_station_cols)].to_dict("records")
    dh_stations = filter_by_bbox(all_stations, DEN_HAAG_BBOX)
    if not dh_stations:
        print(f"  {date_str}: no stations in bbox, skipping")
        return None
    dh_station_ids = {str(s["station_id"]) for s in dh_stations}
    print(f"    Stations: {len(dh_stations)}")

    station_gdf = gpd.GeoDataFrame(
        dh_stations,
        geometry=[Point(s["lon"], s["lat"]) for s in dh_stations],
        crs="EPSG:4326",
    )
    station_joined = gpd.sjoin(
        station_gdf,
        pc4_gdf[["postcode", "geometry"]],
        how="left",
        predicate="within",
    )
    station_to_pc4 = {}
    for _, row in station_joined.iterrows():
        if pd.notna(row["postcode"]):
            station_to_pc4[row["station_id"]] = int(row["postcode"])

    print("    Loading docked-bike data...")
    df_avail = load_docked_day(data_dir, PROVIDER, year, month, day)
    if df_avail is None or df_avail.empty:
        print(f"  {date_str}: docked table not found, skipping")
        return None
    df_avail.columns = df_avail.columns.map(str)
    df_hourly = df_avail.resample("h").first()
    hours = sorted({int(ts.hour) for ts in df_hourly.index})
    if not hours:
        print(f"  {date_str}: no hourly docked data, skipping")
        return None
    max_hour = max(hours)
    print(f"\n  Processing {date_str} (hours {hours[0]}-{max_hour})...")

    hour_to_row = {ts.hour: df_hourly.loc[ts] for ts in df_hourly.index}

    counts = {}
    for pc4 in pc4_gdf["postcode"]:
        pc4_key = str(int(pc4))
        counts[pc4_key] = {
            "c": [0] * (max_hour + 1),
            "s": [0] * (max_hour + 1),
            "f": [0] * (max_hour + 1),
        }

    for hour in hours:
        row = hour_to_row.get(hour)
        if row is None:
            continue
        for station_id in dh_station_ids:
            if station_id not in df_hourly.columns:
                continue
            pc4 = station_to_pc4.get(station_id)
            if not pc4:
                continue
            pc4_key = str(pc4)
            if pc4_key not in counts:
                continue
            raw_value = row.get(station_id, 0)
            value = int(raw_value) if pd.notna(raw_value) else 0
            counts[pc4_key]["s"][hour] += value
            counts[pc4_key]["c"][hour] += value

    bikes_by_hour: dict[str, list[list[float]]] = {str(hour): [] for hour in hours}

    stations_js = []
    for station in dh_stations:
        station_id = station["station_id"]
        pc4 = station_to_pc4.get(station_id, 0)
        hourly_avail = [0] * (max_hour + 1)
        for hour in hours:
            row = hour_to_row.get(hour)
            if row is not None and station_id in df_hourly.columns:
                raw_value = row.get(station_id, 0)
                hourly_avail[hour] = int(raw_value) if pd.notna(raw_value) else 0
        stations_js.append(
            {
                "ll": [round(station["lat"], 6), round(station["lon"], 6)],
                "n": station["name"],
                "cap": int(station["capacity"]) if pd.notna(station["capacity"]) else 0,
                "pc": pc4,
                "av": hourly_avail,
            }
        )

    hotspot = build_hourly_hotspot_data(stations_js, hours, DEN_HAAG_BBOX)

    return {
        "hours": hours,
        "maxHour": max_hour,
        "counts": counts,
        "bikes": bikes_by_hour,
        "stations": stations_js,
        "hotspot": hotspot,
    }


def build_panel_controls_html(
    panel_id: str,
    visualization_options: str,
) -> str:
    """Render the floating controls card for one map panel."""
    return f"""
        <div class="panel-controls" id="{panel_id}-controls-box">
            <div class="drag-handle panel-drag-handle">
                <span class="panel-heading-label" id="{panel_id}-panel-heading-label">Map controls</span>
            </div>
            <div class="controls-row">
                <div class="control-inline slider-row">
                    <div class="control-header">
                        <label for="{panel_id}-date-input">Date</label>
                        <span class="date-readout" id="{panel_id}-date-label"></span>
                    </div>
                    <input type="date" id="{panel_id}-date-input">
                </div>
            </div>
            <div class="controls-row">
                <div class="control-inline slider-row">
                    <div class="control-header">
                        <label for="{panel_id}-hour-slider">Time</label>
                        <span class="hour-readout" id="{panel_id}-hour-label">00:00</span>
                    </div>
                    <input type="range" id="{panel_id}-hour-slider" min="0" max="23" value="0">
                </div>
            </div>
            <div class="controls-row">
                <div class="control-inline">
                    <label for="{panel_id}-visualization-mode">Visualization</label>
                    <select id="{panel_id}-visualization-mode">
{visualization_options}
                    </select>
                </div>
            </div>
        </div>
    """


def build_page_styles(left_map_id: str) -> str:
    """Return page-level styles for the compare-mode layout."""
    return dedent(
        f"""
        <style>
        html, body {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: sans-serif;
            background: #dfe6ea;
            color: #24313f;
        }}

        #map-shell {{
            position: fixed;
            inset: 0;
            padding: 12px;
            box-sizing: border-box;
        }}

        #map-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 12px;
            height: 100%;
        }}

        body.compare-on #map-grid {{
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        }}

        .map-panel {{
            position: relative;
            min-width: 0;
            min-height: 0;
            border-radius: 14px;
            overflow: hidden;
            background: #eef2f3;
            box-shadow: 0 10px 28px rgba(22, 35, 46, 0.14);
        }}

        #panel-right {{
            display: none;
        }}

        body.compare-on #panel-right {{
            display: block;
        }}

        .map-slot {{
            position: absolute;
            inset: 0;
        }}

        #{left_map_id},
        #compare-map-right {{
            position: absolute !important;
            inset: 0;
            width: 100% !important;
            height: 100% !important;
        }}

        .panel-controls {{
            position: absolute;
            right: 16px;
            bottom: 16px;
            z-index: 1200;
            width: min(380px, calc(100% - 32px));
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #c8d0d7;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 12px 26px rgba(34, 49, 63, 0.16);
            box-sizing: border-box;
        }}

        .drag-handle {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 12px;
            margin-bottom: 10px;
            cursor: grab;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
        }}

        .drag-handle.is-dragging {{
            cursor: grabbing;
        }}

        .panel-heading-label {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #5c6a78;
        }}

        .controls-row {{
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }}

        .controls-row:last-child {{
            margin-bottom: 0;
        }}

        .control-inline {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 6px;
            width: min(300px, 100%);
        }}

        .control-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}

        .control-inline label {{
            font-size: 13px;
            font-weight: 600;
            color: #304050;
        }}

        .panel-controls select,
        .panel-controls input[type="range"],
        .panel-controls input[type="date"] {{
            accent-color: #1e6bb8;
        }}

        .panel-controls select,
        .panel-controls input[type="date"] {{
            font-size: 13px;
            padding: 4px 8px;
            border: 1px solid #c6cdd4;
            border-radius: 6px;
            background: white;
            color: #1d2b38;
            width: 100%;
        }}

        .date-readout,
        .hour-readout {{
            font-size: 14px;
            font-weight: 700;
            text-align: right;
            color: #1d2b38;
        }}

        .slider-row input[type="range"] {{
            width: 100%;
        }}

        .legend-scale-zero {{
            color: #6f7882;
        }}

        .legend-scale-low {{
            color: #d62728;
        }}

        .legend-scale-medium {{
            color: #ff7f0e;
        }}

        .legend-scale-high {{
            color: #2ca02c;
        }}

        .hotspot-size-label {{
            min-width: 44px;
            font-size: 13px;
            text-align: right;
            color: #51606f;
        }}

        .legend-hotspot-control {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #e1e7ec;
        }}

        .legend-hotspot-control-stack {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .legend-hotspot-control label {{
            font-size: 13px;
            font-weight: 600;
            color: #304050;
        }}

        .legend-hotspot-control input[type="range"] {{
            flex: 1 1 auto;
            min-width: 120px;
        }}

        #legend-box {{
            position: fixed;
            left: 24px;
            bottom: 24px;
            z-index: 1300;
            width: min(330px, calc(100% - 48px));
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #c2ccd5;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 14px 28px rgba(36, 49, 63, 0.18);
            box-sizing: border-box;
            font-size: 13px;
            line-height: 1.5;
        }}

        .runtime-status {{
            position: fixed;
            top: 18px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1400;
            max-width: min(680px, calc(100% - 36px));
            padding: 10px 14px;
            border: 1px solid #c8d0d7;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 10px 24px rgba(36, 49, 63, 0.16);
            box-sizing: border-box;
            color: #22303d;
            font-size: 13px;
            line-height: 1.4;
        }}

        .runtime-status-text {{
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .runtime-status-progress {{
            height: 8px;
            border-radius: 999px;
            background: #dfe7ee;
            overflow: hidden;
        }}

        .runtime-status-progress-bar {{
            width: 0%;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #1f7ae0 0%, #4db6ff 100%);
            transition: width 180ms ease;
        }}

        .runtime-status.is-error {{
            border-color: #c94747;
            color: #8f1f1f;
        }}

        .runtime-status.is-error .runtime-status-progress {{
            display: none;
        }}

        .legend-top {{
            display: flex;
            align-items: flex-start;
            justify-content: flex-end;
            gap: 12px;
            margin-bottom: 8px;
        }}

        .legend-title {{
            font-size: 15px;
            font-weight: 700;
            color: #1d2b38;
        }}

        .legend-credit {{
            margin-top: 2px;
            font-size: 11px;
            color: #6a7a89;
        }}

        .legend-toggle {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #41515f;
            user-select: none;
        }}

        .legend-toggle input {{
            margin: 0;
        }}

        .legend-drag-handle {{
            margin-bottom: 8px;
        }}

        .compare-sync-options {{
            display: none;
            gap: 6px;
            flex-direction: column;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e1e7ec;
        }}

        body.compare-on .compare-sync-options {{
            display: flex;
        }}

        .legend-static {{
            color: #4d5c6a;
            margin-bottom: 10px;
        }}

        .legend-select-filter {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 10px;
        }}

        .legend-select-filter label {{
            font-size: 13px;
            font-weight: 600;
            color: #304050;
        }}

        .legend-select-filter select {{
            font-size: 13px;
            padding: 4px 8px;
            border: 1px solid #c6cdd4;
            border-radius: 6px;
            background: white;
            color: #1d2b38;
            width: 100%;
        }}

        #legend-dynamic {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        #legend-right-content {{
            display: none;
        }}

        body.compare-on #legend-right-content {{
            display: block;
        }}

        body.compare-on.legend-unified #legend-right-content {{
            display: none;
        }}

        .leaflet-popup-content {{
            margin: 10px 12px;
        }}

        body.theme-dark {{
            background: #111923;
            color: #e5edf5;
        }}

        body.theme-dark .map-panel {{
            background: #16202b;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.34);
        }}

        body.theme-dark .panel-controls,
        body.theme-dark #legend-box {{
            background: rgba(20, 29, 39, 0.95);
            border-color: #314253;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.34);
        }}

        body.theme-dark .runtime-status {{
            background: rgba(20, 29, 39, 0.95);
            border-color: #314253;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.34);
            color: #d6e1ec;
        }}

        body.theme-dark .runtime-status-progress {{
            background: #223244;
        }}

        body.theme-dark .runtime-status.is-error {{
            border-color: #da6c6c;
            color: #ffb1b1;
        }}

        body.theme-dark .panel-heading-label,
        body.theme-dark .hotspot-size-label,
        body.theme-dark .legend-static,
        body.theme-dark .legend-toggle {{
            color: #c3d0dc;
        }}

        body.theme-dark .control-inline label,
        body.theme-dark .legend-title,
        body.theme-dark .legend-select-filter label,
        body.theme-dark .legend-hotspot-control label,
        body.theme-dark .date-readout,
        body.theme-dark .hour-readout {{
            color: #edf4fb;
        }}

        body.theme-dark .legend-credit {{
            color: #9fb1c3;
        }}

        body.theme-dark .panel-controls select,
        body.theme-dark .panel-controls input[type="date"] {{
            background: #13202c;
            color: #edf4fb;
            border-color: #38506a;
        }}

        body.theme-dark .legend-select-filter select {{
            background: #13202c;
            color: #edf4fb;
            border-color: #38506a;
        }}

        body.theme-dark .panel-controls select,
        body.theme-dark .panel-controls input[type="date"],
        body.theme-dark .panel-controls input[type="range"],
        body.theme-dark .legend-select-filter select,
        body.theme-dark .legend-hotspot-control input[type="range"] {{
            accent-color: #66b7ff;
        }}

        body.theme-dark .legend-hotspot-control,
        body.theme-dark .compare-sync-options {{
            border-color: #314253;
        }}

        body.theme-dark .legend-scale-zero {{
            color: #8a949e;
        }}

        body.theme-dark .legend-scale-low {{
            color: #ff7373;
        }}

        body.theme-dark .legend-scale-medium {{
            color: #ffb454;
        }}

        body.theme-dark .legend-scale-high {{
            color: #5ee38b;
        }}

        body.theme-dark .leaflet-control-zoom a {{
            background: rgba(20, 29, 39, 0.96);
            color: #edf4fb;
            border-bottom-color: #314253;
        }}

        body.theme-dark .leaflet-control-zoom a:hover {{
            background: #233242;
        }}

        body.theme-dark .leaflet-control-attribution {{
            background: rgba(20, 29, 39, 0.92);
            color: #c3d0dc;
        }}

        body.theme-dark .leaflet-control-attribution a {{
            color: #8ec8ff;
        }}

        body.theme-dark .leaflet-popup-content-wrapper,
        body.theme-dark .leaflet-popup-tip {{
            background: #16202b;
            color: #edf4fb;
        }}

        body.theme-dark .leaflet-popup-content {{
            color: #edf4fb;
        }}

        @media (max-width: 1100px) {{
            body.compare-on #map-grid {{
                grid-template-columns: minmax(0, 1fr);
                grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
            }}

            .panel-controls {{
                left: auto;
                right: 12px;
                bottom: 12px;
                width: auto;
            }}

            #legend-box {{
                left: 12px;
                right: 12px;
                bottom: 12px;
                width: auto;
                max-width: none;
            }}
        }}

        .house-marker-icon,
        .house-cluster-icon {{
            background: none !important;
            border: none !important;
        }}
        </style>
        """
    ).strip()


def build_area_level_options_html() -> str:
    return "\n".join(
        f'                        <option value="{level}">{cfg["label"]}</option>'
        for level, cfg in POSTCODE_LEVEL_CONFIG.items()
    )


def build_page_html(
    visualization_options: str,
    area_level_options: str,
) -> str:
    """Return the compare-mode page chrome."""
    left_controls = build_panel_controls_html(
        "left",
        visualization_options,
    )
    right_controls = build_panel_controls_html(
        "right",
        visualization_options,
    )
    return dedent(
        f"""
        <div id="map-shell">
            <div id="map-grid">
                <section class="map-panel" id="panel-left">
                    <div class="map-slot" id="left-map-slot"></div>
                    {left_controls}
                </section>
                <section class="map-panel" id="panel-right">
                    <div class="map-slot">
                        <div id="compare-map-right"></div>
                    </div>
                    {right_controls}
                </section>
            </div>
        </div>
        <div id="legend-box">
            <div class="drag-handle legend-drag-handle" id="legend-drag-handle">
                <span class="legend-title">Available Bikes</span>
                <div class="legend-credit">By Lorenzo Rota</div>
            </div>
            <div class="legend-top">
                <label class="legend-toggle" for="theme-toggle">
                    <input type="checkbox" id="theme-toggle">
                    <span>Dark mode</span>
                </label>
                <label class="legend-toggle" for="compare-toggle">
                    <input type="checkbox" id="compare-toggle">
                    <span>Compare mode</span>
                </label>
            </div>
            <div class="compare-sync-options" id="compare-sync-options">
                <label class="legend-toggle" for="sync-movement-toggle">
                    <input type="checkbox" id="sync-movement-toggle" checked>
                    <span>Sync movement</span>
                </label>
                <label class="legend-toggle" for="sync-visualization-toggle">
                    <input type="checkbox" id="sync-visualization-toggle" checked>
                    <span>Sync visualization</span>
                </label>
                <label class="legend-toggle" for="sync-date-toggle">
                    <input type="checkbox" id="sync-date-toggle">
                    <span>Sync date</span>
                </label>
                <label class="legend-toggle" for="sync-time-toggle">
                    <input type="checkbox" id="sync-time-toggle">
                    <span>Sync time</span>
                </label>
            </div>
            <div class="legend-static">
                <div class="legend-select-filter">
                    <label for="provider-filter">Provider</label>
                    <select id="provider-filter"></select>
                </div>
                <div class="legend-select-filter">
                    <label for="postcode-level-toggle">Area level</label>
                    <select id="postcode-level-toggle">
{area_level_options}
                    </select>
                </div>
                Station bikes available:<br>
                <span class="legend-scale-zero">&#9679;</span> 0<br>
                <span class="legend-scale-low">&#9679;</span> 1 &ndash; 3<br>
                <span class="legend-scale-medium">&#9679;</span> 4 &ndash; 6<br>
                <span class="legend-scale-high">&#9679;</span> 7+
            </div>
            <div id="legend-dynamic">
                <div id="legend-left-content"></div>
                <div id="legend-right-content"></div>
            </div>
        </div>
        <div id="runtime-status" class="runtime-status" hidden>
            <div id="runtime-status-text" class="runtime-status-text"></div>
            <div class="runtime-status-progress">
                <div id="runtime-status-progress-bar" class="runtime-status-progress-bar"></div>
            </div>
        </div>
        """
    ).strip()


def build_custom_js(
    *,
    map_js: str,
    map_id: str,
    default_visualization_json: str,
    visualization_js: str,
    artifacts_index_path: str,
    postcode_geojson_paths_json: str,
    postcode_configs_json: str,
    postcode_levels_json: str,
    default_postcode_level_json: str,
    house_data_path: str,
) -> str:
    """Build the client-side compare-mode behavior."""
    js_template = """
    <script>
    window.addEventListener('load', async function() {
        document.body.classList.add('compare-off');
        document.body.classList.add('theme-light');

        var leftMap = __LEFT_MAP__;
        var allData = Object.create(null);
        var dates = [];
        var globalMax = 0;
        var globalMaxByLevel = Object.create(null);
        var areaGeojsonByLevel = Object.create(null);
        var areaIndexByLevel = Object.create(null);
        var postcodeLevelLoadPromises = Object.create(null);
        var dateArtifacts = Object.create(null);
        var stationArtifacts = Object.create(null);
        var dateLoadPromises = Object.create(null);
        var providerLoadPromises = Object.create(null);
        var stationMetadataPromises = Object.create(null);
        var loadedDateKeys = Object.create(null);
        var providerDateData = Object.create(null);
        var defaultVisualizationMode = __DEFAULT_VISUALIZATION_MODE__;
        var defaultInitialDate = '2026-03-01';
        var defaultCenter = __DEFAULT_CENTER__;
        var defaultZoom = __DEFAULT_ZOOM__;
        var artifactsIndexPath = '__ARTIFACTS_INDEX_PATH__';
        var postcodeGeojsonPaths = __POSTCODE_GEOJSON_PATHS__;
        var postcodeConfigs = __POSTCODE_CONFIGS__;
        var postcodeLevels = __POSTCODE_LEVELS__;
        var postcodeLevel = __DEFAULT_POSTCODE_LEVEL__;
        var houseDataPath = '__HOUSE_DATA_PATH__';
        var denHaagBBox = __DEN_HAAG_BBOX__;
        var allProvidersValue = __ALL_PROVIDERS_VALUE__;
        var providerInfo = __MAP_PROVIDER_INFO__;
        var providerOrder = __MAP_PROVIDER_ORDER__;
        var availableProviderKeys = [];
        var activeProvider = __DEFAULT_PROVIDER__;
        var compareEnabled = false;
        var selectedAreaCode = null;
        var viewportSyncInProgress = false;
        var panelStateSyncInProgress = false;
        var rightPanelInitialized = false;
        var panels = null;
        var statusEl = document.getElementById('runtime-status');
        var statusTextEl = document.getElementById('runtime-status-text');
        var statusProgressBarEl = document.getElementById('runtime-status-progress-bar');
        var houseData = null;
        var houseDataPromise = null;

        __VISUALIZATION_JS__

        var leftMapElement = document.getElementById('__LEFT_MAP_ID__');
        var leftMapSlot = document.getElementById('left-map-slot');
        if (leftMapElement && leftMapSlot && leftMapElement.parentNode !== leftMapSlot) {
            leftMapSlot.appendChild(leftMapElement);
        }

        var rightMap = L.map('compare-map-right', {
            zoomControl: true,
            attributionControl: true
        });
        rightMap.setView(defaultCenter, defaultZoom);

        var themeToggle = document.getElementById('theme-toggle');
        var compareToggle = document.getElementById('compare-toggle');
        var syncMovementToggle = document.getElementById('sync-movement-toggle');
        var syncVisualizationToggle = document.getElementById('sync-visualization-toggle');
        var syncDateToggle = document.getElementById('sync-date-toggle');
        var syncTimeToggle = document.getElementById('sync-time-toggle');
        var providerFilterEl = document.getElementById('provider-filter');
        var postcodeLevelEl = document.getElementById('postcode-level-toggle');
        var themeStorageKey = 'fairmss-den-haag-postcode-theme';
        var baseLayers = { left: null, right: null };
        var tileConfig = {
            light: {
                url: '__LIGHT_MAP_TILE_URL__',
                attribution: '__MAP_TILE_ATTRIBUTION__'
            },
            dark: {
                url: '__DARK_MAP_TILE_URL__',
                attribution: '__MAP_TILE_ATTRIBUTION__'
            }
        };
        var syncSettings = {
            movement: true,
            visualization: true,
            date: false,
            time: false
        };
        var POINT_MARKER_RADIUS = 5;
        var POINT_MARKER_RADIUS_SELECTED = 8;

        function normalizeAreaCode(value) {
            if (value === null || value === undefined) return '';
            return String(value).trim();
        }

        function getPostcodeConfig(level) {
            return postcodeConfigs[level] || postcodeConfigs[__DEFAULT_POSTCODE_LEVEL__];
        }

        function getActivePostcodeConfig() {
            return getPostcodeConfig(postcodeLevel);
        }

        function getAreaLabel(level) {
            return getPostcodeConfig(level || postcodeLevel).label;
        }

        function getAreaProperty(level) {
            return getPostcodeConfig(level || postcodeLevel).property;
        }

        function getAreaGeojson(level) {
            return areaGeojsonByLevel[level || postcodeLevel] || null;
        }

        function getAreaIndex(level) {
            return areaIndexByLevel[level || postcodeLevel] || [];
        }

        function getAreaCounts(dateData, level) {
            if (!dateData || !dateData.counts) return {};
            return dateData.counts[level || postcodeLevel] || {};
        }

        function getStationAreaCode(station, level) {
            return normalizeAreaCode(
                station && station.pc ? station.pc[level || postcodeLevel] : ''
            );
        }

        function setStatusProgress(percent) {
            if (!statusProgressBarEl) return;
            var bounded = Math.max(0, Math.min(100, Number(percent) || 0));
            statusProgressBarEl.style.width = bounded + '%';
        }

        function clearStatusSoon(delayMs) {
            window.setTimeout(function() {
                showStatus('', 'info');
            }, delayMs || 0);
        }

        function showStatus(message, kind, progressPercent) {
            if (!statusEl) return;
            if (!message) {
                statusEl.hidden = true;
                if (statusTextEl) {
                    statusTextEl.textContent = '';
                }
                statusEl.classList.remove('is-error');
                setStatusProgress(0);
                return;
            }
            statusEl.hidden = false;
            if (statusTextEl) {
                statusTextEl.textContent = message;
            } else {
                statusEl.textContent = message;
            }
            statusEl.classList.toggle('is-error', kind === 'error');
            if (kind !== 'error') {
                setStatusProgress(progressPercent);
            }
        }

        function fetchJson(path) {
            return fetch(path).then(function(response) {
                if (!response.ok) {
                    throw new Error('Request failed for ' + path + ' (' + response.status + ')');
                }
                return response.json();
            });
        }

        function mergeFeatureCollections(parts) {
            var merged = {
                type: 'FeatureCollection',
                features: []
            };
            for (var i = 0; i < parts.length; i++) {
                var part = parts[i] || {};
                var features = Array.isArray(part.features) ? part.features : [];
                merged.features = merged.features.concat(features);
                if (!merged.crs && part.crs) {
                    merged.crs = part.crs;
                }
            }
            return merged;
        }

        function fetchGeojsonResource(resource) {
            if (Array.isArray(resource)) {
                return Promise.all(resource.map(fetchJson)).then(mergeFeatureCollections);
            }
            return fetchJson(resource);
        }

        function fetchText(path) {
            return fetch(path).then(function(response) {
                if (!response.ok) {
                    throw new Error('Request failed for ' + path + ' (' + response.status + ')');
                }
                return response.text();
            });
        }

        function ensureHouseDataLoaded(silent) {
            if (houseData) {
                return Promise.resolve(houseData);
            }
            if (houseDataPromise) {
                return houseDataPromise;
            }
            if (!silent) {
                showStatus('Loading house points...', 'info', 78);
            }
            houseDataPromise = fetchJson(houseDataPath).then(function(payload) {
                houseData = payload;
                return payload;
            }).then(function(payload) {
                if (!silent) {
                    showStatus('Loaded house points', 'info', 92);
                    clearStatusSoon(500);
                }
                return payload;
            }).catch(function(error) {
                houseDataPromise = null;
                showStatus('Failed to load house points: ' + error.message, 'error');
                throw error;
            });
            return houseDataPromise;
        }

        function csvLines(text) {
            var normalized = String(text || '');
            if (normalized.charCodeAt(0) === 0xFEFF) {
                normalized = normalized.slice(1);
            }
            return normalized
                .split('\\n')
                .map(function(line) {
                    return line.endsWith('\\r') ? line.slice(0, -1) : line;
                })
                .filter(function(line) {
                    return line.trim().length > 0;
                });
        }

        function parseCsvLine(line) {
            var result = [];
            var current = '';
            var inQuotes = false;
            for (var i = 0; i < line.length; i += 1) {
                var ch = line[i];
                if (ch === '"') {
                    if (inQuotes && line[i + 1] === '"') {
                        current += '"';
                        i += 1;
                    } else {
                        inQuotes = !inQuotes;
                    }
                } else if (ch === ',' && !inQuotes) {
                    result.push(current);
                    current = '';
                } else {
                    current += ch;
                }
            }
            result.push(current);
            return result;
        }

        function round6(value) {
            return Math.round(value * 1000000) / 1000000;
        }

        function parseHourFromTimestamp(timestamp) {
            if (!timestamp || timestamp.length < 13) return 0;
            return parseInt(timestamp.slice(11, 13), 10) || 0;
        }

        function zeroArray(length) {
            var arr = new Array(length);
            for (var i = 0; i < length; i += 1) {
                arr[i] = 0;
            }
            return arr;
        }

        function withinDenHaagBBox(lat, lon) {
            return lat >= denHaagBBox.lat_min &&
                   lat <= denHaagBBox.lat_max &&
                   lon >= denHaagBBox.lon_min &&
                   lon <= denHaagBBox.lon_max;
        }

        function pointInRing(lon, lat, ring) {
            var inside = false;
            for (var i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
                var xi = ring[i][0];
                var yi = ring[i][1];
                var xj = ring[j][0];
                var yj = ring[j][1];
                var intersects = ((yi > lat) !== (yj > lat)) &&
                    (lon < ((xj - xi) * (lat - yi)) / ((yj - yi) || 1e-12) + xi);
                if (intersects) {
                    inside = !inside;
                }
            }
            return inside;
        }

        function pointInPolygon(lon, lat, polygon) {
            if (!polygon || polygon.length === 0) return false;
            if (!pointInRing(lon, lat, polygon[0])) return false;
            for (var i = 1; i < polygon.length; i += 1) {
                if (pointInRing(lon, lat, polygon[i])) {
                    return false;
                }
            }
            return true;
        }

        function buildAreaIndex(geojson, level) {
            var areaProperty = getAreaProperty(level);
            var features = geojson && geojson.features ? geojson.features : [];
            return features.map(function(feature) {
                var rawAreaCode = feature.properties && feature.properties[areaProperty];
                var areaCode = normalizeAreaCode(rawAreaCode);
                var polygons = [];
                if (feature.geometry && feature.geometry.type === 'Polygon') {
                    polygons = [feature.geometry.coordinates];
                } else if (feature.geometry && feature.geometry.type === 'MultiPolygon') {
                    polygons = feature.geometry.coordinates;
                }
                var bbox = {
                    minLon: Infinity,
                    minLat: Infinity,
                    maxLon: -Infinity,
                    maxLat: -Infinity
                };
                for (var p = 0; p < polygons.length; p += 1) {
                    for (var r = 0; r < polygons[p].length; r += 1) {
                        for (var c = 0; c < polygons[p][r].length; c += 1) {
                            var coord = polygons[p][r][c];
                            bbox.minLon = Math.min(bbox.minLon, coord[0]);
                            bbox.maxLon = Math.max(bbox.maxLon, coord[0]);
                            bbox.minLat = Math.min(bbox.minLat, coord[1]);
                            bbox.maxLat = Math.max(bbox.maxLat, coord[1]);
                        }
                    }
                }
                return {
                    key: areaCode,
                    areaCode: areaCode,
                    polygons: polygons,
                    bbox: bbox
                };
            }).filter(function(entry) {
                return !!entry.key;
            });
        }

        function findAreaCodeForPoint(lat, lon, level) {
            var areaIndex = getAreaIndex(level);
            for (var i = 0; i < areaIndex.length; i += 1) {
                var entry = areaIndex[i];
                if (lon < entry.bbox.minLon || lon > entry.bbox.maxLon ||
                    lat < entry.bbox.minLat || lat > entry.bbox.maxLat) {
                    continue;
                }
                for (var p = 0; p < entry.polygons.length; p += 1) {
                    if (pointInPolygon(lon, lat, entry.polygons[p])) {
                        return entry.areaCode;
                    }
                }
            }
            return '';
        }

        function buildEmptyCountsByLevel(maxHour) {
            var countsByLevel = Object.create(null);
            for (var l = 0; l < postcodeLevels.length; l += 1) {
                var level = postcodeLevels[l];
                countsByLevel[level] = Object.create(null);
            }
            return countsByLevel;
        }

        function ensureAreaCountEntry(levelCounts, areaCode, maxHour) {
            if (!levelCounts[areaCode]) {
                levelCounts[areaCode] = {
                    c: zeroArray(maxHour + 1),
                    s: zeroArray(maxHour + 1),
                    f: zeroArray(maxHour + 1)
                };
            }
            return levelCounts[areaCode];
        }

        function getProviderLabel(providerKey) {
            var info = providerInfo[providerKey] || {};
            return info.label || providerKey;
        }

        function getSelectedProviderKeys() {
            if (activeProvider === allProvidersValue) {
                return availableProviderKeys.slice();
            }
            return availableProviderKeys.indexOf(activeProvider) !== -1 ? [activeProvider] : [];
        }

        function artifactPathRank(path) {
            if (!path) return 0;
            if (path.indexOf('/docked_') !== -1) return 3;
            if (path.indexOf('/dockless_') !== -1) return 3;
            if (path.indexOf('/stations_') !== -1) return 3;
            if (path.indexOf('/docks_') !== -1) return 2;
            return 1;
        }

        function shouldReplaceArtifactPath(currentPath, nextPath) {
            if (!currentPath) return true;
            return artifactPathRank(nextPath) >= artifactPathRank(currentPath);
        }

        function buildArtifactCatalog(rows) {
            var discoveredProviders = Object.create(null);
            dateArtifacts = Object.create(null);
            stationArtifacts = Object.create(null);
            rows.forEach(function(row) {
                if (!row || !row.path || !row.date || !row.provider) return;
                if (!Object.prototype.hasOwnProperty.call(providerInfo, row.provider)) return;
                discoveredProviders[row.provider] = true;
                if (!dateArtifacts[row.date]) {
                    dateArtifacts[row.date] = Object.create(null);
                }
                if (!dateArtifacts[row.date][row.provider]) {
                    dateArtifacts[row.date][row.provider] = Object.create(null);
                }
                if (!stationArtifacts[row.provider]) {
                    stationArtifacts[row.provider] = [];
                }

                var relativePath = '../' + row.path;
                var providerArtifacts = dateArtifacts[row.date][row.provider];
                if (row.artifact_type === 'docked_data') {
                    if (shouldReplaceArtifactPath(providerArtifacts.docked, relativePath)) {
                        providerArtifacts.docked = relativePath;
                    }
                } else if (row.artifact_type === 'dockless_data') {
                    if (shouldReplaceArtifactPath(providerArtifacts.dockless, relativePath)) {
                        providerArtifacts.dockless = relativePath;
                    }
                } else if (row.artifact_type === 'stations_data') {
                    stationArtifacts[row.provider].push({
                        date: row.date,
                        path: relativePath
                    });
                }
            });

            providerOrder.forEach(function(providerKey) {
                if (!stationArtifacts[providerKey]) {
                    stationArtifacts[providerKey] = [];
                }
                stationArtifacts[providerKey].sort(function(a, b) {
                    return a.date.localeCompare(b.date);
                });
                if (!providerDateData[providerKey]) {
                    providerDateData[providerKey] = Object.create(null);
                }
            });

            availableProviderKeys = providerOrder.filter(function(providerKey) {
                return !!discoveredProviders[providerKey];
            });
            dates = Object.keys(dateArtifacts).filter(function(dateKey) {
                var perProvider = dateArtifacts[dateKey];
                return availableProviderKeys.some(function(providerKey) {
                    return !!(perProvider[providerKey] && perProvider[providerKey].docked);
                });
            }).sort();
        }

        function populateProviderFilterOptions() {
            if (!providerFilterEl) return;
            providerFilterEl.innerHTML = '';

            var allOption = document.createElement('option');
            allOption.value = allProvidersValue;
            allOption.textContent = 'All providers';
            providerFilterEl.appendChild(allOption);

            availableProviderKeys.forEach(function(providerKey) {
                var option = document.createElement('option');
                option.value = providerKey;
                option.textContent = getProviderLabel(providerKey);
                providerFilterEl.appendChild(option);
            });

            if (availableProviderKeys.indexOf(activeProvider) === -1) {
                activeProvider = allProvidersValue;
            }
            providerFilterEl.value = activeProvider;
            providerFilterEl.disabled = availableProviderKeys.length <= 1;
        }

        function getStationArtifactForProviderDate(providerKey, dateKey) {
            var rows = stationArtifacts[providerKey] || [];
            if (!rows.length) return null;
            for (var i = rows.length - 1; i >= 0; i -= 1) {
                if (rows[i].date <= dateKey) {
                    return rows[i];
                }
            }
            return rows[0];
        }

        function loadStationMetadataForProviderDate(providerKey, dateKey) {
            var artifact = getStationArtifactForProviderDate(providerKey, dateKey);
            if (!artifact) {
                return Promise.resolve({
                    sourceDate: null,
                    stations: [],
                    stationToAreaByLevel: Object.create(null)
                });
            }
            if (stationMetadataPromises[artifact.path]) {
                return stationMetadataPromises[artifact.path];
            }
            stationMetadataPromises[artifact.path] = fetchText(artifact.path).then(function(csvText) {
                var lines = csvLines(csvText);
                if (!lines.length) {
                    return {
                        sourceDate: artifact.date,
                        stations: [],
                        stationToAreaByLevel: Object.create(null)
                    };
                }
                var header = parseCsvLine(lines[0]);
                var idxStationId = header.indexOf('station_id');
                var idxName = header.indexOf('name');
                var idxLat = header.indexOf('lat');
                var idxLon = header.indexOf('lon');
                var idxCapacity = header.indexOf('capacity');
                if (idxStationId === -1 || idxName === -1 || idxLat === -1 ||
                    idxLon === -1 || idxCapacity === -1) {
                    throw new Error('Station metadata missing required columns in ' + artifact.path);
                }
                var stations = [];
                var stationToAreaByLevel = Object.create(null);
                for (var l = 0; l < postcodeLevels.length; l += 1) {
                    stationToAreaByLevel[postcodeLevels[l]] = Object.create(null);
                }
                for (var i = 1; i < lines.length; i += 1) {
                    var fields = parseCsvLine(lines[i]);
                    var lat = parseFloat(fields[idxLat]);
                    var lon = parseFloat(fields[idxLon]);
                    if (!isFinite(lat) || !isFinite(lon)) continue;
                    if (!withinDenHaagBBox(lat, lon)) continue;
                    var stationId = String(fields[idxStationId] || '').trim();
                    if (!stationId) continue;
                    var areaCodes = Object.create(null);
                    for (var j = 0; j < postcodeLevels.length; j += 1) {
                        var level = postcodeLevels[j];
                        var areaCode = findAreaCodeForPoint(lat, lon, level);
                        areaCodes[level] = areaCode;
                        if (areaCode) {
                            stationToAreaByLevel[level][stationId] = areaCode;
                        }
                    }
                    stations.push({
                        id: stationId,
                        name: fields[idxName] || stationId,
                        lat: lat,
                        lon: lon,
                        capacity: parseInt(fields[idxCapacity], 10) || 0,
                        pc: areaCodes
                    });
                }
                return {
                    sourceDate: artifact.date,
                    stations: stations,
                    stationToAreaByLevel: stationToAreaByLevel
                };
            });
            return stationMetadataPromises[artifact.path];
        }

        function computeDateMax(counts) {
            var dateMax = 0;
            for (var pc in counts) {
                if (!Object.prototype.hasOwnProperty.call(counts, pc)) continue;
                var values = counts[pc].c || [];
                for (var i = 0; i < values.length; i += 1) {
                    if (values[i] > dateMax) {
                        dateMax = values[i];
                    }
                }
            }
            return dateMax;
        }

        function computeDateMaxByLevel(dateData) {
            var maxByLevel = Object.create(null);
            for (var i = 0; i < postcodeLevels.length; i += 1) {
                var level = postcodeLevels[i];
                maxByLevel[level] = computeDateMax(getAreaCounts(dateData, level));
            }
            return maxByLevel;
        }

        function buildHotspotData(stations, bikesByHour, hours) {
            var hotspot = Object.create(null);
            for (var i = 0; i < hours.length; i += 1) {
                var hour = hours[i];
                var stationPoints = [];
                var stationMax = 0;
                for (var j = 0; j < stations.length; j += 1) {
                    var station = stations[j];
                    if (hour >= station.av.length) continue;
                    var avail = station.av[hour];
                    if (avail <= 0) continue;
                    stationPoints.push([station.ll[0], station.ll[1], avail]);
                    if (avail > stationMax) {
                        stationMax = avail;
                    }
                }
                hotspot[String(hour)] = {
                    stations: stationPoints,
                    stationMax: stationMax
                };
            }
            return hotspot;
        }

        function buildDateData(providerKey, dateKey, stationMeta, dockedCsvText) {
            var dockedLines = csvLines(dockedCsvText);
            if (dockedLines.length < 2) {
                throw new Error('Docked table is empty for ' + dateKey);
            }
            var header = parseCsvLine(dockedLines[0]);
            var colIndexByStationId = Object.create(null);
            for (var c = 1; c < header.length; c += 1) {
                colIndexByStationId[String(header[c])] = c;
            }

            var hourToRow = Object.create(null);
            for (var i = 1; i < dockedLines.length; i += 1) {
                var fields = parseCsvLine(dockedLines[i]);
                if (!fields.length || !fields[0]) continue;
                var hour = parseHourFromTimestamp(fields[0]);
                if (!Object.prototype.hasOwnProperty.call(hourToRow, hour)) {
                    hourToRow[hour] = fields;
                }
            }

            var hours = Object.keys(hourToRow).map(function(value) {
                return parseInt(value, 10);
            }).sort(function(a, b) {
                return a - b;
            });
            if (!hours.length) {
                throw new Error('No hourly docked data found for ' + dateKey);
            }

            var maxHour = hours[hours.length - 1];
            var counts = buildEmptyCountsByLevel(maxHour);

            var stationsJs = [];
            var stationToAreaByLevel = stationMeta.stationToAreaByLevel || Object.create(null);
            var stations = stationMeta.stations || [];
            for (var s = 0; s < stations.length; s += 1) {
                var station = stations[s];
                var stationId = station.id;
                var hourlyAvail = zeroArray(maxHour + 1);
                var colIndex = colIndexByStationId[stationId];
                if (colIndex !== undefined) {
                    for (var h = 0; h < hours.length; h += 1) {
                        var hourKey = hours[h];
                        var row = hourToRow[hourKey];
                        var rawValue = row && row[colIndex] !== undefined ? row[colIndex] : '0';
                        var value = parseInt(rawValue, 10) || 0;
                        hourlyAvail[hourKey] = value;
                        for (var l = 0; l < postcodeLevels.length; l += 1) {
                            var level = postcodeLevels[l];
                            var areaCode = normalizeAreaCode(
                                stationToAreaByLevel[level] &&
                                stationToAreaByLevel[level][stationId]
                            );
                            if (areaCode && counts[level]) {
                                var stationCountEntry = ensureAreaCountEntry(
                                    counts[level],
                                    areaCode,
                                    maxHour
                                );
                                stationCountEntry.s[hourKey] += value;
                                stationCountEntry.c[hourKey] += value;
                            }
                        }
                    }
                }
                var stationAreaCodes = Object.create(null);
                for (var pcLevelIdx = 0; pcLevelIdx < postcodeLevels.length; pcLevelIdx += 1) {
                    var pcLevel = postcodeLevels[pcLevelIdx];
                    stationAreaCodes[pcLevel] = normalizeAreaCode(
                        station.pc && station.pc[pcLevel]
                    );
                }
                stationsJs.push({
                    ll: [round6(station.lat), round6(station.lon)],
                    n: station.name,
                    cap: station.capacity,
                    pc: stationAreaCodes,
                    av: hourlyAvail,
                    pr: providerKey
                });
            }

            var bikesByHour = Object.create(null);
            for (var h2 = 0; h2 < hours.length; h2 += 1) {
                bikesByHour[String(hours[h2])] = [];
            }

            var dateData = {
                hours: hours,
                maxHour: maxHour,
                counts: counts,
                bikes: bikesByHour,
                stations: stationsJs,
                hotspot: buildHotspotData(stationsJs, bikesByHour, hours),
                areaLevelsReady: Object.create(null)
            };
            for (var l2 = 0; l2 < postcodeLevels.length; l2 += 1) {
                var readyLevel = postcodeLevels[l2];
                dateData.areaLevelsReady[readyLevel] = getAreaIndex(readyLevel).length > 0;
            }
            dateData.dateMaxByLevel = computeDateMaxByLevel(dateData);
            return dateData;
        }

        function composeDateData(dateKey) {
            var providerKeys = getSelectedProviderKeys();
            var sourceItems = [];
            var hoursLookup = Object.create(null);
            var maxHour = 0;
            for (var i = 0; i < providerKeys.length; i += 1) {
                var providerKey = providerKeys[i];
                var providerDates = providerDateData[providerKey] || Object.create(null);
                var sourceDateData = providerDates[dateKey];
                if (!sourceDateData) continue;
                sourceItems.push(sourceDateData);
                var sourceHours = sourceDateData.hours || [];
                for (var j = 0; j < sourceHours.length; j += 1) {
                    var hour = sourceHours[j];
                    hoursLookup[hour] = true;
                    if (hour > maxHour) {
                        maxHour = hour;
                    }
                }
            }

            if (!sourceItems.length) {
                delete allData[dateKey];
                return null;
            }

            var hours = Object.keys(hoursLookup).map(function(value) {
                return parseInt(value, 10);
            }).sort(function(a, b) {
                return a - b;
            });
            var counts = buildEmptyCountsByLevel(maxHour);

            var stations = [];
            var bikesByHour = Object.create(null);
            for (var h = 0; h < hours.length; h += 1) {
                bikesByHour[String(hours[h])] = [];
            }

            for (var s = 0; s < sourceItems.length; s += 1) {
                var item = sourceItems[s];
                for (var l = 0; l < postcodeLevels.length; l += 1) {
                    var level = postcodeLevels[l];
                    var itemCounts = getAreaCounts(item, level);
                    var targetLevelCounts = counts[level] || {};
                    for (var pcKey in itemCounts) {
                        if (!Object.prototype.hasOwnProperty.call(itemCounts, pcKey)) continue;
                        var targetCount = ensureAreaCountEntry(
                            targetLevelCounts,
                            pcKey,
                            maxHour
                        );
                        var sourceCount = itemCounts[pcKey];
                        for (var idx = 0; idx < sourceCount.c.length; idx += 1) {
                            targetCount.c[idx] += sourceCount.c[idx] || 0;
                            targetCount.s[idx] += sourceCount.s[idx] || 0;
                            targetCount.f[idx] += sourceCount.f[idx] || 0;
                        }
                    }
                }

                var itemStations = item.stations || [];
                for (var st = 0; st < itemStations.length; st += 1) {
                    stations.push(itemStations[st]);
                }

                var itemHours = item.hours || [];
                for (var ih = 0; ih < itemHours.length; ih += 1) {
                    var hourKey = String(itemHours[ih]);
                    var bikes = item.bikes[hourKey] || [];
                    for (var b = 0; b < bikes.length; b += 1) {
                        bikesByHour[hourKey].push(bikes[b]);
                    }
                }
            }

            var dateData = {
                hours: hours,
                maxHour: maxHour,
                counts: counts,
                bikes: bikesByHour,
                stations: stations,
                hotspot: buildHotspotData(stations, bikesByHour, hours),
                areaLevelsReady: Object.create(null)
            };
            for (var l2 = 0; l2 < postcodeLevels.length; l2 += 1) {
                var readyLevel = postcodeLevels[l2];
                dateData.areaLevelsReady[readyLevel] = sourceItems.every(function(item) {
                    return !!(item.areaLevelsReady && item.areaLevelsReady[readyLevel]);
                });
            }
            dateData.dateMaxByLevel = computeDateMaxByLevel(dateData);
            allData[dateKey] = dateData;
            return dateData;
        }

        function getLoadedDates() {
            return dates.filter(function(dateKey) {
                return !!loadedDateKeys[dateKey];
            });
        }

        function recomputeActiveData() {
            allData = Object.create(null);
            globalMaxByLevel = Object.create(null);
            for (var l = 0; l < postcodeLevels.length; l += 1) {
                globalMaxByLevel[postcodeLevels[l]] = 0;
            }
            var loadedDates = getLoadedDates();
            for (var i = 0; i < loadedDates.length; i += 1) {
                var dateData = composeDateData(loadedDates[i]);
                if (dateData) {
                    for (var j = 0; j < postcodeLevels.length; j += 1) {
                        var level = postcodeLevels[j];
                        globalMaxByLevel[level] = Math.max(
                            globalMaxByLevel[level],
                            (dateData.dateMaxByLevel && dateData.dateMaxByLevel[level]) || 0
                        );
                    }
                }
            }
            globalMax = globalMaxByLevel[postcodeLevel] || 0;
        }

        function refreshActivePanels() {
            if (!panels) return;
            panels.left.renderAll();
            panels.right.renderAll();
            updateLegendLayout();
        }

        function rebuildPanelAreaLayer(panel) {
            if (!panel) return;
            if (panel.geojsonLayer) {
                panel.map.removeLayer(panel.geojsonLayer);
            }
            panel.geojsonLayer = buildPolygonLayer(panel);
        }

        function ensureDateDataHasLevel(dateData, level) {
            if (!dateData) return;
            if (!dateData.areaLevelsReady) {
                dateData.areaLevelsReady = Object.create(null);
            }
            if (dateData.areaLevelsReady[level]) {
                return;
            }

            var counts = Object.create(null);
            var hours = dateData.hours || [];
            var maxHour = dateData.maxHour || 0;
            var stations = dateData.stations || [];
            for (var i = 0; i < stations.length; i += 1) {
                var station = stations[i];
                if (!station.pc) {
                    station.pc = Object.create(null);
                }
                var areaCode = getStationAreaCode(station, level);
                if (!areaCode) {
                    areaCode = findAreaCodeForPoint(station.ll[0], station.ll[1], level);
                    station.pc[level] = areaCode;
                }
                if (!areaCode) continue;
                var stationEntry = ensureAreaCountEntry(counts, areaCode, maxHour);
                for (var h = 0; h < hours.length; h += 1) {
                    var hourKey = hours[h];
                    var value = station.av[hourKey] || 0;
                    if (value <= 0) continue;
                    stationEntry.s[hourKey] += value;
                    stationEntry.c[hourKey] += value;
                }
            }

            dateData.counts[level] = counts;
            if (!dateData.dateMaxByLevel) {
                dateData.dateMaxByLevel = Object.create(null);
            }
            dateData.dateMaxByLevel[level] = computeDateMax(counts);
            dateData.areaLevelsReady[level] = true;
        }

        function ensurePostcodeLevelReady(level, silent) {
            if (getAreaGeojson(level) && getAreaIndex(level).length) {
                return Promise.resolve();
            }
            if (postcodeLevelLoadPromises[level]) {
                return postcodeLevelLoadPromises[level];
            }
            if (!silent) {
                showStatus('Loading ' + getAreaLabel(level) + ' boundaries...', 'info', 30);
            }
            postcodeLevelLoadPromises[level] = fetchGeojsonResource(postcodeGeojsonPaths[level])
                .then(function(geojson) {
                    areaGeojsonByLevel[level] = geojson;
                    areaIndexByLevel[level] = buildAreaIndex(geojson, level);

                    for (var providerKey in providerDateData) {
                        if (!Object.prototype.hasOwnProperty.call(providerDateData, providerKey)) {
                            continue;
                        }
                        var providerDates = providerDateData[providerKey] || {};
                        for (var dateKey in providerDates) {
                            if (!Object.prototype.hasOwnProperty.call(providerDates, dateKey)) {
                                continue;
                            }
                            ensureDateDataHasLevel(providerDates[dateKey], level);
                        }
                    }
                    recomputeActiveData();
                    if (!silent) {
                        showStatus('Loaded ' + getAreaLabel(level) + ' boundaries', 'info', 90);
                        clearStatusSoon(500);
                    }
                })
                .catch(function(error) {
                    delete postcodeLevelLoadPromises[level];
                    if (!silent) {
                        showStatus(
                            'Failed to load ' + getAreaLabel(level) + ': ' + error.message,
                            'error'
                        );
                    }
                    throw error;
                });
            return postcodeLevelLoadPromises[level];
        }

        function setPostcodeLevel(nextLevel) {
            var level = Object.prototype.hasOwnProperty.call(postcodeConfigs, nextLevel)
                ? nextLevel
                : __DEFAULT_POSTCODE_LEVEL__;
            if (postcodeLevelEl) {
                postcodeLevelEl.value = level;
                postcodeLevelEl.disabled = true;
            }
            return ensurePostcodeLevelReady(level, false).then(function() {
                postcodeLevel = level;
                globalMax = globalMaxByLevel[postcodeLevel] || 0;
                clearSelection();
                if (!panels) return;
                rebuildPanelAreaLayer(panels.left);
                rebuildPanelAreaLayer(panels.right);
                refreshActivePanels();
            }).finally(function() {
                if (postcodeLevelEl) {
                    postcodeLevelEl.disabled = false;
                    postcodeLevelEl.value = postcodeLevel;
                }
            });
        }

        function ensureDateDataLoaded(dateKey, options) {
            var silent = options && options.silent;
            if (loadedDateKeys[dateKey]) {
                return Promise.resolve(getDateData(dateKey));
            }
            if (dateLoadPromises[dateKey]) {
                return dateLoadPromises[dateKey];
            }
            var artifactsByProvider = dateArtifacts[dateKey];
            if (!artifactsByProvider) {
                return Promise.reject(new Error('Missing docked table for ' + dateKey));
            }
            if (!silent) {
                showStatus('Loading ' + dateKey + '...', 'info', 70);
            }
            var providerLoads = availableProviderKeys.map(function(providerKey) {
                var artifacts = artifactsByProvider[providerKey];
                if (!artifacts || !artifacts.docked) {
                    return Promise.resolve(null);
                }
                if (!providerDateData[providerKey]) {
                    providerDateData[providerKey] = Object.create(null);
                }
                if (providerDateData[providerKey][dateKey]) {
                    return Promise.resolve(providerDateData[providerKey][dateKey]);
                }
                var cacheKey = providerKey + '|' + dateKey;
                if (providerLoadPromises[cacheKey]) {
                    return providerLoadPromises[cacheKey];
                }
                providerLoadPromises[cacheKey] = Promise.all([
                    loadStationMetadataForProviderDate(providerKey, dateKey),
                    fetchText(artifacts.docked)
                ]).then(function(results) {
                    var dateData = buildDateData(
                        providerKey,
                        dateKey,
                        results[0],
                        results[1]
                    );
                    providerDateData[providerKey][dateKey] = dateData;
                    return dateData;
                });
                return providerLoadPromises[cacheKey];
            });
            dateLoadPromises[dateKey] = Promise.all(providerLoads).then(function() {
                loadedDateKeys[dateKey] = true;
                recomputeActiveData();
                var dateData = getDateData(dateKey);
                if (!silent) {
                    showStatus('Loaded ' + dateKey, 'info', 90);
                    clearStatusSoon(500);
                } else if (panels &&
                           (panels.left.visualizationMode === 'global' ||
                            panels.right.visualizationMode === 'global')) {
                    refreshActivePanels();
                }
                return dateData;
            }).catch(function(error) {
                delete dateLoadPromises[dateKey];
                if (!silent) {
                    showStatus(
                        'Failed to load ' + dateKey + ': ' + error.message,
                        'error',
                    );
                }
                throw error;
            });
            return dateLoadPromises[dateKey];
        }

        function warmRemainingDates() {
            var pendingDates = dates.filter(function(dateKey) {
                return !loadedDateKeys[dateKey];
            });
            function loadNext(index) {
                if (index >= pendingDates.length) {
                    return;
                }
                ensureDateDataLoaded(pendingDates[index], { silent: true })
                    .catch(function() {
                        return null;
                    })
                    .finally(function() {
                        window.setTimeout(function() {
                            loadNext(index + 1);
                        }, 0);
                    });
            }
            loadNext(0);
        }

        function readStoredTheme() {
            try {
                return window.localStorage.getItem(themeStorageKey);
            } catch (error) {
                return null;
            }
        }

        function storeTheme(themeName) {
            try {
                window.localStorage.setItem(themeStorageKey, themeName);
            } catch (error) {
                return;
            }
        }

        function applyBodyTheme(themeName) {
            var dark = themeName === 'dark';
            document.body.classList.toggle('theme-dark', dark);
            document.body.classList.toggle('theme-light', !dark);
            if (themeToggle) {
                themeToggle.checked = dark;
            }
        }

        function setBaseLayer(map, panelId, themeName) {
            if (baseLayers[panelId]) {
                map.removeLayer(baseLayers[panelId]);
            }
            var cfg = tileConfig[themeName] || tileConfig.light;
            baseLayers[panelId] = L.tileLayer(cfg.url, {
                subdomains: 'abcd',
                maxZoom: 20,
                attribution: cfg.attribution
            }).addTo(map);
        }

        function applyTheme(themeName) {
            var normalized = themeName === 'dark' ? 'dark' : 'light';
            applyBodyTheme(normalized);
            storeTheme(normalized);
            setBaseLayer(leftMap, 'left', normalized);
            setBaseLayer(rightMap, 'right', normalized);
            if (panels) {
                panels.left.renderAll();
                panels.right.renderAll();
                updateLegendLayout();
            }
        }

        function getStationMarkerStyle(isSelected, muted, availability) {
            var selectedColor = themeColor('selectedMarker');
            var baseColor = stationColorForAvailability(availability);
            return {
                color: isSelected ? selectedColor : baseColor,
                fillColor: baseColor,
                fillOpacity: isSelected ? 1.0 : (muted ? 0.45 : 0.8),
                weight: isSelected ? 2.5 : 1.5
            };
        }

        function buildHouseNoticeHtml(message) {
            return (
                '<div style="margin-top:8px;color:' + themeColor('legendSubtleText') + ';">' +
                escapeHtml(message) +
                '</div>'
            );
        }

        function getHouseMarkerStyle(distanceMeters) {
            var fillColor = houseColorForDistance(distanceMeters);
            return {
                color: fillColor,
                fillColor: fillColor,
                fillOpacity: distanceMeters === null ? 0.28 : 0.56,
                weight: 0
            };
        }

        function formatDistanceMeters(distanceMeters) {
            if (distanceMeters === null || !isFinite(distanceMeters)) {
                return 'No available bike';
            }
            if (distanceMeters < 1000) {
                return Math.round(distanceMeters) + ' m';
            }
            return (distanceMeters / 1000).toFixed(2) + ' km';
        }

        function approximateDistanceMeters(lat1, lon1, lat2, lon2) {
            var radians = Math.PI / 180;
            var x = (lon2 - lon1) * radians * Math.cos(((lat1 + lat2) / 2) * radians);
            var y = (lat2 - lat1) * radians;
            return 6371000 * Math.sqrt((x * x) + (y * y));
        }

        function collectVisibleHousePoints(map, limit, callback) {
            if (!houseData || !houseData.cells) {
                return {
                    houses: [],
                    truncated: false
                };
            }
            var bounds = map.getBounds();
            var south = bounds.getSouth();
            var north = bounds.getNorth();
            var west = bounds.getWest();
            var east = bounds.getEast();
            var cellSize = Number(houseData.cellSize) || 0.0025;
            var minCellX = Math.floor(west / cellSize);
            var maxCellX = Math.floor(east / cellSize);
            var minCellY = Math.floor(south / cellSize);
            var maxCellY = Math.floor(north / cellSize);
            var visible = [];
            var maxItems = Number(limit);
            var truncated = false;

            for (var cellY = minCellY; cellY <= maxCellY; cellY += 1) {
                for (var cellX = minCellX; cellX <= maxCellX; cellX += 1) {
                    var cellKey = cellX + ':' + cellY;
                    var coords = houseData.cells[cellKey];
                    if (!coords) continue;
                    // Triples: [lat_i, lon_i, address_count, ...]
                    for (var i = 0; i < coords.length; i += 3) {
                        var lat = coords[i] / 1000000;
                        var lon = coords[i + 1] / 1000000;
                        var addrCount = coords[i + 2] || 1;
                        if (lat < south || lat > north || lon < west || lon > east) {
                            continue;
                        }
                        var house = [lat, lon, addrCount];
                        if (callback) {
                            callback(house);
                        }
                        if (!isFinite(maxItems) || maxItems <= 0 || visible.length < maxItems) {
                            visible.push(house);
                        } else {
                            truncated = true;
                        }
                    }
                }
            }
            return {
                houses: visible,
                truncated: truncated
            };
        }

        function getVisibleHousePoints(map, limit) {
            return collectVisibleHousePoints(map, limit, null);
        }

        function getHouseSupplyPoints(dateData, hour) {
            if (!dateData) {
                return [];
            }
            if (!dateData.houseSupplyPoints) {
                dateData.houseSupplyPoints = Object.create(null);
            }
            var hourKey = String(hour);
            if (dateData.houseSupplyPoints[hourKey]) {
                return dateData.houseSupplyPoints[hourKey];
            }

            var points = [];
            var seen = Object.create(null);
            var stations = dateData.stations || [];
            for (var j = 0; j < stations.length; j += 1) {
                var station = stations[j];
                var availability = station.av[hour] !== undefined ? station.av[hour] : 0;
                if (availability <= 0) continue;
                var stationKey = station.ll[0] + '|' + station.ll[1];
                if (seen[stationKey]) continue;
                seen[stationKey] = true;
                points.push([station.ll[0], station.ll[1]]);
            }

            dateData.houseSupplyPoints[hourKey] = points;
            return points;
        }

        function getClosestSupplyDistance(lat, lon, supplyPoints) {
            if (!supplyPoints.length) {
                return null;
            }
            var best = Infinity;
            for (var i = 0; i < supplyPoints.length; i += 1) {
                var point = supplyPoints[i];
                var distanceMeters = approximateDistanceMeters(lat, lon, point[0], point[1]);
                if (distanceMeters < best) {
                    best = distanceMeters;
                }
            }
            return isFinite(best) ? best : null;
        }

        applyBodyTheme(readStoredTheme() === 'dark' ? 'dark' : 'light');
        setBaseLayer(leftMap, 'left', getThemeName());
        setBaseLayer(rightMap, 'right', getThemeName());

        function buildControls(panelId) {
            return {
                container: document.getElementById(panelId + '-controls-box'),
                dragHandle: document.querySelector(
                    '#' + panelId + '-controls-box .panel-drag-handle'
                ),
                headingLabel: document.getElementById(panelId + '-panel-heading-label'),
                dateLabel: document.getElementById(panelId + '-date-label'),
                dateInput: document.getElementById(panelId + '-date-input'),
                hourLabel: document.getElementById(panelId + '-hour-label'),
                hourSlider: document.getElementById(panelId + '-hour-slider'),
                visualization: document.getElementById(panelId + '-visualization-mode')
            };
        }

        function getPanelLegendElement(panelId) {
            return document.getElementById('legend-' + panelId + '-content');
        }

        function getDateData(dateKey) {
            return allData[dateKey] || null;
        }

        function getClosestAvailableDate(dateKey) {
            if (!dates.length) return null;
            if (dates.indexOf(dateKey) !== -1) return dateKey;
            if (dateKey <= dates[0]) return dates[0];
            if (dateKey >= dates[dates.length - 1]) return dates[dates.length - 1];
            for (var i = 1; i < dates.length; i += 1) {
                if (dates[i] >= dateKey) {
                    var previousDate = dates[i - 1];
                    var nextDate = dates[i];
                    var previousDistance = Math.abs(
                        new Date(previousDate + 'T00:00:00').getTime() -
                        new Date(dateKey + 'T00:00:00').getTime()
                    );
                    var nextDistance = Math.abs(
                        new Date(nextDate + 'T00:00:00').getTime() -
                        new Date(dateKey + 'T00:00:00').getTime()
                    );
                    return previousDistance <= nextDistance ? previousDate : nextDate;
                }
            }
            return dates[dates.length - 1];
        }

        function normalizeHour(dateData, requestedHour) {
            if (!dateData || !dateData.hours || dateData.hours.length === 0) return 0;
            if (dateData.hours.indexOf(requestedHour) !== -1) return requestedHour;
            return dateData.hours[0];
        }

        function getOtherPanel(panel) {
            return panel.id === 'left' ? panels.right : panels.left;
        }

        function shouldShowPointMarkers(panel) {
            return !isHotspotMode(panel.visualizationMode) ||
                   panel.map.getZoom() >= HOTSPOT_MARKER_ZOOM_THRESHOLD;
        }

        function getPolygonWeight(panel, areaCode) {
            if (isHouseMode(panel.visualizationMode)) return 0.8;
            if (selectedAreaCode !== null && areaCode === selectedAreaCode) return 5;
            return isHotspotMode(panel.visualizationMode) ? 1.5 : 2.5;
        }

        function getHoverWeight(panel) {
            if (isHouseMode(panel.visualizationMode)) return 1.1;
            return isHotspotMode(panel.visualizationMode) ? 3.0 : 5.0;
        }

        function closePanelPopup(panel) {
            if (panel.activePopup) {
                panel.map.closePopup(panel.activePopup);
                panel.activePopup = null;
            } else {
                panel.map.closePopup();
            }
        }

        function closeAllPopups() {
            if (!panels) return;
            closePanelPopup(panels.left);
            closePanelPopup(panels.right);
        }

        function updatePanelHeadings() {
            if (!panels) return;
            panels.left.controls.headingLabel.textContent = compareEnabled
                ? 'Left map controls'
                : 'Map controls';
            panels.right.controls.headingLabel.textContent = 'Right map controls';
        }

        function getPanelLegendMarkup(panel) {
            return (panel.legendBaseHtml || '') + (panel.legendControlHtml || '');
        }

        function buildHotspotControlHtml(panel, label) {
            return (
                '<div class="legend-hotspot-control">' +
                '<label for="' + panel.id + '-legend-hotspot-size-slider">' +
                escapeHtml(label) +
                '</label>' +
                '<input type="range" id="' + panel.id + '-legend-hotspot-size-slider" min="60" max="300" value="' +
                Math.round(panel.hotspotRadiusScale * 100) + '">' +
                '<span class="hotspot-size-label" id="' + panel.id + '-legend-hotspot-size-label">' +
                Math.round(panel.hotspotRadiusScale * 100) + '%</span>' +
                '</div>'
            );
        }

        function buildUnifiedHotspotControl() {
            return (
                '<div class="legend-hotspot-control">' +
                '<label for="legend-unified-hotspot-size-slider">Base hotspot size</label>' +
                '<input type="range" id="legend-unified-hotspot-size-slider" min="60" max="300" value="' +
                Math.round(panels.left.hotspotRadiusScale * 100) + '">' +
                '<span class="hotspot-size-label" id="legend-unified-hotspot-size-label">' +
                Math.round(panels.left.hotspotRadiusScale * 100) + '%</span>' +
                '</div>'
            );
        }

        function bindUnifiedHotspotControl() {
            var hotspotSlider = document.getElementById('legend-unified-hotspot-size-slider');
            if (!hotspotSlider || hotspotSlider.dataset.bound) return;

            hotspotSlider.dataset.bound = 'true';
            hotspotSlider.addEventListener('input', function() {
                var sliderValue = parseInt(this.value, 10);
                var scale = sliderValue / 100.0;
                panels.left.hotspotRadiusScale = scale;
                panels.right.hotspotRadiusScale = scale;
                var label = document.getElementById('legend-unified-hotspot-size-label');
                if (label) {
                    label.textContent = sliderValue + '%';
                }
                panels.left.renderHotspotLayer();
                panels.right.renderHotspotLayer();
            });
        }

        function updateLegendLayout() {
            if (!panels) return;
            var leftBaseLegend = panels.left.legendBaseHtml || '';
            var rightBaseLegend = panels.right.legendBaseHtml || '';
            var leftLegend = getPanelLegendMarkup(panels.left);
            var rightLegend = getPanelLegendMarkup(panels.right);
            var unified = compareEnabled &&
                panels.left.visualizationMode === panels.right.visualizationMode;
            document.body.classList.toggle('legend-unified', unified);
            if (unified) {
                if (isHotspotMode(panels.left.visualizationMode)) {
                    panels.right.hotspotRadiusScale = panels.left.hotspotRadiusScale;
                }
                if (leftBaseLegend === rightBaseLegend) {
                    panels.left.legendEl.innerHTML = leftBaseLegend;
                    if (isHotspotMode(panels.left.visualizationMode)) {
                        panels.left.legendEl.innerHTML += buildUnifiedHotspotControl();
                    }
                } else {
                    panels.left.legendEl.innerHTML = leftLegend;
                    if (isHotspotMode(panels.left.visualizationMode)) {
                        panels.left.legendEl.innerHTML += buildUnifiedHotspotControl();
                    }
                }
                panels.right.legendEl.innerHTML = '';
                bindUnifiedHotspotControl();
            } else {
                panels.left.legendEl.innerHTML = leftLegend;
                panels.right.legendEl.innerHTML = rightLegend;
            }
        }

        function syncViewport(sourcePanel, targetPanel) {
            if (!compareEnabled || !syncSettings.movement || viewportSyncInProgress) return;
            viewportSyncInProgress = true;
            targetPanel.map.setView(sourcePanel.map.getCenter(), sourcePanel.map.getZoom(), {
                animate: false,
                reset: true
            });
            window.setTimeout(function() {
                viewportSyncInProgress = false;
            }, 0);
        }

        function syncViewportFrom(panel) {
            if (!compareEnabled || !syncSettings.movement || viewportSyncInProgress) return;
            syncViewport(panel, getOtherPanel(panel));
        }

        function copyPanelState(sourcePanel, targetPanel) {
            targetPanel.currentDate = sourcePanel.currentDate;
            targetPanel.currentHour = sourcePanel.currentHour;
            targetPanel.requestedHour = sourcePanel.requestedHour;
            targetPanel.visualizationMode = sourcePanel.visualizationMode;
            targetPanel.hotspotRadiusScale = sourcePanel.hotspotRadiusScale;
        }

        function syncPartnerPanel(sourcePanel, options) {
            if (!compareEnabled || panelStateSyncInProgress) return;
            var targetPanel = getOtherPanel(sourcePanel);
            if (!targetPanel) return;

            panelStateSyncInProgress = true;
            var shouldUpdate = false;
            if (options.date && syncSettings.date) {
                targetPanel.currentDate = sourcePanel.currentDate;
                targetPanel.currentHour = normalizeHour(
                    getDateData(targetPanel.currentDate),
                    targetPanel.requestedHour
                );
                shouldUpdate = true;
            }
            if (options.hour && syncSettings.time) {
                targetPanel.requestedHour = sourcePanel.requestedHour;
                targetPanel.currentHour = normalizeHour(
                    getDateData(targetPanel.currentDate),
                    targetPanel.requestedHour
                );
                shouldUpdate = true;
            }
            if (options.visualization && syncSettings.visualization) {
                targetPanel.visualizationMode = sourcePanel.visualizationMode;
                shouldUpdate = true;
            }
            if (shouldUpdate) {
                targetPanel.syncControlsFromState();
                targetPanel.refreshFromState();
            }
            panelStateSyncInProgress = false;
        }

        function getBoundsRect(boundsTarget) {
            if (boundsTarget === window) {
                return {
                    left: 0,
                    top: 0,
                    width: window.innerWidth,
                    height: window.innerHeight
                };
            }
            return boundsTarget.getBoundingClientRect();
        }

        function enableDragging(element, handle, boundsTarget) {
            var dragState = null;

            function onPointerMove(event) {
                if (!dragState) return;
                var boundsRect = getBoundsRect(boundsTarget);
                var maxLeft = Math.max(boundsRect.width - element.offsetWidth, 0);
                var maxTop = Math.max(boundsRect.height - element.offsetHeight, 0);
                var nextLeft = dragState.startLeft + (event.clientX - dragState.startX);
                var nextTop = dragState.startTop + (event.clientY - dragState.startY);
                nextLeft = Math.min(Math.max(nextLeft, 0), maxLeft);
                nextTop = Math.min(Math.max(nextTop, 0), maxTop);
                element.style.left = nextLeft + 'px';
                element.style.top = nextTop + 'px';
            }

            function onPointerUp() {
                if (!dragState) return;
                dragState = null;
                handle.classList.remove('is-dragging');
                document.removeEventListener('pointermove', onPointerMove);
                document.removeEventListener('pointerup', onPointerUp);
            }

            handle.addEventListener('pointerdown', function(event) {
                if (event.button !== 0) return;
                event.preventDefault();
                var boundsRect = getBoundsRect(boundsTarget);
                var rect = element.getBoundingClientRect();
                dragState = {
                    startX: event.clientX,
                    startY: event.clientY,
                    startLeft: rect.left - boundsRect.left,
                    startTop: rect.top - boundsRect.top
                };
                element.style.left = dragState.startLeft + 'px';
                element.style.top = dragState.startTop + 'px';
                element.style.right = 'auto';
                element.style.bottom = 'auto';
                handle.classList.add('is-dragging');
                document.addEventListener('pointermove', onPointerMove);
                document.addEventListener('pointerup', onPointerUp);
            });
        }

        function getAreaStats(panel, areaCode) {
            var dateData = getDateData(panel.currentDate);
            var dateCounts = getAreaCounts(dateData);
            var pcData = dateCounts[String(areaCode)] || { c: [], s: [], f: [] };
            return {
                total: pcData.c[panel.currentHour] || 0,
                docked: pcData.s[panel.currentHour] || 0
            };
        }

        function openPanelPopup(panel, areaCode, latlng) {
            var stats = getAreaStats(panel, areaCode);
            closePanelPopup(panel);
            panel.activePopup = L.popup()
                .setLatLng(latlng)
                .setContent(
                    '<b>' + escapeHtml(getAreaLabel()) + ' ' + escapeHtml(areaCode) + '</b><br>' +
                    'Available bikes: <b>' + stats.total + '</b><br>' +
                    '&nbsp;&nbsp;Station-based: ' + stats.docked
                )
                .openOn(panel.map);
        }

        function clearSelection() {
            if (!panels) return;
            selectedAreaCode = null;
            closeAllPopups();
            panels.left.renderPolygons();
            panels.right.renderPolygons();
            panels.left.applySelection(null);
            panels.right.applySelection(null);
        }

        function setSelectedAreaCode(areaCode, sourcePanel, latlng) {
            if (selectedAreaCode === areaCode) {
                clearSelection();
                return;
            }
            selectedAreaCode = areaCode;
            closeAllPopups();
            panels.left.renderPolygons();
            panels.right.renderPolygons();
            panels.left.applySelection(areaCode);
            panels.right.applySelection(areaCode);
            if (sourcePanel && latlng) {
                openPanelPopup(sourcePanel, areaCode, latlng);
            }
        }

        function buildPolygonLayer(panel) {
            return L.geoJSON(getAreaGeojson() || { type: 'FeatureCollection', features: [] }, {
                style: function(feature) {
                    var areaCode = normalizeAreaCode(
                        feature.properties && feature.properties[getAreaProperty()]
                    );
                    return {
                        color: getPolygonStrokeColor(panel.visualizationMode),
                        weight: getPolygonWeight(panel, areaCode),
                        fillColor: themeColor('scaleZero'),
                        fillOpacity: getPolygonFillOpacity(panel.visualizationMode)
                    };
                },
                onEachFeature: function(feature, layer) {
                    layer.on('mouseover', function() {
                        if (isHouseMode(panel.visualizationMode)) return;
                        var areaCode = normalizeAreaCode(
                            feature.properties && feature.properties[getAreaProperty()]
                        );
                        if (selectedAreaCode !== areaCode) {
                            layer.setStyle({ weight: getHoverWeight(panel) });
                        }
                    });

                    layer.on('mouseout', function() {
                        if (isHouseMode(panel.visualizationMode)) return;
                        var areaCode = normalizeAreaCode(
                            feature.properties && feature.properties[getAreaProperty()]
                        );
                        if (selectedAreaCode !== areaCode) {
                            layer.setStyle({ weight: getPolygonWeight(panel, areaCode) });
                        }
                    });

                    layer.on('click', function(e) {
                        if (isHouseMode(panel.visualizationMode)) return;
                        L.DomEvent.stopPropagation(e);
                        var areaCode = normalizeAreaCode(
                            feature.properties && feature.properties[getAreaProperty()]
                        );
                        setSelectedAreaCode(areaCode, panel, e.latlng);
                    });
                }
            }).addTo(panel.map);
        }

        function createPanel(panelId, mapInstance) {
            var controls = buildControls(panelId);
            var panel = {
                id: panelId,
                map: mapInstance,
                controls: controls,
                legendEl: getPanelLegendElement(panelId),
                geojsonLayer: null,
                houseLayer: L.layerGroup().addTo(mapInstance),
                hotspotLayer: L.layerGroup().addTo(mapInstance),
                stationLayer: L.layerGroup().addTo(mapInstance),
                houseRenderer: L.canvas({ padding: 0.2 }),
                stationMarkers: [],
                currentDate: null,
                currentHour: 0,
                requestedHour: 0,
                visualizationMode: defaultVisualizationMode,
                hotspotRadiusScale: 1.0,
                activePopup: null,
                legendHtml: '',
                legendBaseHtml: '',
                legendControlHtml: '',
                houseNoticeHtml: '',
                loadVersion: 0,
                refreshFromState: function(syncOptions) {
                    var panelRef = this;
                    if (!this.currentDate) {
                        this.syncControlsFromState();
                        this.renderAll();
                        return Promise.resolve(null);
                    }
                    var loadVersion = ++this.loadVersion;
                    return ensureDateDataLoaded(this.currentDate).then(function(dateData) {
                        if (panelRef.loadVersion !== loadVersion) {
                            return dateData;
                        }
                        panelRef.currentHour = normalizeHour(dateData, panelRef.requestedHour);
                        panelRef.syncControlsFromState();
                        panelRef.renderAll();
                        if (syncOptions) {
                            syncPartnerPanel(panelRef, syncOptions);
                        }
                        return dateData;
                    }).catch(function(error) {
                        if (panelRef.loadVersion === loadVersion) {
                            showStatus(
                                'Failed to load ' + panelRef.currentDate + ': ' + error.message,
                                'error'
                            );
                        }
                        return null;
                    });
                },
                setDate: function(dateStr) {
                    if (dates.indexOf(dateStr) === -1) return;
                    this.currentDate = dateStr;
                    closeAllPopups();
                    this.refreshFromState({ date: true, hour: true });
                },
                setHour: function(hour) {
                    this.requestedHour = hour;
                    this.currentHour = hour;
                    closeAllPopups();
                    this.syncControlsFromState();
                    this.renderAll();
                    syncPartnerPanel(this, { hour: true });
                },
                setVisualization: function(mode) {
                    var panelRef = this;
                    var oldMode = this.visualizationMode;
                    this.visualizationMode = mode;
                    if (isHouseMode(oldMode) || isHouseMode(mode)) {
                        clearSelection();
                    }
                    closeAllPopups();
                    this.syncControlsFromState();
                    var renderPromise = isHouseMode(mode)
                        ? ensureHouseDataLoaded(false)
                        : Promise.resolve(null);
                    renderPromise.then(function() {
                        panelRef.renderAll();
                        syncPartnerPanel(panelRef, { visualization: true });
                    }).catch(function() {
                        panelRef.renderAll();
                    });
                },
                syncControlsFromState: function() {
                    var dateData = getDateData(this.currentDate);
                    this.controls.dateInput.disabled = dates.length === 0;
                    this.controls.dateLabel.textContent = this.currentDate || 'No date';
                    this.controls.dateInput.min = dates.length ? dates[0] : '';
                    this.controls.dateInput.max = dates.length ? dates[dates.length - 1] : '';
                    this.controls.dateInput.value = this.currentDate || '';
                    this.controls.visualization.value = this.visualizationMode;
                    this.controls.visualization.disabled = !this.currentDate;
                    this.controls.hourSlider.disabled = !dateData;
                    if (!dateData) {
                        this.controls.hourSlider.min = 0;
                        this.controls.hourSlider.max = 0;
                        this.controls.hourSlider.value = 0;
                        this.controls.hourLabel.textContent = '--:--';
                        return;
                    }
                    this.controls.hourSlider.min = dateData.hours[0];
                    this.controls.hourSlider.max = dateData.maxHour;
                    this.controls.hourSlider.value = this.currentHour;
                    this.controls.hourLabel.textContent =
                        String(this.currentHour).padStart(2, '0') + ':00';
                },
                renderLegend: function() {
                    var legendBaseHtml = legendHtmlForMode(
                        allData,
                        globalMax,
                        this.visualizationMode,
                        this.currentDate,
                        this.currentHour
                    );
                    var legendControlHtml = '';
                    if (isHotspotMode(this.visualizationMode)) {
                        legendControlHtml = buildHotspotControlHtml(this, 'Base hotspot size');
                    } else if (isHouseMode(this.visualizationMode)) {
                        legendControlHtml = this.houseNoticeHtml || '';
                    }
                    this.legendBaseHtml = legendBaseHtml;
                    this.legendControlHtml = legendControlHtml;
                    this.legendHtml = legendBaseHtml + legendControlHtml;
                    var panelRef = this;
                    updateLegendLayout();
                    var hotspotSlider = document.getElementById(
                        this.id + '-legend-hotspot-size-slider'
                    );
                    if (hotspotSlider && !hotspotSlider.dataset.bound) {
                        hotspotSlider.dataset.bound = 'true';
                        hotspotSlider.addEventListener('input', function() {
                            var sliderValue = parseInt(this.value, 10);
                            panelRef.hotspotRadiusScale = sliderValue / 100.0;
                            var label = document.getElementById(
                                panelRef.id + '-legend-hotspot-size-label'
                            );
                            if (label) {
                                label.textContent = sliderValue + '%';
                            }
                            panelRef.renderHotspotLayer();
                        });
                    }
                },
                renderPolygons: function() {
                    var panelRef = this;
                    var dateData = getDateData(this.currentDate);
                    var dateCounts = getAreaCounts(dateData);
                    this.geojsonLayer.eachLayer(function(layer) {
                        var areaCode = normalizeAreaCode(
                            layer.feature.properties[getAreaProperty()]
                        );
                        var pcData = dateCounts[areaCode];
                        var count = (pcData && pcData.c[panelRef.currentHour] !== undefined)
                            ? pcData.c[panelRef.currentHour]
                            : 0;
                        var style = polygonStyleForCount(
                            count,
                            allData,
                            globalMax,
                            panelRef.visualizationMode,
                            panelRef.currentDate,
                            panelRef.currentHour
                        );
                        style.weight = getPolygonWeight(panelRef, areaCode);
                        layer.setStyle(style);
                    });
                },
                renderHotspotLayer: function() {
                    this.hotspotLayer.clearLayers();
                    if (!isHotspotMode(this.visualizationMode)) return;

                    var hotspotData = getHotspotHourData(
                        allData,
                        this.currentDate,
                        this.currentHour
                    );
                    var stations = hotspotData.stations || [];
                    var stationMax = hotspotData.stationMax || 0;
                    var zoomLevel = this.map.getZoom();

                    for (var j = 0; j < stations.length; j++) {
                        var station = stations[j];
                        var avail = station[2];
                        L.circleMarker([station[0], station[1]], {
                            radius: hotspotStationRadius(
                                avail,
                                stationMax,
                                this.hotspotRadiusScale,
                                zoomLevel
                            ),
                            stroke: false,
                            fillColor: hotspotFillColor(avail, stationMax),
                            fillOpacity: hotspotStationOpacity(avail, stationMax),
                            interactive: false
                        }).addTo(this.hotspotLayer);
                    }
                },
                renderHouses: function() {
                    this.houseLayer.clearLayers();
                    this.houseNoticeHtml = '';
                    if (!isHouseMode(this.visualizationMode)) return;
                    if (!houseData) {
                        this.houseNoticeHtml = buildHouseNoticeHtml(
                            'House points are still loading.'
                        );
                        return;
                    }
                    var zoom = this.map.getZoom();
                    if (zoom < HOUSE_MODE_MIN_ZOOM) {
                        this.houseNoticeHtml = buildHouseNoticeHtml(
                            'Zoom to level ' + HOUSE_MODE_MIN_ZOOM + ' or closer for house points.'
                        );
                        return;
                    }

                    var visibleHouseResult = getVisibleHousePoints(
                        this.map,
                        HOUSE_MODE_MAX_VISIBLE
                    );
                    var visibleHouses = visibleHouseResult.houses;
                    if (!visibleHouses.length && !visibleHouseResult.truncated) {
                        this.houseNoticeHtml = buildHouseNoticeHtml(
                            'No house points are visible in the current map view.'
                        );
                        return;
                    }

                    var dateData = getDateData(this.currentDate);
                    var supplyPoints = getHouseSupplyPoints(dateData, this.currentHour);
                    if (!supplyPoints.length) {
                        this.houseNoticeHtml = buildHouseNoticeHtml(
                            'No available bikes were found for the selected hour.'
                        );
                    }

                    var useDetail = zoom >= HOUSE_MODE_DETAIL_ZOOM &&
                                    !visibleHouseResult.truncated;

                    if (useDetail) {
                        // Individual house markers (square icons)
                        var markerSize = zoom >= 17 ? 12 : (zoom >= 16 ? 10 : 8);
                        for (var i = 0; i < visibleHouses.length; i += 1) {
                            var house = visibleHouses[i];
                            var addrCount = house[2] || 1;
                            var distanceMeters = getClosestSupplyDistance(
                                house[0], house[1], supplyPoints
                            );
                            var color = houseColorForDistance(distanceMeters);
                            var icon = createHouseIcon(color, markerSize);
                            var marker = L.marker([house[0], house[1]], {
                                icon: icon,
                                interactive: true
                            });
                            var popupContent =
                                '<div style="font-size:13px;line-height:1.5;">' +
                                '<b>Residential address' + (addrCount > 1 ? 'es' : '') + '</b><br>' +
                                'Addresses at this location: <b>' + addrCount + '</b><br>' +
                                'Closest available bike: <b>' +
                                escapeHtml(formatDistanceMeters(distanceMeters)) + '</b>' +
                                '</div>';
                            marker.bindPopup(popupContent);
                            marker.bindTooltip(
                                addrCount + ' addr \u00b7 ' +
                                escapeHtml(formatDistanceMeters(distanceMeters))
                            );
                            marker.addTo(this.houseLayer);
                        }
                    } else {
                        // Clustered view: aggregate all visible houses into screen-space bins
                        var clusters = clusterHousePoints(this.map, supplyPoints);
                        for (var j = 0; j < clusters.length; j++) {
                            var cl = clusters[j];
                            var clColor = houseColorForDistance(cl.avgDistance);
                            var clSize = Math.min(
                                36,
                                Math.max(20, 16 + Math.log2(Math.max(cl.locationCount, 1)) * 3)
                            );
                            var clIcon = createHouseClusterIcon(clColor, Math.round(clSize), cl.locationCount);
                            var clMarker = L.marker([cl.lat, cl.lon], {
                                icon: clIcon,
                                interactive: true
                            });
                            var clPopup =
                                '<div style="font-size:13px;line-height:1.5;">' +
                                '<b>House cluster</b><br>' +
                                'Buildings: <b>' + cl.locationCount + '</b><br>' +
                                'Total addresses: <b>' + cl.totalAddresses + '</b><br>' +
                                'Avg distance to bike: <b>' +
                                escapeHtml(formatDistanceMeters(cl.avgDistance)) + '</b>' +
                                '</div>';
                            clMarker.bindPopup(clPopup);
                            clMarker.bindTooltip(
                                cl.locationCount + ' buildings \u00b7 ' +
                                escapeHtml(formatDistanceMeters(cl.avgDistance))
                            );
                            clMarker.addTo(this.houseLayer);
                        }
                    }
                },
                renderStations: function() {
                    this.stationLayer.clearLayers();
                    this.stationMarkers = [];
                    if (!shouldShowPointMarkers(this)) return;

                    var dateData = getDateData(this.currentDate);
                    if (!dateData) return;
                    var stations = dateData.stations || [];
                    var muted = isHotspotMode(this.visualizationMode);
                    for (var i = 0; i < stations.length; i++) {
                        var station = stations[i];
                        var avail = (station.av[this.currentHour] !== undefined)
                            ? station.av[this.currentHour]
                            : 0;
                        var stationAreaCode = getStationAreaCode(station);
                        var isSelected = (
                            selectedAreaCode !== null && stationAreaCode === selectedAreaCode
                        );
                        var style = getStationMarkerStyle(isSelected, muted, avail);
                        var marker = L.circleMarker(station.ll, {
                            radius: isSelected ? POINT_MARKER_RADIUS_SELECTED : POINT_MARKER_RADIUS,
                            color: style.color,
                            fillColor: style.fillColor,
                            fillOpacity: style.fillOpacity,
                            weight: style.weight
                        });
                        marker.bindTooltip(
                            escapeHtml(station.n) +
                            '<br>Provider: ' + escapeHtml(getProviderLabel(station.pr)) +
                            '<br>Available: ' + avail + ' / ' + station.cap
                        );
                        marker._areaCode = stationAreaCode;
                        marker._avail = avail;
                        marker.addTo(this.stationLayer);
                        this.stationMarkers.push(marker);
                    }
                },
                applySelection: function(areaCode) {
                    var muted = isHotspotMode(this.visualizationMode);
                    for (var j = 0; j < this.stationMarkers.length; j++) {
                        var stationMarker = this.stationMarkers[j];
                        var stationStyle;
                        if (areaCode !== null && stationMarker._areaCode === areaCode) {
                            stationMarker.setRadius(POINT_MARKER_RADIUS_SELECTED);
                            stationStyle = getStationMarkerStyle(
                                true,
                                muted,
                                stationMarker._avail
                            );
                        } else {
                            stationMarker.setRadius(POINT_MARKER_RADIUS);
                            stationStyle = getStationMarkerStyle(
                                false,
                                muted,
                                stationMarker._avail
                            );
                        }
                        stationMarker.setStyle(stationStyle);
                    }
                },
                clearSelection: function() {
                    this.applySelection(null);
                    closePanelPopup(this);
                },
                renderAll: function() {
                    this.renderPolygons();
                    this.renderHotspotLayer();
                    this.renderHouses();
                    this.renderStations();
                    this.renderLegend();
                    this.applySelection(selectedAreaCode);
                }
            };

            panel.geojsonLayer = buildPolygonLayer(panel);

            panel.controls.dateInput.addEventListener('change', function() {
                var dateValue = getClosestAvailableDate(this.value);
                if (dateValue !== null) {
                    panel.setDate(dateValue);
                }
            });

            panel.controls.hourSlider.addEventListener('input', function() {
                panel.setHour(parseInt(this.value, 10));
            });

            panel.controls.visualization.addEventListener('change', function() {
                panel.setVisualization(this.value);
            });

            panel.map.on('click', function() {
                clearSelection();
            });

            panel.map.on('moveend', function() {
                if (isHouseMode(panel.visualizationMode)) {
                    panel.renderHouses();
                    panel.renderLegend();
                }
                syncViewportFrom(panel);
            });

            panel.map.on('zoomend', function() {
                panel.renderPolygons();
                panel.renderHotspotLayer();
                panel.renderHouses();
                panel.renderStations();
                panel.renderLegend();
                panel.applySelection(selectedAreaCode);
            });

            L.DomEvent.disableClickPropagation(panel.controls.container);
            enableDragging(
                panel.controls.container,
                panel.controls.dragHandle,
                document.getElementById('panel-' + panelId)
            );

            return panel;
        }

        function setCompareMode(enabled) {
            if (!panels) return;
            compareEnabled = enabled;
            document.body.classList.toggle('compare-on', enabled);
            document.body.classList.toggle('compare-off', !enabled);
            compareToggle.checked = enabled;
            updatePanelHeadings();
            updateLegendLayout();

            if (enabled && !rightPanelInitialized) {
                copyPanelState(panels.left, panels.right);
                rightPanelInitialized = true;
                panels.right.syncControlsFromState();
                panels.right.refreshFromState();
            }

            window.setTimeout(function() {
                panels.left.map.invalidateSize();
                panels.right.map.invalidateSize();
                if (compareEnabled && syncSettings.movement) {
                    syncViewport(panels.left, panels.right);
                }
            }, 120);

            panels.left.renderLegend();
            panels.right.renderLegend();
            updateSyncSettings();
        }

        function syncRightPanelFromLeft(options) {
            if (!panels) return;
            panelStateSyncInProgress = true;
            if (options.visualization) {
                panels.right.visualizationMode = panels.left.visualizationMode;
            }
            if (options.date) {
                panels.right.currentDate = panels.left.currentDate;
                panels.right.currentHour = normalizeHour(
                    getDateData(panels.right.currentDate),
                    panels.right.requestedHour
                );
            }
            if (options.time) {
                panels.right.requestedHour = panels.left.requestedHour;
                panels.right.currentHour = normalizeHour(
                    getDateData(panels.right.currentDate),
                    panels.right.requestedHour
                );
            }
            panels.right.syncControlsFromState();
            panels.right.refreshFromState();
            panelStateSyncInProgress = false;
        }

        function refreshPanelsForProviderChange() {
            if (!panels) return;
            panels.left.currentHour = normalizeHour(
                getDateData(panels.left.currentDate),
                panels.left.requestedHour
            );
            panels.right.currentHour = normalizeHour(
                getDateData(panels.right.currentDate),
                panels.right.requestedHour
            );
            panels.left.syncControlsFromState();
            panels.right.syncControlsFromState();
            panels.left.renderAll();
            panels.right.renderAll();
            updateLegendLayout();
        }

        function updateSyncSettings() {
            syncSettings.movement = syncMovementToggle.checked;
            syncSettings.visualization = syncVisualizationToggle.checked;
            syncSettings.date = syncDateToggle.checked;
            syncSettings.time = syncTimeToggle.checked;

            if (!compareEnabled) return;
            if (syncSettings.movement) {
                syncViewport(panels.left, panels.right);
            }
            if (syncSettings.visualization || syncSettings.date || syncSettings.time) {
                syncRightPanelFromLeft({
                    visualization: syncSettings.visualization,
                    date: syncSettings.date,
                    time: syncSettings.time
                });
            } else {
                updateLegendLayout();
            }
        }

        if (themeToggle) {
            themeToggle.addEventListener('change', function() {
                applyTheme(this.checked ? 'dark' : 'light');
            });
        }

        compareToggle.addEventListener('change', function() {
            setCompareMode(this.checked);
        });

        syncMovementToggle.addEventListener('change', updateSyncSettings);
        syncVisualizationToggle.addEventListener('change', updateSyncSettings);
        syncDateToggle.addEventListener('change', updateSyncSettings);
        syncTimeToggle.addEventListener('change', updateSyncSettings);
        if (providerFilterEl) {
            providerFilterEl.addEventListener('change', function() {
                activeProvider = this.value || allProvidersValue;
                recomputeActiveData();
                refreshPanelsForProviderChange();
            });
        }
        if (postcodeLevelEl) {
            postcodeLevelEl.value = postcodeLevel;
            postcodeLevelEl.addEventListener('change', function() {
                setPostcodeLevel(this.value || __DEFAULT_POSTCODE_LEVEL__);
            });
        }

        L.DomEvent.disableClickPropagation(document.getElementById('legend-box'));
        enableDragging(
            document.getElementById('legend-box'),
            document.getElementById('legend-drag-handle'),
            window
        );

        if (window.location.protocol === 'file:') {
            showStatus(
                'Runtime loading needs a local web server. Open this map via http://localhost/...',
                'error',
            );
        }

        try {
            showStatus('Loading map assets...', 'info', 15);
            var initResults = await Promise.all([
                fetchJson(artifactsIndexPath),
                ensurePostcodeLevelReady(postcodeLevel, true)
            ]);
            showStatus('Preparing provider and area data...', 'info', 40);
            buildArtifactCatalog(initResults[0]);
            populateProviderFilterOptions();

            panels = {
                left: createPanel('left', leftMap),
                right: createPanel('right', rightMap)
            };

            if (!dates.length) {
                updatePanelHeadings();
                panels.left.syncControlsFromState();
                panels.right.syncControlsFromState();
                panels.left.renderAll();
                panels.right.renderAll();
                showStatus('No processed docked data found under output/data.', 'error');
                return;
            }

            var initialDate = getClosestAvailableDate(defaultInitialDate);
            panels.left.currentDate = initialDate;
            panels.right.currentDate = initialDate;

            showStatus('Loading first date...', 'info', 60);
            var firstDateData = await ensureDateDataLoaded(initialDate);
            panels.left.requestedHour = 0;
            panels.left.currentHour = normalizeHour(firstDateData, panels.left.requestedHour);
            panels.right.requestedHour = panels.left.requestedHour;
            panels.right.currentHour = panels.left.currentHour;
            updatePanelHeadings();
            panels.left.syncControlsFromState();
            panels.right.syncControlsFromState();
            panels.left.renderAll();
            panels.right.renderAll();
            updateSyncSettings();
            updateLegendLayout();
            showStatus('Map ready', 'info', 100);
            clearStatusSoon(900);
        } catch (error) {
            showStatus('Failed to initialize map: ' + error.message, 'error');
        }
    });
    </script>
    """
    return (
        js_template.replace("__LEFT_MAP__", map_js)
        .replace("__DEFAULT_VISUALIZATION_MODE__", default_visualization_json)
        .replace("__DEFAULT_CENTER__", json.dumps(DEN_HAAG_CENTER))
        .replace("__DEFAULT_ZOOM__", str(DEFAULT_ZOOM))
        .replace("__ALL_PROVIDERS_VALUE__", json.dumps(ALL_PROVIDERS_VALUE))
        .replace("__MAP_PROVIDER_INFO__", json.dumps(MAP_PROVIDER_INFO))
        .replace("__MAP_PROVIDER_ORDER__", json.dumps(list(MAP_PROVIDER_INFO)))
        .replace("__DEFAULT_PROVIDER__", json.dumps(ALL_PROVIDERS_VALUE))
        .replace("__VISUALIZATION_JS__", visualization_js)
        .replace("__LEFT_MAP_ID__", map_id)
        .replace("__LIGHT_MAP_TILE_URL__", LIGHT_MAP_TILE_URL)
        .replace("__DARK_MAP_TILE_URL__", DARK_MAP_TILE_URL)
        .replace("__MAP_TILE_ATTRIBUTION__", MAP_TILE_ATTRIBUTION)
        .replace("__ARTIFACTS_INDEX_PATH__", artifacts_index_path)
        .replace("__POSTCODE_GEOJSON_PATHS__", postcode_geojson_paths_json)
        .replace("__POSTCODE_CONFIGS__", postcode_configs_json)
        .replace("__POSTCODE_LEVELS__", postcode_levels_json)
        .replace("__DEFAULT_POSTCODE_LEVEL__", default_postcode_level_json)
        .replace("__HOUSE_DATA_PATH__", house_data_path)
        .replace("__DEN_HAAG_BBOX__", json.dumps(DEN_HAAG_BBOX))
    )


def main():
    ensure_output_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory with processed tables (default: output/data).",
    )
    parser.add_argument(
        "--output",
        default=str(MAPS_DIR / "den_haag.html"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Processed data directory not found yet: {data_dir}")
        print(
            "Generating the map shell anyway. It will load data at runtime if files appear."
        )

    print("Step 1: Ensuring postcode polygon boundaries are cached...")
    postcode_gdfs = {}
    for level, cfg in POSTCODE_LEVEL_CONFIG.items():
        postcode_gdfs[level] = load_area_polygons(bbox=DEN_HAAG_BBOX, cfg=cfg)

    print("\nStep 2: Ensuring house points are cached...")
    house_points = download_house_points(DEN_HAAG_BBOX, HOUSE_POINTS_CACHE)

    print("\nStep 3: Creating runtime-loading map shell...")
    m = folium.Map(
        location=DEN_HAAG_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
    )
    map_js = m.get_name()
    map_id = map_js

    default_visualization_json = json.dumps(DEFAULT_VISUALIZATION_MODE)
    visualization_options = build_visualization_options_html()
    area_level_options = build_area_level_options_html()
    visualization_js = build_visualization_js()

    root = cast(Figure, m.get_root())
    root.header.add_child(folium.Element(build_page_styles(map_id)))
    root.html.add_child(
        folium.Element(build_page_html(visualization_options, area_level_options))
    )
    root.html.add_child(
        folium.Element(
            build_custom_js(
                map_js=map_js,
                map_id=map_id,
                default_visualization_json=default_visualization_json,
                visualization_js=visualization_js,
                artifacts_index_path="../index/artifacts.json",
                postcode_geojson_paths_json=json.dumps(
                    {
                        level: _build_runtime_geojson_resource(
                            _geojson_resource_path(cfg)
                        )
                        for level, cfg in POSTCODE_LEVEL_CONFIG.items()
                    }
                ),
                postcode_configs_json=json.dumps(
                    {
                        level: {
                            "label": cfg["label"],
                            "property": cfg["property"],
                        }
                        for level, cfg in POSTCODE_LEVEL_CONFIG.items()
                    }
                ),
                postcode_levels_json=json.dumps(list(POSTCODE_LEVELS)),
                default_postcode_level_json=json.dumps(DEFAULT_POSTCODE_LEVEL),
                house_data_path="../geodata/houses_den_haag.json",
            )
        )
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nMap saved to: {args.output}")
    print("\nSummary:")
    for level in POSTCODE_LEVELS:
        print(
            f"  {POSTCODE_LEVEL_CONFIG[level]['label']} areas cached: "
            f"{len(postcode_gdfs[level])}"
        )
    print(f"  Unique residential points cached: {house_points['count']}")
    print(
        "  Data is fetched at runtime from output/index, output/data, output/geodata, "
        "and service_zone_calculation/output"
    )

    csv_path, json_path, _ = rebuild_artifact_index()
    print(f"\nArtifact index updated: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
