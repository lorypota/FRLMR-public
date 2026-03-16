"""
Task 2b: Interactive PC4 Map with Time Slider (All Available Bikes)
====================================================================

Run (sequentially):
    uv run python preliminary_studies/empirical_analysis/build_data_tables.py
    uv run python preliminary_studies/empirical_analysis/map_den_haag_pc4.py

Creates an interactive Folium map showing Den Haag's 4-digit postcode (PC4)
areas and bike availability over time with multiple visualization modes.

Output:
    output/maps/den_haag_pc4.html
"""

import argparse
import json
import os
import sys
import urllib.request
from textwrap import dedent

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_index import rebuild_artifact_index
from internal.data_utils import (
    DEN_HAAG_BBOX,
    DEN_HAAG_CENTER,
    PROVIDER,
    filter_by_bbox,
)
from internal.paths import DATA_DIR, GEODATA_DIR, MAPS_DIR, ensure_output_dirs
from internal.pc4_visualizations import (
    DEFAULT_VISUALIZATION_MODE,
    build_visualization_js,
    build_visualization_options_html,
)
from internal.pc4_visualizations.hotspot import build_hourly_hotspot_data
from internal.processed_data_utils import (
    discover_docked_dates,
    load_docked_day,
    load_dockless_day,
    load_station_day,
)

PC4_CACHE = str(GEODATA_DIR / "pc4_den_haag.geojson")

PDOK_URL = (
    "https://api.pdok.nl/cbs/postcode4/ogc/v1/collections/postcode4/items"
    "?f=json&limit=500"
    "&bbox={lon_min},{lat_min},{lon_max},{lat_max}"
    "&jaarcode=2024"
)

LIGHT_MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
DARK_MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
MAP_TILE_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)
DEFAULT_ZOOM = 13


def download_pc4_polygons(bbox: dict, cache_path: str) -> gpd.GeoDataFrame:
    """Download PC4 polygons from PDOK CBS API, with local caching."""
    if os.path.exists(cache_path):
        print(f"  Loading cached PC4 polygons from {cache_path}")
        return gpd.read_file(cache_path)

    url = PDOK_URL.format(**bbox)
    print("  Downloading PC4 polygons from PDOK...")
    with urllib.request.urlopen(url) as resp:
        geojson_data = json.loads(resp.read().decode("utf-8"))

    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")
    print(f"  Downloaded {len(gdf)} PC4 areas")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    print(f"  Cached to {cache_path}")
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

    print("    Loading dockless-bike data...")
    bikes_df = load_dockless_day(data_dir, PROVIDER, year, month, day)
    if bikes_df is None:
        bikes_df = pd.DataFrame(columns=["timestamp", "bike_id", "lat", "lon"])

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

    bikes_by_hour: dict[str, list[list[float]]] = {}
    if not bikes_df.empty and {"timestamp", "bike_id", "lat", "lon"}.issubset(
        bikes_df.columns
    ):
        bikes_df["timestamp"] = pd.to_datetime(bikes_df["timestamp"])
        bikes_df["lat"] = pd.to_numeric(bikes_df["lat"], errors="coerce")
        bikes_df["lon"] = pd.to_numeric(bikes_df["lon"], errors="coerce")
        bikes_df = bikes_df.dropna(subset=["timestamp", "bike_id", "lat", "lon"])
        bikes_df["hour_idx"] = bikes_df["timestamp"].dt.hour

        bike_gdf = gpd.GeoDataFrame(
            bikes_df,
            geometry=[
                Point(lon, lat)
                for lon, lat in zip(bikes_df["lon"], bikes_df["lat"], strict=True)
            ],
            crs="EPSG:4326",
        )
        bikes_with_pc4 = gpd.sjoin(
            bike_gdf,
            pc4_gdf[["postcode", "geometry"]],
            how="left",
            predicate="within",
        )

        free_bike_counts = (
            bikes_with_pc4.groupby(["hour_idx", "postcode"])["bike_id"]
            .nunique()
            .reset_index(name="bike_count")
        )
        for _, row in free_bike_counts.iterrows():
            hour = int(row["hour_idx"])
            if pd.isna(row["postcode"]) or hour > max_hour:
                continue
            pc4_key = str(int(row["postcode"]))
            if pc4_key in counts:
                counts[pc4_key]["f"][hour] += int(row["bike_count"])
                counts[pc4_key]["c"][hour] += int(row["bike_count"])

        hourly_bikes = (
            bikes_with_pc4.sort_values("timestamp")
            .groupby(["hour_idx", "bike_id"])
            .first()
            .reset_index()
        )
        for hour in hours:
            hour_data = hourly_bikes[hourly_bikes["hour_idx"] == hour]
            bikes_by_hour[str(hour)] = [
                [round(r["lat"], 6), round(r["lon"], 6), int(r["postcode"])]
                for _, r in hour_data.iterrows()
                if pd.notna(r["postcode"])
            ]
            print(
                f"      Hour {hour:02d}: {len(bikes_by_hour[str(hour)])} dockless bikes"
            )
    else:
        for hour in hours:
            bikes_by_hour[str(hour)] = []

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

    hotspot = build_hourly_hotspot_data(
        stations_js,
        bikes_by_hour,
        hours,
        DEN_HAAG_BBOX,
    )

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
    date_slider_max: int,
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
                        <label for="{panel_id}-date-slider">Date</label>
                        <span class="date-readout" id="{panel_id}-date-label"></span>
                    </div>
                    <input type="range" id="{panel_id}-date-slider" min="0" max="{date_slider_max}" value="0" step="1">
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
        .panel-controls input[type="range"] {{
            accent-color: #1e6bb8;
        }}

        .panel-controls select {{
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

        .legend-symbol-dockless {{
            color: #333;
        }}

        .legend-scale-zero {{
            color: #3388ff;
        }}

        .legend-scale-low {{
            color: #2ca02c;
        }}

        .legend-scale-medium {{
            color: #ff7f0e;
        }}

        .legend-scale-high {{
            color: #d62728;
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

        .legend-section {{
            border-top: 1px solid #e1e7ec;
            padding-top: 8px;
            margin-top: 8px;
        }}

        #legend-right-section {{
            display: none;
        }}

        body.compare-on #legend-right-section {{
            display: block;
        }}

        body.compare-on.legend-unified #legend-right-section {{
            display: none;
        }}

        body.compare-on.legend-unified .legend-panel-title {{
            display: none;
        }}

        .legend-unified-grid {{
            display: grid;
            gap: 10px;
        }}

        .legend-unified-pane {{
            padding-top: 2px;
        }}

        .legend-unified-label {{
            margin-bottom: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #627181;
        }}

        .legend-panel-title {{
            margin-bottom: 4px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #627181;
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

        body.theme-dark .panel-heading-label,
        body.theme-dark .hotspot-size-label,
        body.theme-dark .legend-static,
        body.theme-dark .legend-unified-label,
        body.theme-dark .legend-panel-title,
        body.theme-dark .legend-toggle {{
            color: #c3d0dc;
        }}

        body.theme-dark .control-inline label,
        body.theme-dark .legend-title,
        body.theme-dark .legend-hotspot-control label,
        body.theme-dark .date-readout,
        body.theme-dark .hour-readout {{
            color: #edf4fb;
        }}

        body.theme-dark .panel-controls select {{
            background: #13202c;
            color: #edf4fb;
            border-color: #38506a;
        }}

        body.theme-dark .panel-controls select,
        body.theme-dark .panel-controls input[type="range"],
        body.theme-dark .legend-hotspot-control input[type="range"] {{
            accent-color: #66b7ff;
        }}

        body.theme-dark .legend-hotspot-control,
        body.theme-dark .compare-sync-options,
        body.theme-dark .legend-section {{
            border-color: #314253;
        }}

        body.theme-dark .legend-symbol-dockless {{
            color: #d0dae4;
        }}

        body.theme-dark .legend-scale-zero {{
            color: #58a6ff;
        }}

        body.theme-dark .legend-scale-low {{
            color: #5ee38b;
        }}

        body.theme-dark .legend-scale-medium {{
            color: #ffb454;
        }}

        body.theme-dark .legend-scale-high {{
            color: #ff7373;
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
        </style>
        """
    ).strip()


def build_page_html(
    date_slider_max: int,
    visualization_options: str,
) -> str:
    """Return the compare-mode page chrome."""
    left_controls = build_panel_controls_html(
        "left",
        date_slider_max,
        visualization_options,
    )
    right_controls = build_panel_controls_html(
        "right",
        date_slider_max,
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
                <span class="legend-symbol-dockless">&#9679;</span> Dockless bike<br>
                Station bikes stored:<br>
                <span class="legend-scale-zero">&#9679;</span> 0<br>
                <span class="legend-scale-low">&#9679;</span> 1 &ndash; 3<br>
                <span class="legend-scale-medium">&#9679;</span> 4 &ndash; 6<br>
                <span class="legend-scale-high">&#9679;</span> 7+
            </div>
            <div class="legend-section" id="legend-left-section">
                <div class="legend-panel-title" id="legend-left-title">Current map</div>
                <div id="legend-left-content"></div>
            </div>
            <div class="legend-section" id="legend-right-section">
                <div class="legend-panel-title" id="legend-right-title">Right map</div>
                <div id="legend-right-content"></div>
            </div>
        </div>
        """
    ).strip()


def build_custom_js(
    *,
    map_js: str,
    map_id: str,
    all_data_json: str,
    dates_json: str,
    global_max: int,
    pc4_geojson_json: str,
    default_visualization_json: str,
    visualization_js: str,
) -> str:
    """Build the client-side compare-mode behavior."""
    js_template = """
    <script>
    window.addEventListener('load', function() {
        document.body.classList.add('compare-off');
        document.body.classList.add('theme-light');

        var leftMap = __LEFT_MAP__;
        var allData = __ALL_DATA__;
        var dates = __DATES__;
        var globalMax = __GLOBAL_MAX__;
        var pc4Geojson = __PC4_GEOJSON__;
        var defaultVisualizationMode = __DEFAULT_VISUALIZATION_MODE__;
        var defaultCenter = __DEFAULT_CENTER__;
        var defaultZoom = __DEFAULT_ZOOM__;
        var compareEnabled = false;
        var selectedPC4 = null;
        var viewportSyncInProgress = false;
        var panelStateSyncInProgress = false;
        var rightPanelInitialized = false;
        var panels = null;

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
        var legendLeftTitle = document.getElementById('legend-left-title');
        var legendRightTitle = document.getElementById('legend-right-title');
        var themeStorageKey = 'fairmss-den-haag-pc4-theme';
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

        function getDocklessMarkerStyle(isSelected, muted) {
            var selectedColor = themeColor('selectedMarker');
            var baseColor = themeColor('docklessMarker');
            return {
                color: isSelected ? selectedColor : baseColor,
                fillColor: isSelected ? selectedColor : baseColor,
                fillOpacity: isSelected ? 0.95 : (muted ? 0.25 : 0.7),
                weight: isSelected ? 2 : 1
            };
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
                dateSlider: document.getElementById(panelId + '-date-slider'),
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

        function getDateIndex(dateKey) {
            var index = dates.indexOf(dateKey);
            return index === -1 ? 0 : index;
        }

        function getDateAtIndex(index) {
            if (dates.length === 0) return null;
            var boundedIndex = Math.max(0, Math.min(index, dates.length - 1));
            return dates[boundedIndex];
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

        function getPolygonWeight(panel, pc4) {
            if (selectedPC4 !== null && pc4 === selectedPC4) return 5;
            return isHotspotMode(panel.visualizationMode) ? 1.5 : 2.5;
        }

        function getHoverWeight(panel) {
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
            closePanelPopup(panels.left);
            closePanelPopup(panels.right);
        }

        function updateLegendTitles() {
            legendLeftTitle.textContent = compareEnabled ? 'Left map' : 'Current map';
            legendRightTitle.textContent = 'Right map';
        }

        function updatePanelHeadings() {
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
                '<label for="legend-unified-hotspot-size-slider">Hotspot size</label>' +
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
            var leftBaseLegend = panels.left.legendBaseHtml || '';
            var rightBaseLegend = panels.right.legendBaseHtml || '';
            var leftLegend = getPanelLegendMarkup(panels.left);
            var rightLegend = getPanelLegendMarkup(panels.right);
            var unified = compareEnabled &&
                panels.left.visualizationMode === panels.right.visualizationMode;
            document.body.classList.toggle('legend-unified', unified);
            if (unified) {
                legendLeftTitle.textContent = 'Shared legend';
                if (isHotspotMode(panels.left.visualizationMode)) {
                    panels.right.hotspotRadiusScale = panels.left.hotspotRadiusScale;
                }
                if (leftBaseLegend === rightBaseLegend) {
                    panels.left.legendEl.innerHTML = leftBaseLegend;
                    if (isHotspotMode(panels.left.visualizationMode)) {
                        panels.left.legendEl.innerHTML += buildUnifiedHotspotControl();
                    }
                } else {
                    panels.left.legendEl.innerHTML =
                        '<div class="legend-unified-grid">' +
                        '<div class="legend-unified-pane">' +
                        '<div class="legend-unified-label">Left map</div>' +
                        leftLegend +
                        '</div>' +
                        '<div class="legend-unified-pane">' +
                        '<div class="legend-unified-label">Right map</div>' +
                        rightBaseLegend +
                        '</div>' +
                        '</div>';
                    if (isHotspotMode(panels.left.visualizationMode)) {
                        panels.left.legendEl.innerHTML += buildUnifiedHotspotControl();
                    }
                }
                panels.right.legendEl.innerHTML = '';
                bindUnifiedHotspotControl();
            } else {
                updateLegendTitles();
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
                    targetPanel.currentHour
                );
                shouldUpdate = true;
            }
            if (options.hour && syncSettings.time) {
                targetPanel.currentHour = normalizeHour(
                    getDateData(targetPanel.currentDate),
                    sourcePanel.currentHour
                );
                shouldUpdate = true;
            }
            if (options.visualization && syncSettings.visualization) {
                targetPanel.visualizationMode = sourcePanel.visualizationMode;
                shouldUpdate = true;
            }
            if (shouldUpdate) {
                targetPanel.syncControlsFromState();
                targetPanel.renderAll();
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

        function getPc4Stats(panel, pc4) {
            var dateData = getDateData(panel.currentDate);
            var dateCounts = dateData ? dateData.counts : {};
            var pcData = dateCounts[String(pc4)] || { c: [], s: [], f: [] };
            return {
                total: pcData.c[panel.currentHour] || 0,
                docked: pcData.s[panel.currentHour] || 0,
                dockless: pcData.f[panel.currentHour] || 0
            };
        }

        function openPanelPopup(panel, pc4, latlng) {
            var stats = getPc4Stats(panel, pc4);
            closePanelPopup(panel);
            panel.activePopup = L.popup()
                .setLatLng(latlng)
                .setContent(
                    '<b>PC4 ' + escapeHtml(pc4) + '</b><br>' +
                    'Available bikes: <b>' + stats.total + '</b><br>' +
                    '&nbsp;&nbsp;Docked: ' + stats.docked + '<br>' +
                    '&nbsp;&nbsp;Dockless: ' + stats.dockless
                )
                .openOn(panel.map);
        }

        function clearSelection() {
            selectedPC4 = null;
            closeAllPopups();
            panels.left.renderPolygons();
            panels.right.renderPolygons();
            panels.left.applySelection(null);
            panels.right.applySelection(null);
        }

        function setSelectedPC4(pc4, sourcePanel, latlng) {
            if (selectedPC4 === pc4) {
                clearSelection();
                return;
            }
            selectedPC4 = pc4;
            closeAllPopups();
            panels.left.renderPolygons();
            panels.right.renderPolygons();
            panels.left.applySelection(pc4);
            panels.right.applySelection(pc4);
            if (sourcePanel && latlng) {
                openPanelPopup(sourcePanel, pc4, latlng);
            }
        }

        function buildPolygonLayer(panel) {
            return L.geoJSON(pc4Geojson, {
                style: function(feature) {
                    var pc = parseInt(feature.properties.postcode, 10) ||
                             feature.properties.postcode;
                    return {
                        color: getPolygonStrokeColor(panel.visualizationMode),
                        weight: getPolygonWeight(panel, pc),
                        fillColor: themeColor('scaleBlue'),
                        fillOpacity: getPolygonFillOpacity(panel.visualizationMode)
                    };
                },
                onEachFeature: function(feature, layer) {
                    layer.on('mouseover', function() {
                        var pc = parseInt(feature.properties.postcode, 10) ||
                                 feature.properties.postcode;
                        if (selectedPC4 !== pc) {
                            layer.setStyle({ weight: getHoverWeight(panel) });
                        }
                    });

                    layer.on('mouseout', function() {
                        var pc = parseInt(feature.properties.postcode, 10) ||
                                 feature.properties.postcode;
                        if (selectedPC4 !== pc) {
                            layer.setStyle({ weight: getPolygonWeight(panel, pc) });
                        }
                    });

                    layer.on('click', function(e) {
                        L.DomEvent.stopPropagation(e);
                        var pc = parseInt(feature.properties.postcode, 10) ||
                                 feature.properties.postcode;
                        setSelectedPC4(pc, panel, e.latlng);
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
                hotspotLayer: L.layerGroup().addTo(mapInstance),
                bikeLayer: L.layerGroup().addTo(mapInstance),
                stationLayer: L.layerGroup().addTo(mapInstance),
                bikeMarkers: [],
                stationMarkers: [],
                currentDate: dates[0],
                currentHour: 0,
                visualizationMode: defaultVisualizationMode,
                hotspotRadiusScale: 1.0,
                activePopup: null,
                legendHtml: '',
                legendBaseHtml: '',
                legendControlHtml: '',
                setDate: function(dateStr) {
                    if (!allData[dateStr]) return;
                    this.currentDate = dateStr;
                    this.currentHour = normalizeHour(allData[dateStr], this.currentHour);
                    closeAllPopups();
                    this.syncControlsFromState();
                    this.renderAll();
                    syncPartnerPanel(this, { date: true, hour: true });
                },
                setHour: function(hour) {
                    this.currentHour = hour;
                    closeAllPopups();
                    this.syncControlsFromState();
                    this.renderAll();
                    syncPartnerPanel(this, { hour: true });
                },
                setVisualization: function(mode) {
                    this.visualizationMode = mode;
                    closeAllPopups();
                    this.syncControlsFromState();
                    this.renderAll();
                    syncPartnerPanel(this, { visualization: true });
                },
                syncControlsFromState: function() {
                    var dateData = getDateData(this.currentDate);
                    if (!dateData) return;
                    this.controls.dateLabel.textContent = this.currentDate;
                    this.controls.dateSlider.max = Math.max(dates.length - 1, 0);
                    this.controls.dateSlider.value = getDateIndex(this.currentDate);
                    this.controls.visualization.value = this.visualizationMode;
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
                        legendControlHtml = buildHotspotControlHtml(this, 'Hotspot size');
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
                    var dateCounts = dateData ? dateData.counts : {};
                    this.geojsonLayer.eachLayer(function(layer) {
                        var pc = layer.feature.properties.postcode;
                        var pcKey = String(pc);
                        var pcValue = parseInt(pc, 10) || pc;
                        var pcData = dateCounts[pcKey];
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
                        style.weight = getPolygonWeight(panelRef, pcValue);
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
                    var dockless = hotspotData.dockless || [];
                    var stations = hotspotData.stations || [];
                    var stationMax = hotspotData.stationMax || 0;

                    for (var i = 0; i < dockless.length; i++) {
                        L.circle(dockless[i], {
                            radius: DOCKLESS_RADIUS_METERS * this.hotspotRadiusScale,
                            stroke: false,
                            fillColor: themeColor('hotspotDockless'),
                            fillOpacity: 0.15,
                            interactive: false
                        }).addTo(this.hotspotLayer);
                    }

                    for (var j = 0; j < stations.length; j++) {
                        var station = stations[j];
                        var avail = station[2];
                        L.circle([station[0], station[1]], {
                            radius: hotspotStationRadius(
                                avail,
                                stationMax,
                                this.hotspotRadiusScale
                            ),
                            stroke: false,
                            fillColor: hotspotFillColor(avail, stationMax),
                            fillOpacity: hotspotStationOpacity(avail, stationMax),
                            interactive: false
                        }).addTo(this.hotspotLayer);
                    }
                },
                renderBikes: function() {
                    this.bikeLayer.clearLayers();
                    this.bikeMarkers = [];
                    if (!shouldShowPointMarkers(this)) return;

                    var dateData = getDateData(this.currentDate);
                    var bikes = (dateData && dateData.bikes[String(this.currentHour)]) || [];
                    var muted = isHotspotMode(this.visualizationMode);
                    for (var i = 0; i < bikes.length; i++) {
                        var isSelected = (selectedPC4 !== null && bikes[i][2] === selectedPC4);
                        var marker = L.circleMarker([bikes[i][0], bikes[i][1]], {
                            radius: isSelected ? POINT_MARKER_RADIUS_SELECTED : POINT_MARKER_RADIUS,
                            color: getDocklessMarkerStyle(isSelected, muted).color,
                            fillColor: getDocklessMarkerStyle(isSelected, muted).fillColor,
                            fillOpacity: getDocklessMarkerStyle(isSelected, muted).fillOpacity,
                            weight: getDocklessMarkerStyle(isSelected, muted).weight
                        });
                        marker._pc4 = bikes[i][2];
                        marker.addTo(this.bikeLayer);
                        this.bikeMarkers.push(marker);
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
                        var isSelected = (
                            selectedPC4 !== null && station.pc === selectedPC4
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
                            '<br>Available: ' + avail + ' / ' + station.cap
                        );
                        marker._pc4 = station.pc;
                        marker._avail = avail;
                        marker.addTo(this.stationLayer);
                        this.stationMarkers.push(marker);
                    }
                },
                applySelection: function(pc4) {
                    var muted = isHotspotMode(this.visualizationMode);
                    for (var i = 0; i < this.bikeMarkers.length; i++) {
                        var bikeMarker = this.bikeMarkers[i];
                        if (pc4 !== null && bikeMarker._pc4 === pc4) {
                            bikeMarker.setRadius(POINT_MARKER_RADIUS_SELECTED);
                            bikeMarker.setStyle(getDocklessMarkerStyle(true, muted));
                        } else {
                            bikeMarker.setRadius(POINT_MARKER_RADIUS);
                            bikeMarker.setStyle(getDocklessMarkerStyle(false, muted));
                        }
                    }
                    for (var j = 0; j < this.stationMarkers.length; j++) {
                        var stationMarker = this.stationMarkers[j];
                        var stationStyle;
                        if (pc4 !== null && stationMarker._pc4 === pc4) {
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
                    this.renderLegend();
                    this.renderPolygons();
                    this.renderHotspotLayer();
                    this.renderBikes();
                    this.renderStations();
                    this.applySelection(selectedPC4);
                }
            };

            panel.geojsonLayer = buildPolygonLayer(panel);

            panel.controls.dateSlider.addEventListener('input', function() {
                var dateValue = getDateAtIndex(parseInt(this.value, 10));
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
                syncViewportFrom(panel);
            });

            panel.map.on('zoomend', function() {
                panel.renderBikes();
                panel.renderStations();
                panel.applySelection(selectedPC4);
            });

            L.DomEvent.disableClickPropagation(panel.controls.container);
            enableDragging(
                panel.controls.container,
                panel.controls.dragHandle,
                document.getElementById('panel-' + panelId)
            );

            return panel;
        }

        var panels = {
            left: createPanel('left', leftMap),
            right: createPanel('right', rightMap)
        };

        function setCompareMode(enabled) {
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
                panels.right.renderAll();
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
            panelStateSyncInProgress = true;
            if (options.visualization) {
                panels.right.visualizationMode = panels.left.visualizationMode;
            }
            if (options.date) {
                panels.right.currentDate = panels.left.currentDate;
                panels.right.currentHour = normalizeHour(
                    getDateData(panels.right.currentDate),
                    panels.right.currentHour
                );
            }
            if (options.time) {
                panels.right.currentHour = normalizeHour(
                    getDateData(panels.right.currentDate),
                    panels.left.currentHour
                );
            }
            panels.right.syncControlsFromState();
            panels.right.renderAll();
            panelStateSyncInProgress = false;
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

        L.DomEvent.disableClickPropagation(document.getElementById('legend-box'));
        enableDragging(
            document.getElementById('legend-box'),
            document.getElementById('legend-drag-handle'),
            window
        );

        panels.left.currentDate = dates[0];
        panels.right.currentDate = dates[0];
        panels.left.currentHour = normalizeHour(getDateData(dates[0]), 0);
        panels.right.currentHour = panels.left.currentHour;
        updatePanelHeadings();
        panels.left.syncControlsFromState();
        panels.right.syncControlsFromState();
        panels.left.renderAll();
        panels.right.renderAll();
        updateSyncSettings();
        updateLegendLayout();
    });
    </script>
    """
    return (
        js_template.replace("__LEFT_MAP__", map_js)
        .replace("__ALL_DATA__", all_data_json)
        .replace("__DATES__", dates_json)
        .replace("__GLOBAL_MAX__", str(global_max))
        .replace("__PC4_GEOJSON__", pc4_geojson_json)
        .replace("__DEFAULT_VISUALIZATION_MODE__", default_visualization_json)
        .replace("__DEFAULT_CENTER__", json.dumps(DEN_HAAG_CENTER))
        .replace("__DEFAULT_ZOOM__", str(DEFAULT_ZOOM))
        .replace("__VISUALIZATION_JS__", visualization_js)
        .replace("__LEFT_MAP_ID__", map_id)
        .replace("__LIGHT_MAP_TILE_URL__", LIGHT_MAP_TILE_URL)
        .replace("__DARK_MAP_TILE_URL__", DARK_MAP_TILE_URL)
        .replace("__MAP_TILE_ATTRIBUTION__", MAP_TILE_ATTRIBUTION)
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
        default=str(MAPS_DIR / "den_haag_pc4.html"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Processed data directory not found: {data_dir}")
        print("Run build_data_tables.py first to generate output/data tables.")
        return

    print("Step 1: Loading PC4 polygon boundaries...")
    pc4_gdf = download_pc4_polygons(DEN_HAAG_BBOX, PC4_CACHE)

    print("\nStep 2: Discovering available dates from processed data...")
    dates = discover_docked_dates(data_dir, provider=PROVIDER)
    if not dates:
        print("  No processed donkey_denHaag docked tables found!")
        return
    for year, month, day in dates:
        print(f"  Found: {year}-{month:02d}-{day:02d}")

    print("\nStep 3: Processing data for each date...")
    all_date_data = {}
    for year, month, day in dates:
        date_str = f"{year}-{month:02d}-{day:02d}"
        data = process_date(data_dir, year, month, day, pc4_gdf)
        if data:
            all_date_data[date_str] = data

    if not all_date_data:
        print("  No valid date data processed!")
        print(
            "  Ensure station tables exist under output/data/stations. "
            "Run build_data_tables.py if needed."
        )
        return

    sorted_dates = sorted(all_date_data.keys())
    print(f"\n  Processed {len(sorted_dates)} dates: {sorted_dates}")

    global_max = 0
    for date_data in all_date_data.values():
        date_max = 0
        for pc4_data in date_data["counts"].values():
            for value in pc4_data["c"]:
                if value > date_max:
                    date_max = value
        date_data["dateMax"] = date_max
        global_max = max(global_max, date_max)
    print(f"  Max bikes in any PC4/hour: {global_max}")

    print("\nStep 4: Creating interactive map...")
    m = folium.Map(
        location=DEN_HAAG_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
    )
    map_js = m.get_name()
    map_id = map_js

    pc4_borders = pc4_gdf[["postcode", "geometry"]].copy()
    pc4_geojson = json.loads(pc4_borders.to_json())

    all_data_json = json.dumps(all_date_data, separators=(",", ":"))
    dates_json = json.dumps(sorted_dates, separators=(",", ":"))
    pc4_geojson_json = json.dumps(pc4_geojson, separators=(",", ":"))
    default_visualization_json = json.dumps(DEFAULT_VISUALIZATION_MODE)
    visualization_options = build_visualization_options_html()
    visualization_js = build_visualization_js()
    date_slider_max = max(len(sorted_dates) - 1, 0)

    m.get_root().header.add_child(folium.Element(build_page_styles(map_id)))
    m.get_root().html.add_child(
        folium.Element(build_page_html(date_slider_max, visualization_options))
    )
    m.get_root().html.add_child(
        folium.Element(
            build_custom_js(
                map_js=map_js,
                map_id=map_id,
                all_data_json=all_data_json,
                dates_json=dates_json,
                global_max=global_max,
                pc4_geojson_json=pc4_geojson_json,
                default_visualization_json=default_visualization_json,
                visualization_js=visualization_js,
            )
        )
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nMap saved to: {args.output}")
    print("\nSummary:")
    print(f"  PC4 areas: {len(pc4_geojson['features'])}")
    print(f"  Dates available: {sorted_dates}")
    for date in sorted_dates:
        date_data = all_date_data[date]
        print(
            f"    {date}: hours {date_data['hours'][0]}-{date_data['maxHour']}, "
            f"{len(date_data['stations'])} stations"
        )

    csv_path, json_path, _ = rebuild_artifact_index()
    print(f"\nArtifact index updated: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
