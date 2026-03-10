"""
Task 2b: Interactive PC4 Map with Time Slider (All Available Bikes)
====================================================================

Run:
    uv run python preliminary_studies/empirical_analysis/build_data_tables.py
    uv run python preliminary_studies/empirical_analysis/map_den_haag_pc4_timeslider.py

Creates an interactive folium map showing Den Haag's 4-digit postcode (PC4)
areas as polygons, colored by the total number of available Donkey bikes
(docked + dockless) in each area.

Auto-discovers all available dates from processed docked tables in `output/data`.
A date dropdown
and hour slider let the user explore any date/hour combination. The slider
adapts to the available hours for each date.

Interactions:
    - Date dropdown: switches between available dates
    - Slider: changes hour, updates everything
    - Click polygon: bold border, popup with bike count breakdown
    - Hover polygon: thicker border preview

Color scheme (matching den_haag_stations.html):
    blue   = 0 bikes
    green  = 1-3 bikes
    orange = 4-6 bikes
    red    = 7+ bikes

Output:
    output/maps/den_haag_pc4_timeslider.html
"""

import argparse
import json
import os
import sys
import urllib.request

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
from internal.paths import (
    DATA_DIR,
    GEODATA_DIR,
    MAPS_DIR,
    ensure_output_dirs,
)
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

COLOR_BLUE = "#3388ff"
COLOR_GREEN = "#2ca02c"
COLOR_ORANGE = "#ff7f0e"
COLOR_RED = "#d62728"


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
    """Process one date's processed tables. Returns dict with hours/counts/bikes/stations."""
    date_str = f"{year}-{month:02d}-{day:02d}"
    stations_df = load_station_day(data_dir, PROVIDER, year, month, day)
    if stations_df is None or stations_df.empty:
        print(f"  {date_str}: station metadata not found, skipping")
        return None

    required_station_cols = {"station_id", "name", "lat", "lon", "capacity"}
    if not required_station_cols.issubset(stations_df.columns):
        print(f"  {date_str}: station metadata columns missing, skipping")
        return None

    # Convert to dict records and apply the same Den Haag bbox filter as before.
    all_stations = stations_df[list(required_station_cols)].to_dict("records")
    dh_stations = filter_by_bbox(all_stations, DEN_HAAG_BBOX)
    if not dh_stations:
        print(f"  {date_str}: no stations in bbox, skipping")
        return None
    dh_station_ids = {str(s["station_id"]) for s in dh_stations}
    print(f"    Stations: {len(dh_stations)}")

    # Spatial join stations to PC4
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

    # Load docked-bike counts table.
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

    # Build hour-to-row lookup
    hour_to_row = {}
    for ts in df_hourly.index:
        hour_to_row[ts.hour] = df_hourly.loc[ts]

    # Load dockless-bike positions
    print("    Loading dockless-bike data...")
    bikes_df = load_dockless_day(data_dir, PROVIDER, year, month, day)
    if bikes_df is None:
        bikes_df = pd.DataFrame(columns=["timestamp", "bike_id", "lat", "lon"])

    # Initialize counts for all PC4 areas
    counts = {}
    for pc4 in pc4_gdf["postcode"]:
        pc4_key = str(int(pc4))
        counts[pc4_key] = {
            "c": [0] * (max_hour + 1),
            "s": [0] * (max_hour + 1),
            "f": [0] * (max_hour + 1),
        }

    # Fill station counts per PC4 per hour
    for h in hours:
        row = hour_to_row.get(h)
        if row is None:
            continue
        for sid in dh_station_ids:
            if sid in df_hourly.columns:
                pc4 = station_to_pc4.get(sid)
                if pc4:
                    pc4_key = str(pc4)
                    if pc4_key in counts:
                        raw = row.get(sid, 0)
                        val = int(raw) if pd.notna(raw) else 0
                        counts[pc4_key]["s"][h] += val
                        counts[pc4_key]["c"][h] += val

    # Dockless-bike processing
    bikes_by_hour = {}
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

        # Dockless-bike counts per PC4 per hour
        free_bike_counts = (
            bikes_with_pc4.groupby(["hour_idx", "postcode"])["bike_id"]
            .nunique()
            .reset_index(name="bike_count")
        )
        for _, row in free_bike_counts.iterrows():
            h = int(row["hour_idx"])
            pc4_key = str(int(row["postcode"]))
            if pc4_key in counts and h <= max_hour:
                counts[pc4_key]["f"][h] += int(row["bike_count"])
                counts[pc4_key]["c"][h] += int(row["bike_count"])

        # Bike positions per hour (first observation per bike per hour)
        hourly_bikes = (
            bikes_with_pc4.sort_values("timestamp")
            .groupby(["hour_idx", "bike_id"])
            .first()
            .reset_index()
        )
        for h in hours:
            hour_data = hourly_bikes[hourly_bikes["hour_idx"] == h]
            bikes_by_hour[str(h)] = [
                [round(r["lat"], 6), round(r["lon"], 6), int(r["postcode"])]
                for _, r in hour_data.iterrows()
                if pd.notna(r["postcode"])
            ]
            print(f"      Hour {h:02d}: {len(bikes_by_hour[str(h)])} dockless bikes")
    else:
        for h in hours:
            bikes_by_hour[str(h)] = []

    # Station data with hourly availability
    stations_js = []
    for s in dh_stations:
        sid = s["station_id"]
        pc4 = station_to_pc4.get(sid, 0)
        hourly_avail = [0] * (max_hour + 1)
        for h in hours:
            row = hour_to_row.get(h)
            if row is not None and sid in df_hourly.columns:
                raw = row.get(sid, 0)
                hourly_avail[h] = int(raw) if pd.notna(raw) else 0
        stations_js.append(
            {
                "ll": [round(s["lat"], 6), round(s["lon"], 6)],
                "n": s["name"],
                "cap": int(s["capacity"]) if pd.notna(s["capacity"]) else 0,
                "pc": pc4,
                "av": hourly_avail,
            }
        )

    return {
        "hours": hours,
        "maxHour": max_hour,
        "counts": counts,
        "bikes": bikes_by_hour,
        "stations": stations_js,
    }


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
        default=str(MAPS_DIR / "den_haag_pc4_timeslider.html"),
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Processed data directory not found: {data_dir}")
        print("Run build_data_tables.py first to generate output/data tables.")
        return

    # Step 1: PC4 polygons
    print("Step 1: Loading PC4 polygon boundaries...")
    pc4_gdf = download_pc4_polygons(DEN_HAAG_BBOX, PC4_CACHE)

    # Step 2: Auto-discover available dates from processed docked tables
    print("\nStep 2: Discovering available dates from processed data...")
    dates = discover_docked_dates(data_dir, provider=PROVIDER)
    if not dates:
        print("  No processed donkey_denHaag docked tables found!")
        return
    for y, m, d in dates:
        print(f"  Found: {y}-{m:02d}-{d:02d}")

    # Step 3: Process each date
    print("\nStep 3: Processing data for each date...")
    all_date_data = {}
    for year, month, day in dates:
        date_str = f"{year}-{month:02d}-{day:02d}"
        data = process_date(
            data_dir,
            year,
            month,
            day,
            pc4_gdf,
        )
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

    # Compute max bikes for proportional coloring modes
    global_max = 0
    for dd in all_date_data.values():
        date_max = 0
        for pc4_data in dd["counts"].values():
            for val in pc4_data["c"]:
                if val > date_max:
                    date_max = val
        dd["dateMax"] = date_max
        if date_max > global_max:
            global_max = date_max
    print(f"  Max bikes in any PC4/hour: {global_max}")

    # Step 4: Build the map
    print("\nStep 4: Creating interactive map...")
    m = folium.Map(location=DEN_HAAG_CENTER, zoom_start=13, tiles="CartoDB positron")

    # PC4 GeoJSON (geometry only, no count data embedded)
    pc4_borders = pc4_gdf[["postcode", "geometry"]].copy()
    pc4_geojson = json.loads(pc4_borders.to_json())

    geojson_layer = folium.GeoJson(
        pc4_geojson,
        style_function=lambda f: {
            "color": "black",
            "weight": 2.5,
            "fillColor": COLOR_BLUE,
            "fillOpacity": 0.6,
        },
        name="PC4 areas",
    )
    geojson_layer.add_to(m)

    geojson_js = geojson_layer.get_name()
    map_js = m.get_name()

    # Serialize all date data
    all_data_json = json.dumps(all_date_data, separators=(",", ":"))
    dates_json = json.dumps(sorted_dates, separators=(",", ":"))

    # Legend (dynamic, updated by JS)
    legend_html = """
    <div id="legend-box" style="position: fixed; bottom: 80px; left: 30px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 13px; line-height: 1.8;">
        <b>Available Bikes</b><br>
        <span id="legend-content"></span>
        <span style="color: #333;">&#9679;</span> Dockless bike<br>
        <span style="color: #e377c2;">&#9679;</span> Station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Date selector + color mode + slider HTML
    date_options = "\n".join(
        f'            <option value="{d}">{d}</option>' for d in sorted_dates
    )
    slider_html = folium.Element(f"""
    <div id="controls-box" style="position: fixed; bottom: 20px; left: 50%;
                transform: translateX(-50%); z-index: 1000;
                background: white; padding: 12px 24px; border: 2px solid grey;
                border-radius: 8px; text-align: center; font-family: sans-serif;">
        <div style="margin-bottom: 8px;">
            <label for="date-selector" style="font-size: 14px; font-weight: bold;">Date:</label>
            <select id="date-selector" style="font-size: 14px; padding: 4px 8px;
                    border: 1px solid #ccc; border-radius: 4px; cursor: pointer;">
{date_options}
            </select>
            &nbsp;&nbsp;
            <b id="hour-label" style="font-size: 16px;">00:00</b>
            &nbsp;&nbsp;
            <label for="color-mode" style="font-size: 13px;">Color:</label>
            <select id="color-mode" style="font-size: 13px; padding: 3px 6px;
                    border: 1px solid #ccc; border-radius: 4px; cursor: pointer;">
                <option value="global">Global proportional</option>
                <option value="per-date">Per-date proportional</option>
                <option value="per-hour">Per-hour proportional</option>
                <option value="gradient">Continuous gradient</option>
                <option value="fixed">Fixed (0 / 1-3 / 4-6 / 7+)</option>
            </select>
        </div>
        <input type="range" id="hour-slider" min="0" max="23" value="0"
               style="width: 400px; cursor: pointer;">
    </div>
    """)
    m.get_root().html.add_child(slider_html)

    # Custom JS
    custom_js = folium.Element(f"""
    <script>
    window.addEventListener('load', function() {{
        var map = {map_js};
        var geojsonLayer = {geojson_js};
        var allData = {all_data_json};
        var dates = {dates_json};
        var globalMax = {global_max};
        var currentDate = dates[0];
        var currentHour = 0;
        var selectedPC4 = null;
        var colorMode = 'global';

        function getHourMax(h) {{
            var dateData = allData[currentDate];
            if (!dateData) return 1;
            var dateCounts = dateData.counts;
            var mx = 0;
            for (var pc in dateCounts) {{
                var v = dateCounts[pc].c[h];
                if (v !== undefined && v > mx) mx = v;
            }}
            return mx || 1;
        }}

        function getEffectiveMax(h) {{
            if (colorMode === 'global') return globalMax || 1;
            if (colorMode === 'gradient') {{
                var dd = allData[currentDate];
                return (dd && dd.dateMax) ? dd.dateMax : 1;
            }}
            if (colorMode === 'per-date') {{
                var dd = allData[currentDate];
                return (dd && dd.dateMax) ? dd.dateMax : 1;
            }}
            if (colorMode === 'per-hour') return getHourMax(h);
            return 1; // fixed mode — not used for ratio
        }}

        // Interpolate between two [r,g,b] arrays
        function lerpColor(a, b, t) {{
            return [
                Math.round(a[0] + (b[0] - a[0]) * t),
                Math.round(a[1] + (b[1] - a[1]) * t),
                Math.round(a[2] + (b[2] - a[2]) * t)
            ];
        }}

        var LIGHT_BLUE = '#d0e4ff';

        function ratioToGradient(ratio) {{
            // blue(0) -> green(0.33) -> orange(0.66) -> red(1)
            var blue   = [51, 136, 255];
            var green  = [44, 160, 44];
            var orange = [255, 127, 14];
            var red    = [214, 39, 40];
            var rgb;
            if (ratio <= 0.33) {{
                rgb = lerpColor(blue, green, ratio / 0.33);
            }} else if (ratio <= 0.66) {{
                rgb = lerpColor(green, orange, (ratio - 0.33) / 0.33);
            }} else {{
                rgb = lerpColor(orange, red, (ratio - 0.66) / 0.34);
            }}
            return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
        }}

        function countToColor(c, h) {{
            if (colorMode === 'gradient') {{
                if (c === 0) return LIGHT_BLUE;
                var mx = getEffectiveMax(h);
                // Stretch 1-5 across the first 40% of the gradient for more
                // visual separation at low counts, then 5+ fills the rest.
                var lowCut = Math.min(5, mx);
                var ratio;
                if (c <= lowCut) {{
                    ratio = (c / lowCut) * 0.4;
                }} else {{
                    ratio = 0.4 + ((c - lowCut) / (mx - lowCut)) * 0.6;
                }}
                return ratioToGradient(Math.min(ratio, 1.0));
            }}
            if (c === 0) return '{COLOR_BLUE}';
            if (colorMode === 'fixed') {{
                if (c <= 3) return '{COLOR_GREEN}';
                if (c <= 6) return '{COLOR_ORANGE}';
                return '{COLOR_RED}';
            }}
            var ratio = Math.min(c / getEffectiveMax(h), 1.0);
            if (ratio <= 0.33) return '{COLOR_GREEN}';
            if (ratio <= 0.66) return '{COLOR_ORANGE}';
            return '{COLOR_RED}';
        }}

        function updateLegend(h) {{
            var el = document.getElementById('legend-content');
            if (colorMode === 'fixed') {{
                el.innerHTML =
                    '<span style="color:{COLOR_BLUE}">&#9632;</span> 0 bikes<br>' +
                    '<span style="color:{COLOR_GREEN}">&#9632;</span> 1 &ndash; 3<br>' +
                    '<span style="color:{COLOR_ORANGE}">&#9632;</span> 4 &ndash; 6<br>' +
                    '<span style="color:{COLOR_RED}">&#9632;</span> 7+<br>';
            }} else if (colorMode === 'gradient') {{
                var mx = getEffectiveMax(h);
                el.innerHTML =
                    '<span style="color:#d0e4ff">&#9632;</span> 0<br>' +
                    '<div style="height:14px;width:120px;border:1px solid #999;border-radius:2px;' +
                    'background:linear-gradient(to right,rgb(51,136,255),rgb(44,160,44),rgb(255,127,14),rgb(214,39,40));' +
                    'margin:4px 0;"></div>' +
                    '1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ' + mx + '<br>';
            }} else {{
                var mx = getEffectiveMax(h);
                var t1 = Math.round(mx * 0.33);
                var t2 = Math.round(mx * 0.66);
                var label = colorMode === 'global' ? 'global' :
                            colorMode === 'per-date' ? 'this date' : 'this hour';
                el.innerHTML =
                    '<span style="color:{COLOR_BLUE}">&#9632;</span> 0<br>' +
                    '<span style="color:{COLOR_GREEN}">&#9632;</span> 1 &ndash; ' + t1 + '<br>' +
                    '<span style="color:{COLOR_ORANGE}">&#9632;</span> ' + (t1+1) + ' &ndash; ' + t2 + '<br>' +
                    '<span style="color:{COLOR_RED}">&#9632;</span> ' + (t2+1) + '+ (max ' + mx + ' ' + label + ')<br>';
            }}
        }}

        // Layers for dockless bikes and stations
        var bikeLayer = L.layerGroup().addTo(map);
        var stationLayer = L.layerGroup().addTo(map);
        var bikeMarkers = [];
        var stationMarkers = [];

        function updateBikes(h) {{
            bikeLayer.clearLayers();
            bikeMarkers = [];
            var dateData = allData[currentDate];
            var bikes = (dateData && dateData.bikes[String(h)]) || [];
            for (var i = 0; i < bikes.length; i++) {{
                var isSelected = (selectedPC4 !== null && bikes[i][2] === selectedPC4);
                var marker = L.circleMarker([bikes[i][0], bikes[i][1]], {{
                    radius: isSelected ? 8 : 5,
                    color: isSelected ? '#000' : '#333',
                    fillColor: isSelected ? '#000' : '#333',
                    fillOpacity: isSelected ? 1.0 : 0.7,
                    weight: isSelected ? 2 : 1
                }});
                marker._pc4 = bikes[i][2];
                marker.addTo(bikeLayer);
                bikeMarkers.push(marker);
            }}
        }}

        function updateStations(h) {{
            stationLayer.clearLayers();
            stationMarkers = [];
            var dateData = allData[currentDate];
            if (!dateData) return;
            var stns = dateData.stations;
            for (var i = 0; i < stns.length; i++) {{
                var s = stns[i];
                var avail = (s.av[h] !== undefined) ? s.av[h] : 0;
                var isSelected = (selectedPC4 !== null && s.pc === selectedPC4);
                var marker = L.circleMarker(s.ll, {{
                    radius: isSelected ? 8 : 5,
                    color: isSelected ? '#000' : '#e377c2',
                    fillColor: '#e377c2',
                    fillOpacity: isSelected ? 1.0 : 0.8,
                    weight: isSelected ? 2.5 : 1.5
                }});
                marker.bindTooltip(s.n + '<br>Available: ' + avail + ' / ' + s.cap);
                marker._pc4 = s.pc;
                marker.addTo(stationLayer);
                stationMarkers.push(marker);
            }}
        }}

        function getWeight(pc) {{
            if (selectedPC4 !== null && pc === selectedPC4) return 5;
            return 2.5;
        }}

        function updatePolygons(h) {{
            var dateData = allData[currentDate];
            var dateCounts = dateData ? dateData.counts : {{}};
            geojsonLayer.eachLayer(function(layer) {{
                var pc = layer.feature.properties.postcode;
                var pcKey = String(pc);
                var pcData = dateCounts[pcKey];
                var count = (pcData && pcData.c[h] !== undefined) ? pcData.c[h] : 0;
                layer.setStyle({{
                    fillColor: countToColor(count, h),
                    fillOpacity: 0.6,
                    weight: getWeight(parseInt(pc) || pc),
                    color: 'black'
                }});
            }});
        }}

        function highlightPC4(pc4) {{
            for (var i = 0; i < bikeMarkers.length; i++) {{
                var m = bikeMarkers[i];
                if (m._pc4 === pc4) {{
                    m.setRadius(6);
                    m.setStyle({{ color: '#000', fillColor: '#000', fillOpacity: 1.0, weight: 2 }});
                }} else {{
                    m.setRadius(3);
                    m.setStyle({{ color: '#333', fillColor: '#333', fillOpacity: 0.7, weight: 1 }});
                }}
            }}
            for (var i = 0; i < stationMarkers.length; i++) {{
                var m = stationMarkers[i];
                if (m._pc4 === pc4) {{
                    m.setRadius(8);
                    m.setStyle({{ color: '#000', weight: 2.5, fillOpacity: 1.0 }});
                }} else {{
                    m.setRadius(5);
                    m.setStyle({{ color: '#e377c2', weight: 1.5, fillOpacity: 0.8 }});
                }}
            }}
        }}

        function updateHour(h) {{
            currentHour = h;
            document.getElementById('hour-label').textContent =
                String(h).padStart(2, '0') + ':00';
            updateLegend(h);
            updatePolygons(h);
            updateBikes(h);
            updateStations(h);
        }}

        function switchDate(dateStr) {{
            currentDate = dateStr;
            selectedPC4 = null;
            map.closePopup();

            var dateData = allData[dateStr];
            var slider = document.getElementById('hour-slider');
            if (dateData) {{
                var hrs = dateData.hours;
                slider.min = hrs[0];
                slider.max = dateData.maxHour;
                slider.value = hrs[0];
                updateHour(hrs[0]);
            }}
        }}

        // Date selector
        document.getElementById('date-selector').addEventListener('change', function() {{
            switchDate(this.value);
        }});

        // Hour slider
        document.getElementById('hour-slider').addEventListener('input', function() {{
            updateHour(parseInt(this.value));
        }});

        // Color mode selector
        document.getElementById('color-mode').addEventListener('change', function() {{
            colorMode = this.value;
            updateLegend(currentHour);
            updatePolygons(currentHour);
        }});

        // Hover + click on polygons
        geojsonLayer.eachLayer(function(layer) {{
            layer.on('mouseover', function(e) {{
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                if (selectedPC4 !== pc) {{
                    layer.setStyle({{ weight: 5 }});
                }}
            }});
            layer.on('mouseout', function(e) {{
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                if (selectedPC4 !== pc) {{
                    layer.setStyle({{ weight: 2.5 }});
                }}
            }});

            layer.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                var pcKey = String(layer.feature.properties.postcode);
                var dateData = allData[currentDate];
                var dateCounts = dateData ? dateData.counts : {{}};
                var pcData = dateCounts[pcKey] || {{c:[], s:[], f:[]}};
                var total = pcData.c[currentHour] || 0;
                var fromStations = pcData.s[currentHour] || 0;
                var fromFree = pcData.f[currentHour] || 0;

                if (selectedPC4 === pc) {{
                    selectedPC4 = null;
                    map.closePopup();
                }} else {{
                    selectedPC4 = pc;
                    L.popup()
                        .setLatLng(e.latlng)
                        .setContent(
                            '<b>PC4 ' + pc + '</b><br>' +
                            'Available bikes: <b>' + total + '</b><br>' +
                            '&nbsp;&nbsp;Docked: ' + fromStations + '<br>' +
                            '&nbsp;&nbsp;Dockless: ' + fromFree
                        )
                        .openOn(map);
                }}

                updatePolygons(currentHour);
                highlightPC4(selectedPC4 !== null ? selectedPC4 : -1);
            }});
        }});

        map.on('click', function() {{
            selectedPC4 = null;
            updatePolygons(currentHour);
            highlightPC4(-1);
        }});

        // Initialize with first date
        switchDate(dates[0]);
    }});
    </script>
    """)
    m.get_root().html.add_child(custom_js)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nMap saved to: {args.output}")
    print("\nSummary:")
    print(f"  PC4 areas: {len(pc4_geojson['features'])}")
    print(f"  Dates available: {sorted_dates}")
    for d in sorted_dates:
        dd = all_date_data[d]
        print(
            f"    {d}: hours {dd['hours'][0]}-{dd['maxHour']}, "
            f"{len(dd['stations'])} stations"
        )

    csv_path, json_path, _ = rebuild_artifact_index()
    print(f"\nArtifact index updated: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
