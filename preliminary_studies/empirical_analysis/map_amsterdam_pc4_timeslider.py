"""
Amsterdam Interactive PC4 Map with Time Slider
==============================================

Same logic as map_den_haag_pc4_timeslider.py but for Amsterdam using donkey_am.

Key differences from Den Haag:
    - Provider: donkey_am (station-only, no free floating bikes)
    - Capacity not in station_information, computed from station_status
    - Different PC4 areas and bbox

Output:
    output/maps/amsterdam_pc4_timeslider.html
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
from data_utils import (
    AMSTERDAM_BBOX,
    AMSTERDAM_CENTER,
    AMSTERDAM_PROVIDER,
    DEFAULT_DATA_ROOT,
    discover_available_dates,
    discover_available_hours,
    filter_by_bbox,
    get_station_info,
    get_station_status,
    list_tar_files,
    load_day_availability,
)
from paths import (
    GEODATA_DIR,
    MAPS_DIR,
    ensure_output_dirs,
    provider_docked_data_dir,
)

PC4_CACHE = str(GEODATA_DIR / "pc4_amsterdam.geojson")

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
    data_root: str,
    year: int,
    month: int,
    day: int,
    pc4_gdf: gpd.GeoDataFrame,
    docked_data_dir: str,
) -> dict | None:
    """Process one date's data for Amsterdam. Returns dict with hours, counts, stations."""
    date_str = f"{year}-{month:02d}-{day:02d}"
    hours = discover_available_hours(
        data_root,
        year,
        month,
        day,
        provider=AMSTERDAM_PROVIDER,
    )
    if not hours:
        print(f"  {date_str}: no data found, skipping")
        return None
    max_hour = max(hours)
    print(f"\n  Processing {date_str} (hours {hours[0]}-{max_hour})...")

    # Station info from first available hour
    files = list_tar_files(
        data_root,
        year,
        month,
        day,
        hour=hours[0],
        provider=AMSTERDAM_PROVIDER,
    )
    if not files:
        return None
    all_stations = get_station_info(files[0])
    if not all_stations:
        return None
    am_stations = filter_by_bbox(all_stations, AMSTERDAM_BBOX)

    # donkey_am doesn't have 'capacity' in station_info — compute from status
    statuses = get_station_status(files[0])
    if statuses:
        status_map = {s["station_id"]: s for s in statuses}
        for s in am_stations:
            st = status_map.get(s["station_id"], {})
            s["capacity"] = st.get("num_bikes_available", 0) + st.get(
                "num_docks_available", 0
            )
    else:
        for s in am_stations:
            s.setdefault("capacity", 0)

    am_station_ids = {s["station_id"] for s in am_stations}
    print(f"    Stations: {len(am_stations)}")

    # Spatial join stations to PC4
    station_gdf = gpd.GeoDataFrame(
        am_stations,
        geometry=[Point(s["lon"], s["lat"]) for s in am_stations],
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

    # Load docked-bike counts (hourly)
    print("    Loading docked-bike data...")
    df_avail = load_day_availability(
        data_root,
        year,
        month,
        day,
        cache_dir=docked_data_dir,
        provider=AMSTERDAM_PROVIDER,
    )
    df_hourly = df_avail.resample("h").first()

    hour_to_row = {}
    for ts in df_hourly.index:
        hour_to_row[ts.hour] = df_hourly.loc[ts]

    # Initialize counts for all PC4 areas (station only, no free bikes)
    counts = {}
    for pc4 in pc4_gdf["postcode"]:
        pc4_key = str(int(pc4))
        counts[pc4_key] = {"c": [0] * (max_hour + 1)}

    for h in hours:
        row = hour_to_row.get(h)
        if row is None:
            continue
        for sid in am_station_ids:
            if sid in df_hourly.columns:
                pc4 = station_to_pc4.get(sid)
                if pc4:
                    pc4_key = str(pc4)
                    if pc4_key in counts:
                        raw = row.get(sid, 0)
                        val = int(raw) if pd.notna(raw) else 0
                        counts[pc4_key]["c"][h] += val

    # Station data with hourly availability
    stations_js = []
    for s in am_stations:
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
                "cap": s["capacity"],
                "pc": pc4,
                "av": hourly_avail,
            }
        )

    return {
        "hours": hours,
        "maxHour": max_hour,
        "counts": counts,
        "stations": stations_js,
    }


def main():
    ensure_output_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output",
        default=str(MAPS_DIR / "amsterdam_pc4_timeslider.html"),
    )
    args = parser.parse_args()

    # Step 1: PC4 polygons
    print("Step 1: Loading PC4 polygon boundaries...")
    pc4_gdf = download_pc4_polygons(AMSTERDAM_BBOX, PC4_CACHE)

    # Step 2: Auto-discover available dates
    print("\nStep 2: Discovering available dates...")
    dates = discover_available_dates(args.data_root, provider=AMSTERDAM_PROVIDER)
    if not dates:
        print("  No donkey_am data found!")
        return
    for y, m, d in dates:
        print(f"  Found: {y}-{m:02d}-{d:02d}")

    # Step 3: Process each date
    print("\nStep 3: Processing data for each date...")
    all_date_data = {}
    docked_data_dir = str(provider_docked_data_dir(AMSTERDAM_PROVIDER))
    for year, month, day in dates:
        date_str = f"{year}-{month:02d}-{day:02d}"
        data = process_date(
            args.data_root,
            year,
            month,
            day,
            pc4_gdf,
            docked_data_dir,
        )
        if data:
            all_date_data[date_str] = data

    if not all_date_data:
        print("  No valid date data processed!")
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
    m = folium.Map(
        location=AMSTERDAM_CENTER,
        zoom_start=13,
        tiles="CartoDB positron",
    )

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

    all_data_json = json.dumps(all_date_data, separators=(",", ":"))
    dates_json = json.dumps(sorted_dates, separators=(",", ":"))

    # Legend (dynamic, updated by JS)
    legend_html = """
    <div id="legend-box" style="position: fixed; bottom: 80px; left: 30px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 13px; line-height: 1.8;">
        <b>Available Bikes</b><br>
        <span id="legend-content"></span>
        <span style="color: #e377c2;">&#9679;</span> Station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Date selector + color mode + slider
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
            return 1;
        }}

        function lerpColor(a, b, t) {{
            return [
                Math.round(a[0] + (b[0] - a[0]) * t),
                Math.round(a[1] + (b[1] - a[1]) * t),
                Math.round(a[2] + (b[2] - a[2]) * t)
            ];
        }}

        var LIGHT_BLUE = '#d0e4ff';

        function ratioToGradient(ratio) {{
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

        var stationLayer = L.layerGroup().addTo(map);
        var stationMarkers = [];

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

        document.getElementById('date-selector').addEventListener('change', function() {{
            switchDate(this.value);
        }});

        document.getElementById('hour-slider').addEventListener('input', function() {{
            updateHour(parseInt(this.value));
        }});

        document.getElementById('color-mode').addEventListener('change', function() {{
            colorMode = this.value;
            updateLegend(currentHour);
            updatePolygons(currentHour);
        }});

        geojsonLayer.eachLayer(function(layer) {{
            layer.on('mouseover', function(e) {{
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                if (selectedPC4 !== pc) layer.setStyle({{ weight: 5 }});
            }});
            layer.on('mouseout', function(e) {{
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                if (selectedPC4 !== pc) layer.setStyle({{ weight: 2.5 }});
            }});
            layer.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                var pc = parseInt(layer.feature.properties.postcode) || layer.feature.properties.postcode;
                var pcKey = String(layer.feature.properties.postcode);
                var dateData = allData[currentDate];
                var dateCounts = dateData ? dateData.counts : {{}};
                var pcData = dateCounts[pcKey] || {{c:[]}};
                var total = pcData.c[currentHour] || 0;

                if (selectedPC4 === pc) {{
                    selectedPC4 = null;
                    map.closePopup();
                }} else {{
                    selectedPC4 = pc;
                    L.popup()
                        .setLatLng(e.latlng)
                        .setContent(
                            '<b>PC4 ' + pc + '</b><br>' +
                            'Available bikes: <b>' + total + '</b>'
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

        switchDate(dates[0]);
    }});
    </script>
    """)
    m.get_root().html.add_child(custom_js)

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
