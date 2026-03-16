"""
Task 2: Map Den Haag Donkey Stations
=====================================

Usage:
    uv run preliminary_studies/empirical_analysis/map_den_haag_stations.py

Loads processed station metadata from output/data/stations and filters stations
within Den Haag's bounding box (lat 52.03-52.12, lon 4.22-4.38). It then
plots an interactive folium map, color-coded by capacity.

Output:
    output/maps/den_haag_stations.html
"""

import argparse
import os
import sys
from datetime import datetime

import folium
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_index import rebuild_artifact_index
from internal.data_utils import (
    DEN_HAAG_BBOX,
    DEN_HAAG_CENTER,
    filter_by_bbox,
)
from internal.paths import DATA_DIR, MAPS_DIR, ensure_output_dirs
from internal.processed_data_utils import (
    discover_station_dates,
    latest_date,
    load_station_day,
)


def capacity_color(cap: int) -> str:
    """Map station capacity to a marker color."""
    if cap <= 3:
        return "blue"
    elif cap <= 6:
        return "green"
    elif cap <= 12:
        return "orange"
    else:
        return "red"


def main():
    ensure_output_dirs()

    parser = argparse.ArgumentParser(description="Map Den Haag Donkey stations")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory with processed tables (default: output/data).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to map in YYYY-MM-DD format. Default: latest available.",
    )
    parser.add_argument("--output", default=str(MAPS_DIR / "den_haag_stations.html"))
    args = parser.parse_args()

    station_dates = discover_station_dates(args.data_dir, provider="donkey_denHaag")
    if not station_dates:
        print("No processed station metadata found for donkey_denHaag.")
        print("Run build_data_tables.py first.")
        return

    if args.date:
        try:
            dt = datetime.strptime(args.date, "%Y-%m-%d")
            selected_date = (dt.year, dt.month, dt.day)
        except ValueError:
            print(f"Invalid --date: {args.date} (expected YYYY-MM-DD)")
            return
        if selected_date not in station_dates:
            print(f"No station metadata for date {args.date}")
            return
    else:
        selected_date = latest_date(station_dates)
        if selected_date is None:
            print("No station metadata dates found.")
            return

    year, month, day = selected_date
    all_stations_df = load_station_day(
        args.data_dir,
        provider="donkey_denHaag",
        year=year,
        month=month,
        day=day,
    )
    if all_stations_df is None or all_stations_df.empty:
        print("Failed to load station metadata table.")
        return

    needed = {"station_id", "name", "lat", "lon", "capacity"}
    if not needed.issubset(all_stations_df.columns):
        print("Station metadata missing required columns.")
        return

    all_stations_df["lat"] = pd.to_numeric(all_stations_df["lat"], errors="coerce")
    all_stations_df["lon"] = pd.to_numeric(all_stations_df["lon"], errors="coerce")
    all_stations_df = all_stations_df.dropna(subset=["lat", "lon"])
    all_stations = all_stations_df[list(needed)].to_dict("records")

    dh_stations = filter_by_bbox(all_stations, DEN_HAAG_BBOX)
    print(f"Total stations: {len(all_stations)}")
    print(f"Den Haag stations (in bbox): {len(dh_stations)}")
    print(f"Date used: {year}-{month:02d}-{day:02d}")

    # Create map
    m = folium.Map(location=DEN_HAAG_CENTER, zoom_start=13, tiles="OpenStreetMap")

    # Draw bounding box
    bbox = DEN_HAAG_BBOX
    folium.Rectangle(
        bounds=[
            [bbox["lat_min"], bbox["lon_min"]],
            [bbox["lat_max"], bbox["lon_max"]],
        ],
        color="gray",
        weight=2,
        dash_array="5 5",
        fill=False,
        popup="Den Haag bounding box",
    ).add_to(m)

    # Add station markers
    for s in dh_stations:
        cap = int(s["capacity"]) if pd.notna(s["capacity"]) else 0
        color = capacity_color(cap)
        radius = max(4, cap * 0.6)
        popup_text = (
            f"<b>{s['name']}</b><br>"
            f"ID: {s['station_id']}<br>"
            f"Capacity: {cap}<br>"
            f"Lat: {s['lat']:.6f}<br>"
            f"Lon: {s['lon']:.6f}"
        )
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=s["name"],
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 13px;">
        <b>Capacity</b><br>
        <span style="color: blue;">&#9679;</span> &le; 3<br>
        <span style="color: green;">&#9679;</span> 4 &ndash; 6<br>
        <span style="color: orange;">&#9679;</span> 7 &ndash; 12<br>
        <span style="color: red;">&#9679;</span> &gt; 12<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"\nMap saved to: {args.output}")

    csv_path, json_path, _ = rebuild_artifact_index()
    print(f"Artifact index updated: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
