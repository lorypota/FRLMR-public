"""
Task 2: Map Den Haag Donkey Stations
=====================================

Run:
    uv run python preliminary_studies/empirical_analysis/map_den_haag_stations.py

Filters stations from station_information.json that fall within Den Haag's
bounding box (lat 52.03-52.12, lon 4.22-4.38) and plots them on an
interactive folium map, color-coded by capacity.

Output:
    output/maps/den_haag_stations.html
"""

import argparse
import os
import sys

import folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_index import rebuild_artifact_index
from data_utils import (
    DEFAULT_DATA_ROOT,
    DEN_HAAG_BBOX,
    DEN_HAAG_CENTER,
    filter_den_haag_stations,
    get_station_info,
    list_tar_files,
)
from paths import MAPS_DIR, ensure_output_dirs


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
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output", default=str(MAPS_DIR / "den_haag_stations.html")
    )
    args = parser.parse_args()

    # Load station info from first snapshot of Jan 1
    files = list_tar_files(args.data_root, 2025, 1, 1, hour=0)
    if not files:
        print("No data files found.")
        return

    all_stations = get_station_info(files[0])
    if all_stations is None:
        print("Failed to parse station_information.")
        return

    dh_stations = filter_den_haag_stations(all_stations)
    print(f"Total stations: {len(all_stations)}")
    print(f"Den Haag stations (in bbox): {len(dh_stations)}")

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
        color = capacity_color(s["capacity"])
        radius = max(4, s["capacity"] * 0.6)
        popup_text = (
            f"<b>{s['name']}</b><br>"
            f"ID: {s['station_id']}<br>"
            f"Capacity: {s['capacity']}<br>"
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
