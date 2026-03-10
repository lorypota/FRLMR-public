"""Circle-based hotspot preparation and JS helpers for Den Haag PC4 map."""

from __future__ import annotations

from textwrap import dedent

from .common import HOTSPOT_HIGH, HOTSPOT_LOW, HOTSPOT_MEDIUM, HOTSPOT_PEAK

MODE = {"value": "hotspot", "label": "Space-time hotspot"}

DOCKLESS_RADIUS_METERS = 125
STATION_BASE_RADIUS_METERS = 140
STATION_MAX_RADIUS_METERS = 260
HOTSPOT_MARKER_ZOOM_THRESHOLD = 15


def build_hourly_hotspot_data(
    stations: list[dict],
    bikes_by_hour: dict[str, list[list[float]]],
    hours: list[int],
    bbox: dict[str, float] | None = None,
) -> dict[str, dict[str, list]]:
    """Build a direct circle-source payload for hotspot rendering."""
    del bbox  # The hotspot now renders directly from points rather than a grid.

    hotspot = {}
    for hour in hours:
        dockless_points = [
            [bike[0], bike[1]]
            for bike in bikes_by_hour.get(str(hour), [])
            if len(bike) >= 2
        ]
        station_points = []
        station_max = 0
        for station in stations:
            if hour >= len(station["av"]):
                continue
            avail = station["av"][hour]
            if avail <= 0:
                continue
            station_points.append([station["ll"][0], station["ll"][1], int(avail)])
            station_max = max(station_max, int(avail))

        hotspot[str(hour)] = {
            "dockless": dockless_points,
            "stations": station_points,
            "stationMax": station_max,
        }

    return hotspot


def build_js() -> str:
    """JavaScript helpers for circle-based hotspot rendering."""
    return dedent(
        f"""
        var DOCKLESS_RADIUS_METERS = {DOCKLESS_RADIUS_METERS};
        var STATION_BASE_RADIUS_METERS = {STATION_BASE_RADIUS_METERS};
        var STATION_MAX_RADIUS_METERS = {STATION_MAX_RADIUS_METERS};
        var HOTSPOT_MARKER_ZOOM_THRESHOLD = {HOTSPOT_MARKER_ZOOM_THRESHOLD};

        function getHotspotHourData(allData, dateKey, hour) {{
            var dateData = allData[dateKey];
            if (!dateData || !dateData.hotspot) {{
                return {{ dockless: [], stations: [], stationMax: 0 }};
            }}
            return dateData.hotspot[String(hour)] || {{
                dockless: [],
                stations: [],
                stationMax: 0
            }};
        }}

        function hotspotStationRadius(avail, stationMax, hotspotRadiusScale) {{
            if (stationMax <= 0) return STATION_BASE_RADIUS_METERS;
            var scaled = Math.sqrt(avail / Math.max(stationMax, 1));
            return (STATION_BASE_RADIUS_METERS +
                   scaled * (STATION_MAX_RADIUS_METERS - STATION_BASE_RADIUS_METERS)) *
                   hotspotRadiusScale;
        }}

        function hotspotStationOpacity(avail, stationMax) {{
            if (stationMax <= 0) return 0.28;
            var ratio = Math.min(avail / Math.max(stationMax, 1), 1.0);
            return 0.24 + (ratio * 0.24);
        }}

        function hotspotLegendHtml() {{
            return '<div style="margin-bottom:4px;color:#555;">Overlapping density circles</div>' +
                   '<span style="color:{HOTSPOT_LOW}">&#9632;</span> Dockless bikes: vivid magenta-violet circles<br>' +
                   '<span style="color:{HOTSPOT_HIGH}">&#9632;</span> Stations with availability: larger brighter circles<br>' +
                   '<span style="color:{HOTSPOT_PEAK}">&#9632;</span> Brighter overlap = more supply in this area';
        }}

        function hotspotFillColor(avail, stationMax) {{
            var ratio = stationMax > 0 ? Math.min(avail / stationMax, 1.0) : 0;
            if (ratio <= 0.33) return '{HOTSPOT_MEDIUM}';
            if (ratio <= 0.75) return '{HOTSPOT_HIGH}';
            return '{HOTSPOT_PEAK}';
        }}
        """
    ).strip()
