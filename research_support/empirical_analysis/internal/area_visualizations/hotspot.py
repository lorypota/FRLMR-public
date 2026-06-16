"""Hotspot preparation and JS helpers for the Den Haag area map."""

from __future__ import annotations

from textwrap import dedent

MODE = {"value": "hotspot", "label": "Space-time hotspot"}

DOCKLESS_BASE_RADIUS_PX = 11
DOCKLESS_MIN_RADIUS_PX = 7
DOCKLESS_MAX_RADIUS_PX = 26
STATION_BASE_RADIUS_PX = 12
STATION_MAX_RADIUS_PX = 26
STATION_MIN_RADIUS_PX = 10
STATION_ABSOLUTE_MAX_RADIUS_PX = 42
HOTSPOT_ZOOM_BASELINE = 13
HOTSPOT_ZOOM_SCALE_STEP = 1.18
HOTSPOT_MIN_ZOOM_SCALE = 0.72
HOTSPOT_MAX_ZOOM_SCALE = 1.9
HOTSPOT_MARKER_ZOOM_THRESHOLD = 15


def build_hourly_hotspot_data(
    stations: list[dict],
    hours: list[int],
    bbox: dict[str, float] | None = None,
) -> dict[str, dict[str, list | int]]:
    """Build a direct circle-source payload for hotspot rendering."""
    del bbox  # The hotspot now renders directly from points rather than a grid.

    hotspot = {}
    for hour in hours:
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
            "stations": station_points,
            "stationMax": station_max,
        }

    return hotspot


def build_js() -> str:
    """JavaScript helpers for zoom-aware hotspot rendering."""
    return dedent(
        f"""
        var DOCKLESS_BASE_RADIUS_PX = {DOCKLESS_BASE_RADIUS_PX};
        var DOCKLESS_MIN_RADIUS_PX = {DOCKLESS_MIN_RADIUS_PX};
        var DOCKLESS_MAX_RADIUS_PX = {DOCKLESS_MAX_RADIUS_PX};
        var STATION_BASE_RADIUS_PX = {STATION_BASE_RADIUS_PX};
        var STATION_MAX_RADIUS_PX = {STATION_MAX_RADIUS_PX};
        var STATION_MIN_RADIUS_PX = {STATION_MIN_RADIUS_PX};
        var STATION_ABSOLUTE_MAX_RADIUS_PX = {STATION_ABSOLUTE_MAX_RADIUS_PX};
        var HOTSPOT_ZOOM_BASELINE = {HOTSPOT_ZOOM_BASELINE};
        var HOTSPOT_ZOOM_SCALE_STEP = {HOTSPOT_ZOOM_SCALE_STEP};
        var HOTSPOT_MIN_ZOOM_SCALE = {HOTSPOT_MIN_ZOOM_SCALE};
        var HOTSPOT_MAX_ZOOM_SCALE = {HOTSPOT_MAX_ZOOM_SCALE};
        var HOTSPOT_MARKER_ZOOM_THRESHOLD = {HOTSPOT_MARKER_ZOOM_THRESHOLD};

        function getHotspotHourData(allData, dateKey, hour) {{
            var dateData = allData[dateKey];
            if (!dateData || !dateData.hotspot) {{
                return {{ stations: [], stationMax: 0 }};
            }}
            return dateData.hotspot[String(hour)] || {{
                stations: [],
                stationMax: 0
            }};
        }}

        function hotspotClamp(value, minValue, maxValue) {{
            return Math.max(minValue, Math.min(maxValue, value));
        }}

        function hotspotZoomScale(zoom) {{
            var zoomDelta = zoom - HOTSPOT_ZOOM_BASELINE;
            return hotspotClamp(
                Math.pow(HOTSPOT_ZOOM_SCALE_STEP, zoomDelta),
                HOTSPOT_MIN_ZOOM_SCALE,
                HOTSPOT_MAX_ZOOM_SCALE
            );
        }}

        function hotspotDocklessRadius(zoom, hotspotRadiusScale) {{
            return hotspotClamp(
                DOCKLESS_BASE_RADIUS_PX * hotspotZoomScale(zoom) * hotspotRadiusScale,
                DOCKLESS_MIN_RADIUS_PX,
                DOCKLESS_MAX_RADIUS_PX
            );
        }}

        function hotspotStationRadius(avail, stationMax, hotspotRadiusScale, zoom) {{
            if (stationMax <= 0) {{
                return hotspotClamp(
                    STATION_BASE_RADIUS_PX * hotspotZoomScale(zoom) * hotspotRadiusScale,
                    STATION_MIN_RADIUS_PX,
                    STATION_ABSOLUTE_MAX_RADIUS_PX
                );
            }}
            var scaled = Math.sqrt(avail / Math.max(stationMax, 1));
            var baseRadius = STATION_BASE_RADIUS_PX +
                scaled * (STATION_MAX_RADIUS_PX - STATION_BASE_RADIUS_PX);
            return hotspotClamp(
                baseRadius * hotspotZoomScale(zoom) * hotspotRadiusScale,
                STATION_MIN_RADIUS_PX,
                STATION_ABSOLUTE_MAX_RADIUS_PX
            );
        }}

        function hotspotStationOpacity(avail, stationMax) {{
            if (stationMax <= 0) return 0.28;
            var ratio = Math.min(avail / Math.max(stationMax, 1), 1.0);
            return 0.24 + (ratio * 0.24);
        }}

        function hotspotLegendHtml() {{
            return '<div style="margin-bottom:4px;color:' + themeColor('legendSubtleText') + ';">Zoom-aware density circles</div>' +
                   '<span style="color:' + themeColor('hotspotLow') + ';">&#9632;</span> Low station availability<br>' +
                   '<span style="color:' + themeColor('hotspotPeak') + ';">&#9632;</span> High station availability';
        }}

        function hotspotFillColor(avail, stationMax) {{
            var ratio = stationMax > 0 ? Math.min(avail / stationMax, 1.0) : 0;
            if (ratio <= 0.33) return themeColor('hotspotLow');
            if (ratio <= 0.75) return themeColor('hotspotMedium');
            return themeColor('hotspotPeak');
        }}
        """
    ).strip()
