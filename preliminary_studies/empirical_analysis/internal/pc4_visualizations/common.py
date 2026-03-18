"""Shared helpers for Den Haag PC4 map visualizations."""

from textwrap import dedent

COLOR_BLUE = "#3388ff"
COLOR_GREEN = "#2ca02c"
COLOR_ORANGE = "#ff7f0e"
COLOR_RED = "#d62728"
LIGHT_BLUE = "#d0e4ff"

HOTSPOT_LOW = "#8f1cff"
HOTSPOT_MEDIUM = "#e31c79"
HOTSPOT_HIGH = "#ff6f3c"
HOTSPOT_PEAK = "#ffd166"

HOUSE_NEAR = "#1a9850"
HOUSE_MID = "#91cf60"
HOUSE_FAR = "#fdae61"
HOUSE_VERY_FAR = "#d73027"
HOUSE_NONE = "#7f8c8d"

DEFAULT_VISUALIZATION_MODE = "gradient"


def build_options_html(modes: list[dict[str, str]]) -> str:
    """Render visualization select options from registry metadata."""
    return "\n".join(
        f'                <option value="{mode["value"]}">{mode["label"]}</option>'
        for mode in modes
    )


def build_common_js() -> str:
    """Shared JavaScript helpers used by multiple visualization modes."""
    return dedent(
        f"""
        var THEME_PALETTES = {{
            light: {{
                scaleZero: '{LIGHT_BLUE}',
                scaleBlue: '{COLOR_BLUE}',
                scaleGreen: '{COLOR_GREEN}',
                scaleOrange: '{COLOR_ORANGE}',
                scaleRed: '{COLOR_RED}',
                polygonStroke: '#000000',
                hotspotPolygonFill: '#ffffff',
                hotspotPolygonStroke: '#7a7a7a',
                docklessMarker: '#333333',
                stationMarker: '#e377c2',
                selectedMarker: '#000000',
                hotspotDockless: '{HOTSPOT_LOW}',
                hotspotLow: '{HOTSPOT_MEDIUM}',
                hotspotMedium: '{HOTSPOT_HIGH}',
                hotspotPeak: '{HOTSPOT_PEAK}',
                houseNear: '{HOUSE_NEAR}',
                houseMid: '{HOUSE_MID}',
                houseFar: '{HOUSE_FAR}',
                houseVeryFar: '{HOUSE_VERY_FAR}',
                houseNone: '{HOUSE_NONE}',
                housePolygonStroke: '#9aa7b4',
                gradientBorder: '#999999',
                legendSubtleText: '#555555'
            }},
            dark: {{
                scaleZero: '#35526e',
                scaleBlue: '#58a6ff',
                scaleGreen: '#5ee38b',
                scaleOrange: '#ffb454',
                scaleRed: '#ff7373',
                polygonStroke: '#e3ebf3',
                hotspotPolygonFill: '#d7e4f1',
                hotspotPolygonStroke: '#a7b8ca',
                docklessMarker: '#d0dae4',
                stationMarker: '#ff8fda',
                selectedMarker: '#ffffff',
                hotspotDockless: '#b96aff',
                hotspotLow: '#ff5db8',
                hotspotMedium: '#ff9d57',
                hotspotPeak: '#ffe08a',
                houseNear: '#4fd889',
                houseMid: '#a8e76d',
                houseFar: '#ffc16b',
                houseVeryFar: '#ff8a7d',
                houseNone: '#8a99a8',
                housePolygonStroke: '#6a7d90',
                gradientBorder: '#c0ccd8',
                legendSubtleText: '#c7d3df'
            }}
        }};

        function isHotspotMode(mode) {{
            return mode === 'hotspot';
        }}

        function isHouseMode(mode) {{
            return mode === 'house-proximity';
        }}

        function isProportionalMode(mode) {{
            return mode === 'global' || mode === 'per-date' || mode === 'per-hour';
        }}

        function lerpColor(a, b, t) {{
            return [
                Math.round(a[0] + (b[0] - a[0]) * t),
                Math.round(a[1] + (b[1] - a[1]) * t),
                Math.round(a[2] + (b[2] - a[2]) * t)
            ];
        }}

        function hexToRgbArray(hex) {{
            var normalized = String(hex || '').replace('#', '');
            if (normalized.length === 3) {{
                normalized = normalized[0] + normalized[0] +
                             normalized[1] + normalized[1] +
                             normalized[2] + normalized[2];
            }}
            return [
                parseInt(normalized.slice(0, 2), 16),
                parseInt(normalized.slice(2, 4), 16),
                parseInt(normalized.slice(4, 6), 16)
            ];
        }}

        function getThemeName() {{
            return document.body.classList.contains('theme-dark') ? 'dark' : 'light';
        }}

        function getThemePalette() {{
            return THEME_PALETTES[getThemeName()];
        }}

        function themeColor(name) {{
            return getThemePalette()[name];
        }}

        function getPolygonStrokeColor(mode) {{
            if (isHotspotMode(mode)) return themeColor('hotspotPolygonStroke');
            if (isHouseMode(mode)) return themeColor('housePolygonStroke');
            return themeColor('polygonStroke');
        }}

        function getPolygonFillOpacity(mode) {{
            if (isHotspotMode(mode)) {{
                return getThemeName() === 'dark' ? 0.08 : 0.04;
            }}
            if (isHouseMode(mode)) return 0.0;
            return getThemeName() === 'dark' ? 0.72 : 0.6;
        }}

        function stationColorForAvailability(avail) {{
            var count = Number(avail);
            if (!isFinite(count) || count <= 0) return themeColor('scaleBlue');
            if (count <= 3) return themeColor('scaleGreen');
            if (count <= 6) return themeColor('scaleOrange');
            return themeColor('scaleRed');
        }}

        function ratioToGradient(ratio) {{
            var palette = getThemePalette();
            var blue = hexToRgbArray(palette.scaleBlue);
            var green = hexToRgbArray(palette.scaleGreen);
            var orange = hexToRgbArray(palette.scaleOrange);
            var red = hexToRgbArray(palette.scaleRed);
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

        function escapeHtml(text) {{
            return String(text)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }}
        """
    ).strip()
