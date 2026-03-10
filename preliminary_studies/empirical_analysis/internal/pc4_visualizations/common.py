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

DEFAULT_VISUALIZATION_MODE = "global"


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
        function isHotspotMode(mode) {{
            return mode === 'hotspot';
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

        function ratioToGradient(ratio) {{
            var blue = [51, 136, 255];
            var green = [44, 160, 44];
            var orange = [255, 127, 14];
            var red = [214, 39, 40];
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

        var LIGHT_BLUE = '{LIGHT_BLUE}';
        """
    ).strip()
