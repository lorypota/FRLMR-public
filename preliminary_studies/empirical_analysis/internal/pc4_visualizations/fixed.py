"""Fixed-threshold polygon visualization for Den Haag PC4 map."""

from textwrap import dedent

from .common import COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED

MODE = {"value": "fixed", "label": "Fixed (0 / 1-3 / 4-6 / 7+)"}


def build_js() -> str:
    """JavaScript helpers for fixed visualization."""
    return dedent(
        f"""
        function fixedColorForCount(count) {{
            if (count === 0) return '{COLOR_BLUE}';
            if (count <= 3) return '{COLOR_GREEN}';
            if (count <= 6) return '{COLOR_ORANGE}';
            return '{COLOR_RED}';
        }}

        function fixedLegendHtml() {{
            return '<span style="color:{COLOR_BLUE}">&#9632;</span> 0 bikes<br>' +
                   '<span style="color:{COLOR_GREEN}">&#9632;</span> 1 &ndash; 3<br>' +
                   '<span style="color:{COLOR_ORANGE}">&#9632;</span> 4 &ndash; 6<br>' +
                   '<span style="color:{COLOR_RED}">&#9632;</span> 7+<br>';
        }}
        """
    ).strip()
