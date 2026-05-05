"""Fixed-threshold polygon visualization for the Den Haag area map."""

from textwrap import dedent

MODE = {"value": "fixed", "label": "Fixed (0 / 1-3 / 4-6 / 7+)"}


def build_js() -> str:
    """JavaScript helpers for fixed visualization."""
    return dedent(
        f"""
        {""}
        function fixedColorForCount(count) {{
            if (count === 0) return themeColor('scaleZero');
            if (count <= 3) return themeColor('scaleRed');
            if (count <= 6) return themeColor('scaleOrange');
            return themeColor('scaleGreen');
        }}

        function fixedLegendHtml() {{
            return '<span style="color:' + themeColor('scaleZero') + ';">&#9632;</span> 0 bikes<br>' +
                   '<span style="color:' + themeColor('scaleRed') + ';">&#9632;</span> 1 &ndash; 3<br>' +
                   '<span style="color:' + themeColor('scaleOrange') + ';">&#9632;</span> 4 &ndash; 6<br>' +
                   '<span style="color:' + themeColor('scaleGreen') + ';">&#9632;</span> 7+<br>';
        }}
        """
    ).strip()
