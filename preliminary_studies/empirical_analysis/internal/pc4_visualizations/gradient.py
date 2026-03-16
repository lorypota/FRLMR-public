"""Continuous gradient polygon visualization for Den Haag PC4 map."""

from textwrap import dedent

MODE = {"value": "gradient", "label": "Continuous gradient"}


def build_js() -> str:
    """JavaScript helpers for gradient visualization."""
    return dedent(
        f"""
        {""}
        function gradientColorForCount(count, allData, globalMax, mode, dateKey, hour) {{
            if (count === 0) return themeColor('scaleZero');
            var mx = getEffectiveMax(allData, globalMax, mode, dateKey, hour);
            var lowCut = Math.min(5, mx);
            var ratio;
            if (count <= lowCut) {{
                ratio = (count / Math.max(lowCut, 1)) * 0.4;
            }} else {{
                ratio = 0.4 + ((count - lowCut) / Math.max(mx - lowCut, 1)) * 0.6;
            }}
            return ratioToGradient(Math.min(ratio, 1.0));
        }}

        function gradientLegendHtml(allData, globalMax, mode, dateKey, hour) {{
            var mx = getEffectiveMax(allData, globalMax, mode, dateKey, hour);
            return '<span style="color:' + themeColor('scaleZero') + ';">&#9632;</span> 0<br>' +
                   '<div style="height:14px;width:120px;border:1px solid ' + themeColor('gradientBorder') + ';border-radius:2px;' +
                   'background:linear-gradient(to right,' +
                   themeColor('scaleBlue') + ',' +
                   themeColor('scaleGreen') + ',' +
                   themeColor('scaleOrange') + ',' +
                   themeColor('scaleRed') + ');' +
                   'margin:4px 0;"></div>' +
                   '1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ' + mx + '<br>';
        }}
        """
    ).strip()
