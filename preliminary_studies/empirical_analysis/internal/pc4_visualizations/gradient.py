"""Continuous gradient polygon visualization for Den Haag PC4 map."""

from textwrap import dedent

MODE = {"value": "gradient", "label": "Continuous gradient"}


def build_js() -> str:
    """JavaScript helpers for gradient visualization."""
    return dedent(
        f"""
        {""}
        function percentileValue(values, percentile) {{
            if (!values.length) return 1;
            var sorted = values.slice().sort(function(a, b) {{
                return a - b;
            }});
            var rank = Math.min(
                sorted.length - 1,
                Math.max(0, Math.ceil((percentile / 100) * sorted.length) - 1)
            );
            return sorted[rank] || 1;
        }}

        function getGradientScaleMax(allData, globalMax, dateKey, hour) {{
            var dateData = allData[dateKey];
            if (!dateData) {{
                return globalMax || 1;
            }}
            var values = [];
            var dateCounts = getVisualizationCounts(dateData);
            for (var pc in dateCounts) {{
                if (!Object.prototype.hasOwnProperty.call(dateCounts, pc)) continue;
                var count = dateCounts[pc].c[hour];
                if (count !== undefined && count > 0) {{
                    values.push(count);
                }}
            }}
            if (!values.length) return 1;
            return Math.max(1, percentileValue(values, 95));
        }}

        function gradientColorForCount(count, allData, globalMax, mode, dateKey, hour) {{
            if (count === 0) return themeColor('scaleZero');
            var mx = getGradientScaleMax(allData, globalMax, dateKey, hour);
            var lowCut = Math.min(5, mx);
            var cappedCount = Math.min(count, mx);
            var ratio;
            if (cappedCount <= lowCut) {{
                ratio = (cappedCount / Math.max(lowCut, 1)) * 0.4;
            }} else {{
                ratio = 0.4 + ((cappedCount - lowCut) / Math.max(mx - lowCut, 1)) * 0.6;
            }}
            return ratioToGradient(Math.min(ratio, 1.0));
        }}

        function gradientLegendHtml(allData, globalMax, mode, dateKey, hour) {{
            var mx = getGradientScaleMax(allData, globalMax, dateKey, hour);
            return '<span style="color:' + themeColor('scaleZero') + ';">&#9632;</span> 0<br>' +
                   '<div style="height:14px;width:120px;border:1px solid ' + themeColor('gradientBorder') + ';border-radius:2px;' +
                   'background:linear-gradient(to right,' +
                   themeColor('scaleZero') + ',' +
                   themeColor('scaleRed') + ',' +
                   themeColor('scaleOrange') + ',' +
                   themeColor('scaleGreen') + ');' +
                   'margin:4px 0;"></div>' +
                   '1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ' +
                   mx + ' (p95 cap)<br>';
        }}
        """
    ).strip()
