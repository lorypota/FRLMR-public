"""Proportional polygon visualizations for Den Haag PC4 map."""

from textwrap import dedent

from .common import COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED

MODES = [
    {"value": "global", "label": "Global proportional"},
    {"value": "per-date", "label": "Per-date proportional"},
    {"value": "per-hour", "label": "Per-hour proportional"},
]


def build_js() -> str:
    """JavaScript helpers for the proportional visualization modes."""
    return dedent(
        f"""
        function getHourMax(allData, dateKey, hour) {{
            var dateData = allData[dateKey];
            if (!dateData) return 1;
            var dateCounts = dateData.counts || {{}};
            var mx = 0;
            for (var pc in dateCounts) {{
                var v = dateCounts[pc].c[hour];
                if (v !== undefined && v > mx) mx = v;
            }}
            return mx || 1;
        }}

        function getEffectiveMax(allData, globalMax, mode, dateKey, hour) {{
            if (mode === 'global') return globalMax || 1;
            if (mode === 'per-date' || mode === 'gradient') {{
                var dateData = allData[dateKey];
                return (dateData && dateData.dateMax) ? dateData.dateMax : 1;
            }}
            if (mode === 'per-hour') return getHourMax(allData, dateKey, hour);
            return 1;
        }}

        function proportionalColorForCount(count, allData, globalMax, mode, dateKey, hour) {{
            if (count === 0) return '{COLOR_BLUE}';
            var ratio = Math.min(
                count / getEffectiveMax(allData, globalMax, mode, dateKey, hour),
                1.0
            );
            if (ratio <= 0.33) return '{COLOR_GREEN}';
            if (ratio <= 0.66) return '{COLOR_ORANGE}';
            return '{COLOR_RED}';
        }}

        function proportionalLegendHtml(allData, globalMax, mode, dateKey, hour) {{
            var mx = getEffectiveMax(allData, globalMax, mode, dateKey, hour);
            var t1 = Math.round(mx * 0.33);
            var t2 = Math.round(mx * 0.66);
            var label = mode === 'global' ? 'global' :
                        mode === 'per-date' ? 'this date' : 'this hour';
            return '<span style="color:{COLOR_BLUE}">&#9632;</span> 0<br>' +
                   '<span style="color:{COLOR_GREEN}">&#9632;</span> 1 &ndash; ' + t1 + '<br>' +
                   '<span style="color:{COLOR_ORANGE}">&#9632;</span> ' + (t1 + 1) + ' &ndash; ' + t2 + '<br>' +
                   '<span style="color:{COLOR_RED}">&#9632;</span> ' + (t2 + 1) +
                   '+ (max ' + mx + ' ' + label + ')<br>';
        }}
        """
    ).strip()
