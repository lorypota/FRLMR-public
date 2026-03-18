"""Registry-backed assembly for Den Haag PC4 visualization modes."""

from textwrap import dedent

from . import fixed, gradient, hotspot, house_proximity, proportional
from .common import DEFAULT_VISUALIZATION_MODE, build_common_js, build_options_html

VISUALIZATION_MODES = [
    gradient.MODE,
    fixed.MODE,
    hotspot.MODE,
    house_proximity.MODE,
]


def build_visualization_options_html() -> str:
    """Render select options from visualization registry metadata."""
    return build_options_html(VISUALIZATION_MODES)


def build_visualization_js() -> str:
    """Assemble visualization-mode JavaScript helpers."""
    return "\n\n".join(
        [
            build_common_js(),
            proportional.build_js(),
            gradient.build_js(),
            fixed.build_js(),
            hotspot.build_js(),
            house_proximity.build_js(),
            dedent(
                """
                function polygonStyleForCount(count, allData, globalMax, mode, dateKey, hour) {
                    if (mode === 'house-proximity') {
                        return {
                            fillColor: themeColor('houseNone'),
                            fillOpacity: getPolygonFillOpacity(mode),
                            color: getPolygonStrokeColor(mode)
                        };
                    }
                    if (mode === 'gradient') {
                        return {
                            fillColor: gradientColorForCount(
                                count,
                                allData,
                                globalMax,
                                mode,
                                dateKey,
                                hour
                            ),
                            fillOpacity: getPolygonFillOpacity(mode),
                            color: getPolygonStrokeColor(mode)
                        };
                    }
                    if (mode === 'fixed') {
                        return {
                            fillColor: fixedColorForCount(count),
                            fillOpacity: getPolygonFillOpacity(mode),
                            color: getPolygonStrokeColor(mode)
                        };
                    }
                    if (mode === 'hotspot') {
                        return {
                            fillColor: themeColor('hotspotPolygonFill'),
                            fillOpacity: getPolygonFillOpacity(mode),
                            color: getPolygonStrokeColor(mode)
                        };
                    }
                    return {
                        fillColor: proportionalColorForCount(
                            count,
                            allData,
                            globalMax,
                            mode,
                            dateKey,
                            hour
                        ),
                        fillOpacity: getPolygonFillOpacity(mode),
                        color: getPolygonStrokeColor(mode)
                    };
                }

                function legendHtmlForMode(allData, globalMax, mode, dateKey, hour) {
                    if (mode === 'fixed') return fixedLegendHtml();
                    if (mode === 'gradient') {
                        return gradientLegendHtml(allData, globalMax, mode, dateKey, hour);
                    }
                    if (mode === 'hotspot') return hotspotLegendHtml();
                    if (mode === 'house-proximity') return houseLegendHtml();
                    return proportionalLegendHtml(allData, globalMax, mode, dateKey, hour);
                }
                """
            ).strip(),
        ]
    )


__all__ = [
    "DEFAULT_VISUALIZATION_MODE",
    "VISUALIZATION_MODES",
    "build_visualization_js",
    "build_visualization_options_html",
]
