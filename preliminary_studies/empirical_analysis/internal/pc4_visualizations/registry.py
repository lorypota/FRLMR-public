"""Registry-backed assembly for Den Haag PC4 visualization modes."""

from textwrap import dedent

from . import fixed, gradient, hotspot, proportional
from .common import DEFAULT_VISUALIZATION_MODE, build_common_js, build_options_html

VISUALIZATION_MODES = proportional.MODES + [gradient.MODE, fixed.MODE, hotspot.MODE]


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
            dedent(
                """
                function polygonStyleForCount(count, allData, globalMax, mode, dateKey, hour) {
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
                            fillOpacity: 0.6,
                            color: 'black'
                        };
                    }
                    if (mode === 'fixed') {
                        return {
                            fillColor: fixedColorForCount(count),
                            fillOpacity: 0.6,
                            color: 'black'
                        };
                    }
                    if (mode === 'hotspot') {
                        return {
                            fillColor: '#ffffff',
                            fillOpacity: 0.04,
                            color: '#7a7a7a'
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
                        fillOpacity: 0.6,
                        color: 'black'
                    };
                }

                function legendHtmlForMode(allData, globalMax, mode, dateKey, hour) {
                    if (mode === 'fixed') return fixedLegendHtml();
                    if (mode === 'gradient') {
                        return gradientLegendHtml(allData, globalMax, mode, dateKey, hour);
                    }
                    if (mode === 'hotspot') return hotspotLegendHtml();
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
