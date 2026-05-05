"""House-level proximity visualization for the Den Haag area map."""

from textwrap import dedent

MODE = {"value": "house-proximity", "label": "House proximity"}

HOUSE_MODE_MIN_ZOOM = 13
HOUSE_MODE_DETAIL_ZOOM = 16
HOUSE_MODE_MAX_VISIBLE = 8000

HOUSE_CLUSTER_PIXEL_SIZE_MIN = 50
HOUSE_CLUSTER_PIXEL_SIZE_MAX = 76


def build_js() -> str:
    """JavaScript helpers for the house-proximity visualization."""
    return dedent(
        f"""
        var HOUSE_MODE_MIN_ZOOM = {HOUSE_MODE_MIN_ZOOM};
        var HOUSE_MODE_DETAIL_ZOOM = {HOUSE_MODE_DETAIL_ZOOM};
        var HOUSE_MODE_MAX_VISIBLE = {HOUSE_MODE_MAX_VISIBLE};
        var HOUSE_CLUSTER_PIXEL_SIZE_MIN = {HOUSE_CLUSTER_PIXEL_SIZE_MIN};
        var HOUSE_CLUSTER_PIXEL_SIZE_MAX = {HOUSE_CLUSTER_PIXEL_SIZE_MAX};

        function houseColorForDistance(distanceMeters) {{
            if (distanceMeters === null || !isFinite(distanceMeters)) {{
                return themeColor('houseNone');
            }}
            if (distanceMeters <= 100) return themeColor('houseNear');
            if (distanceMeters <= 250) return themeColor('houseMid');
            if (distanceMeters <= 500) return themeColor('houseFar');
            return themeColor('houseVeryFar');
        }}

        function houseMarkerSvg(color, size) {{
            // Square SVG icon to distinguish houses from bike circle markers
            var half = size / 2;
            return '<svg width="' + size + '" height="' + size + '" ' +
                   'viewBox="0 0 ' + size + ' ' + size + '" ' +
                   'xmlns="http://www.w3.org/2000/svg">' +
                   '<rect x="1" y="1" width="' + (size - 2) + '" height="' + (size - 2) + '" ' +
                   'rx="1" ry="1" ' +
                   'fill="' + color + '" fill-opacity="0.7" ' +
                   'stroke="' + color + '" stroke-opacity="0.9" stroke-width="1"/>' +
                   '</svg>';
        }}

        function houseClusterSvg(color, size, count) {{
            // Larger square with count label for clustered view
            return '<svg width="' + size + '" height="' + size + '" ' +
                   'viewBox="0 0 ' + size + ' ' + size + '" ' +
                   'xmlns="http://www.w3.org/2000/svg">' +
                   '<rect x="1" y="1" width="' + (size - 2) + '" height="' + (size - 2) + '" ' +
                   'rx="2" ry="2" ' +
                   'fill="' + color + '" fill-opacity="0.6" ' +
                   'stroke="' + color + '" stroke-opacity="0.85" stroke-width="1.5"/>' +
                   '<text x="' + (size / 2) + '" y="' + (size / 2 + 1) + '" ' +
                   'text-anchor="middle" dominant-baseline="central" ' +
                   'fill="white" font-size="' + Math.max(9, size * 0.38) + 'px" ' +
                   'font-weight="bold" font-family="sans-serif">' +
                   count + '</text>' +
                   '</svg>';
        }}

        function createHouseIcon(color, size) {{
            var html = houseMarkerSvg(color, size);
            return L.divIcon({{
                html: html,
                className: 'house-marker-icon',
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2]
            }});
        }}

        function createHouseClusterIcon(color, size, count) {{
            var html = houseClusterSvg(color, size, count);
            return L.divIcon({{
                html: html,
                className: 'house-cluster-icon',
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2]
            }});
        }}

        function houseClusterPixelSize(zoom) {{
            if (zoom <= HOUSE_MODE_MIN_ZOOM) return HOUSE_CLUSTER_PIXEL_SIZE_MAX;
            if (zoom >= HOUSE_MODE_DETAIL_ZOOM) return HOUSE_CLUSTER_PIXEL_SIZE_MIN;
            var ratio = (zoom - HOUSE_MODE_MIN_ZOOM) /
                        Math.max(HOUSE_MODE_DETAIL_ZOOM - HOUSE_MODE_MIN_ZOOM, 1);
            return Math.round(
                HOUSE_CLUSTER_PIXEL_SIZE_MAX -
                (ratio * (HOUSE_CLUSTER_PIXEL_SIZE_MAX - HOUSE_CLUSTER_PIXEL_SIZE_MIN))
            );
        }}

        function clusterHousePoints(map, supplyPoints) {{
            // Aggregate houses into zoom-level pixel bins so clusters stay stable while panning.
            var clusters = {{}};
            var pixelSize = houseClusterPixelSize(map.getZoom());
            var zoom = map.getZoom();
            collectVisibleHousePoints(map, null, function(house) {{
                var lat = house[0];
                var lon = house[1];
                var addrs = house[2] || 1;
                var point = map.project([lat, lon], zoom);
                var cKey = Math.floor(point.x / pixelSize) + ':' +
                           Math.floor(point.y / pixelSize);
                if (!clusters[cKey]) {{
                    clusters[cKey] = {{
                        latSum: 0,
                        lonSum: 0,
                        totalAddresses: 0,
                        locationCount: 0
                    }};
                }}
                var c = clusters[cKey];
                c.latSum += lat * addrs;
                c.lonSum += lon * addrs;
                c.totalAddresses += addrs;
                c.locationCount += 1;
            }});
            var result = [];
            var keys = Object.keys(clusters);
            for (var j = 0; j < keys.length; j++) {{
                var cl = clusters[keys[j]];
                var weight = Math.max(cl.totalAddresses, 1);
                var avgLat = cl.latSum / weight;
                var avgLon = cl.lonSum / weight;
                var avgDist = getClosestSupplyDistance(avgLat, avgLon, supplyPoints);
                result.push({{
                    lat: avgLat,
                    lon: avgLon,
                    totalAddresses: cl.totalAddresses,
                    locationCount: cl.locationCount,
                    avgDistance: avgDist
                }});
            }}
            return result;
        }}

        function houseLegendHtml() {{
            return '<div style="margin-bottom:4px;color:' + themeColor('legendSubtleText') + ';">' +
                   'Distance house to nearest sharing bike:' +
                   '</div>' +
                   '<span style="color:' + themeColor('houseNear') + ';">&#9632;</span> 0 &ndash; 100 m<br>' +
                   '<span style="color:' + themeColor('houseMid') + ';">&#9632;</span> 101 &ndash; 250 m<br>' +
                   '<span style="color:' + themeColor('houseFar') + ';">&#9632;</span> 251 &ndash; 500 m<br>' +
                   '<span style="color:' + themeColor('houseVeryFar') + ';">&#9632;</span> 500+ m';
        }}
        """
    ).strip()
