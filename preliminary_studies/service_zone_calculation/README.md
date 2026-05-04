# Service Zone Calculation

This folder calculates empirical service zones for the Den Haag docked-bike data.

The goal is to replace administrative areas with generated service zones that are better suited for a possible real-data CMDP extension. Administrative polygons are not a good control unit because 500 m station coverage often crosses their boundaries. Service zones give a local spatial unit while still allowing zones to be grouped into ordered service categories.

The current reference setup uses:

- 20 generated service zones
- 5 ordered service categories
- address density to rank zones from lower-density to higher-density areas
- Den Haag docked-bike station data from Donkey Republic and NS OV-fiets

## Method

The script loads the latest Den Haag station snapshot for both providers and keeps stations that also appear in the recent docked-bike count data. It then loads address points and converts both stations and addresses to the Dutch RD coordinate system.

Addresses within 500 m of at least one station are assigned to their nearest station. These address counts become station weights. The script then runs weighted k-means on station coordinates with `K = 20`, so stations that cover more nearby addresses have more influence on the service-zone centers.

The zone centers are converted into Voronoi polygons clipped to the Den Haag boundary. Each zone is assigned an address density, and zones are ranked by density into 5 repeated service categories (the 5 defined by the beta and cmdp paper going from remote (= cat. 0) to central (= cat. 4)).

The main outputs are written to `output/`:

- `service_zone_assignments_k20.csv`: station-to-zone and station-to-category assignments
- `service_zone_density_profile_k20.csv`: address-density statistics per service zone
- `service_zone_boundaries_k20.geojson`: generated zone boundaries

The map in `figures/` visualizes the selected service zones and their category grouping.

This folder is exploratory support for the thesis. It does not train or evaluate the synthetic CMDP model.
