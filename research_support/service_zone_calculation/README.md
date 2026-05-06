# Service Zone Calculation

This folder calculates empirical service zones for the Den Haag docked-bike data.

The goal is to replace administrative areas with generated service zones that are better suited for a possible real-data CMDP extension. Administrative polygons are not a good control unit because 500 m station coverage often crosses their boundaries. Service zones give a local spatial unit while still allowing zones to be grouped into ordered service categories.

The setup uses:

- 20 generated service zones
- 5 ordered service categories going from remote (= cat. 0) to central (= cat. 4)
- a service-pressure score to rank zones from lower-pressure to higher-pressure areas
- Den Haag docked-bike station data from Donkey Republic
- pooled ODiN 2018-2023 demand rates mapped to the generated service zones
- BAG non-residential building functions as an activity proxy

## Method

The script loads the latest Donkey Republic Den Haag station snapshot and keeps stations that also appear in the recent docked-bike count data. It then loads address points and converts both stations and addresses to the Dutch RD coordinate system.

The plotted station points come from the latest retrieved Donkey Republic station snapshot, filtered to stations present in the latest 15 docked-bike count days. NS OV-fiets stations are excluded because they are not part of the rebalancing model. For the generated `k20` outputs in this folder, the station snapshot is from 2026-03-20 and the docked-count filter window runs from 2026-03-06 through 2026-03-20.

Addresses within 500 m of at least one station are assigned to their nearest station. These address counts become station weights. The script then runs weighted k-means on station coordinates with `K = 20`, so stations that cover more nearby addresses have more influence on the service-zone centers.

The zone centers are converted into Voronoi polygons clipped to the Den Haag boundary. The borders are station-centered while category assignment uses a service-pressure score. The score ranks zones by expected failure exposure (departure demand) and service relevance (residential and non-residential places of interest):

```text
service_pressure_score =
    (1/3) * normalized ODiN departure demand
  + (1/3) * normalized address density
  + (1/3) * normalized BAG non-residential activity
```

The three score components are weighted equally. ODiN departure demand captures how often people are likely to need a bike in each zone. Address density captures residential access and equity exposure. BAG non-residential activity captures offices, shops, education, healthcare, hospitality, sport, and meeting places that can create demand beyond residential density.

Bike-share literature treats demand as a mix of residential density, employment, land use, POIs, transit access, and time-varying origin/destination patterns [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3) [[4]](#ref-4).

The main outputs are written to `output/`:

- `service_zone_assignments_k20.csv`: station-to-zone and station-to-category assignments
- `service_zone_density_profile_k20.csv`: density, departure-demand, BAG activity, score, and category statistics per service zone
- `service_zone_boundaries_k20.geojson`: generated zone boundaries

The map in `figures/` visualizes the selected service zones and their category grouping.

This folder is exploratory support for the thesis. It does not train or evaluate the synthetic CMDP model.

## References

<a id="ref-1"></a>[1] Chen, Z., van Lierop, D., and Ettema, D. (2022). [Bike Share Usage and the Built Environment: A Review](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.848169/full)

<a id="ref-2"></a>[2] Ma, X., Cao, R., and Jin, Y. (2019). [Spatiotemporal Clustering Analysis of Bicycle Sharing System with Data Mining Approach](https://www.mdpi.com/2078-2489/10/5/163)

<a id="ref-3"></a>[3] Beairsto, J., Tian, Y., Zheng, L., Zhao, Q., and Hong, J. (2022). [Identifying locations for new bike-sharing stations in Glasgow](https://eprints.gla.ac.uk/242478/)

<a id="ref-4"></a>[4] Gervini, D., and Baur, B. (2019). [Exploring Patterns of Demand in Bike Sharing Systems Via Replicated Point Process Models](https://academic.oup.com/jrsssc/article/68/3/585/7058393)
