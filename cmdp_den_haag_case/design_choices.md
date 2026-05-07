# Den Haag CMDP Case

## ODiN demand station approximation

This folder uses ODiN category-period demand rates as the first empirical input for the Den Haag CMDP. The modeling issue is that ODiN currently provides demand at category-period level, while the station simulator consumes station-level Skellam parameters.

The current approximation divides each category-period rate equally across stations in that category:

```text
lambda_station = lambda_category / number_of_stations_in_category
```

The limitation is that stations in the same category are unlikely to face identical demand. Better allocation rules could distribute category demand by station capacity, by service-zone demand, or by a combined score.
