# Den Haag CMDP Case

## ODiN demand station approximation

This folder uses ODiN 2018 to 2023 category-period demand rates as the first empirical input for the Den Haag CMDP. It also uses real Donkey station counts by service category from `research_support/service_zone_calculation/output/service_zone_assignments_k20.csv`.

The modeling issue is that ODiN currently provides demand at category-period level, while the station simulator consumes station-level Skellam parameters. The current approximation divides each category-period rate equally across the real station count in that category:

```text
lambda_station = lambda_category / number_of_stations_in_category
```

The limitation is that stations in the same category are unlikely to face identical demand. Better allocation rules could distribute category demand by station capacity, by service-zone demand, or by a combined score.

## Exploration calibration with real station counts

The Q-learning agent updates epsilon after every station-level Q-table update. If the real station counts were used with the original fixed epsilon decay, categories with many real stations would stop exploring much faster than they did in the synthetic setup.

For the Den Haag case, epsilon decay is therefore scaled per category by:

```text
epsilon_decay_category = base_epsilon_decay * synthetic_station_count / real_station_count
```

This keeps the exploration schedule approximately tied to training days rather than to the number of station units in a category.

## Scope

This folder is an empirical demand calibration of the existing CMDP model, not a full empirical station-level or service-zone simulator.

Each category contains exchangeable station units based on the real station count. Category demand is divided equally over those units, each unit has the fixed simulator capacity of 100 bikes, and station-level failures occur when sampled synthetic departures exceed the unit's current inventory.

This setup is useful for initial results because it tests how the CMDP fairness formulation behaves when the category demand pressure is calibrated to the available data for Den Haag. The results should be interpreted as evidence on the algorithm sensitivity, not as real estimates.

A fuller empirical model would replace the synthetic units with real spatial inventory units, for example service zones with aggregate capacity and initial bikes. That would require changes to state representation, available actions, capacity handling, reward targets, and evaluation. It is intentionally outside the current initial-results setup.
