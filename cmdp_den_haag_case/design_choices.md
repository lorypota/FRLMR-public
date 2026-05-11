# Den Haag CMDP Case

## Service-zone inventory units

The Den Haag CMDP uses empirical service zones as inventory units. One model node represents one generated service zone from `research_support/service_zone_calculation/output/service_zone_assignments_k20.csv`. With the current `k20` zones, the Den Haag scenario has:

```text
node_list = [4, 4, 4, 4, 4]
boundaries = [0, 4, 8, 12, 16, 20]
```

Each service-zone node stores:

```text
service_category  = category assigned in the k20 service-zone output
station_count     = number of Donkey stations assigned to the service zone
capacity          = sum of assigned station capacities
raw_initial_bikes = sum of station bike counts in the selected docked snapshot
initial_bikes     = min(raw_initial_bikes, capacity)
```

The selected docked snapshot is the last row of the latest available
`docked_*.csv` file in `research_support/empirical_analysis/output/data/docked/donkey_denHaag`.

## Demand allocation and factorization

The model keeps category-factorized learning. There is still one Q-table per service category, and zones in the same category are treated as exchangeable units for learning.

Zones in the same category can still have different capacities, initial bikes, station counts, and service-zone IDs. This does not create separate Q-tables. The shared Q-table sees normalized occupancy states and normalized occupancy actions, not raw bike counts. For example, two zones with different capacities but 50 percent occupancy both map to `(0.50, period)`. An action such as `+0.10` then means a 10 percentage-point occupancy increase. The physical number of bikes moved depends on the zone capacity, but the learned policy remains category-level.

The empirical demand input is the ODiN 2018 to 2023 category-period demand rate. For each category and period, the category arrival and departure rates are divided equally over the service zones in that category:

```text
lambda_zone = lambda_category / number_of_service_zones_in_category
```

## State, actions, and reward

The Den Haag service-zone model uses normalized occupancy states. The bike count of each zone is converted to an occupancy fraction in `[0.0, 1.0]` and binned in steps of `0.01`. The state is:

```text
(occupancy_bin, period)
```

Actions are bounded occupancy changes. They mirror the original station action grid, where capacity was 100 bikes, the action step was 5 bikes, and the maximum absolute action was 30 bikes:

```text
[-0.30, -0.25, ..., 0.25, 0.30]
```

An action changes the zone inventory by `round(action * zone_capacity)` bikes, then clips the resulting inventory to `[0, zone_capacity]`.

The rebalancing-cost logic follows the original reward structure. A nonzero effective inventory change in a service zone incurs:

```text
gamma * phi(category)
```

The cost is an operation indicator. The interpretation is that a nonzero change dispatches rebalancing effort to one service zone. The category multiplier `phi` keeps peripheral zones more expensive to service than central zones.

The fleet size penalty is evaluated in normalized occupancy space, using the original capacity-100 target and threshold values as fractions. The penalty is scaled back to percentage-point units so its numerical scale stays close to the original station model.

## Failure constraints and interpretation

Failures occur when sampled departures exceed current zone inventory. The CMDP dual update keeps the original `f_hat` structure:

```text
f_hat = mean failures per category unit and period
```

In this Den Haag case, the category unit is a service zone. Failure thresholds use the same formula as the original CMDP, with the category-period ODiN departure rate after equal division over service zones.
