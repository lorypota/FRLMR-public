# ODiN Demand Estimation

This folder estimates Den Haag movement-demand rates from ODiN. PC4 is used as the intermediate geography because ODiN stores origin and destination locations as postcode and administrative codes, not exact coordinates. The main CMDP input is the category-period table:

```text
lambda_arrivals_per_hour(service_category, period)
lambda_departures_per_hour(service_category, period)
```

ODiN is used as a proxy for potential movement demand, not observed shared-bike demand.

## Data Access

The script reads ODiN from the database at <https://metadata.tsn.tno.nl/dataset/odin>. Credentials can be set through environment variables or through a local `.env` file defined like below (the real `.env` file is ignored by git and should not be committed).

```powershell
ODIN_DB_HOST=aaaaaaa
ODIN_DB_PORT=bbbbbbb
ODIN_DB_NAME=ccccccc
ODIN_DB_USER=ddddddd
ODIN_DB_PASSWORD=eeeeeee
```

The inspected database currently exposes `odin.odin2018` through `odin.odin2023`. The script accepts both older geography column names such as `vertpc` and 2024 DANS-style names such as `vertpc_pram`.

## Demand Definition

The default filters are:

- regular trips only (direct A-to-B movements, excluding series trips and freight records): `verpl = 1`
- Den Haag origin and/or destination: `vertgem = 518 OR aankgem = 518`
- trip distance from 0.5 km to 10 km: `5 <= afstv <= 100`

`afstv` is measured in hectometers. The lower bound removes very short trips that are often walking trips or spatial assignment noise. The upper bound captures short-to-medium trips for which cycling can plausibly be an option. The 10 km upper bound is treated as a modeling choice and should be tested in sensitivity analysis, but it is consistent with Dutch cycling-range references that discuss short-to-medium cycling and e-bike substitution ranges [[1]](#ref-1) [[2]](#ref-2).

`khvm` is the ODiN grouped main transport mode for a trip. It is queried for diagnostics, but the main estimator doesn't split demand by mode. The primary estimate uses all regular trips in the distance window to avoid fragmenting the sample before spatial aggregation.

## Spatial Mapping

ODiN gives origins and destinations as PC4 codes (`vertpc`, `aankpc`). The script maps each PC4 to the generated `K = 20` service zones using dominant polygon overlap with:

```text
preliminary_studies/service_zone_calculation/output/service_zone_boundaries_k20.geojson
```

It then produces PC4-level, service-zone-level, and service-category-level outputs. PC4 remains useful for checking the spatial bridge, while the category-period output is the most stable first input for the Skellam-style CMDP.

## Usage

Run a small smoke test first:

```bash
uv run preliminary_studies/odin_demand_estimation/estimate_category_demand.py --years 2023 --limit 1000
```

Then run the full supported set:

```bash
uv run preliminary_studies/odin_demand_estimation/estimate_category_demand.py
```

The script writes:

```text
preliminary_studies/odin_demand_estimation/output/category_period_demand_rates.csv
preliminary_studies/odin_demand_estimation/output/pc4_period_demand_rates.csv
preliminary_studies/odin_demand_estimation/output/pc4_od_demand_rates.csv
preliminary_studies/odin_demand_estimation/output/service_zone_period_demand_rates.csv
preliminary_studies/odin_demand_estimation/output/service_zone_od_demand_rates.csv
```

When multiple years are selected, the outputs also include a pooled row group such as `pooled_2018_2023`. Pooled weighted demand is averaged across years rather than summed, so the hourly rates remain annual demand rates.

Rows with fewer than 50 unique respondents are flagged with `low_unique_person_count = true`. These cells should be treated carefully, following the ODiN manual's warning about small filtered samples.

## Outputs interpretation

The outputs are meant for different levels of the modeling study:

- `category_period_demand_rates.csv`: main CMDP input. It gives departures and arrivals by service category and period, matching the current Skellam-style setup with one demand profile per category-period pair.
- `service_zone_period_demand_rates.csv`: richer spatial input. It gives departures and arrivals by service zone and period, useful if the real-data CMDP is later moved from category-level demand to zone-level demand.
- `pc4_period_demand_rates.csv`: intermediate diagnostic output. It shows the PC4-level demand before spatial aggregation, but PC4 cells are often too sparse for direct CMDP training.
- `service_zone_od_demand_rates.csv`: directional movement output between service zones. It is useful for describing OD patterns, but many OD cells are too sparse for the first transition model.
- `pc4_od_demand_rates.csv`: finest OD output. It is mainly diagnostic because PC4-to-PC4 OD demand is highly fragmented.

The full `2018-2023` run showed the following low-count pattern:

```text
category_period_demand_rates.csv:       70 rows, 0 low-count rows
service_zone_period_demand_rates.csv:  280 rows, 27 low-count rows
pc4_period_demand_rates.csv:          2003 rows, 1681 low-count rows
service_zone_od_demand_rates.csv:     4473 rows, 4308 low-count rows
pc4_od_demand_rates.csv:             23347 rows, 23285 low-count rows
```

This means the category-period output is the safest first input for the real-data CMDP. The service-zone-period output is also promising, especially for pooled years. The OD outputs should be treated as descriptive evidence of movement direction unless further aggregation is introduced.

## References

<a id="ref-1"></a>[1] KiM, [Cycling Facts 2023](https://english.kimnet.nl/documents/publications/2024/01/10/cycling-facts-2023)

<a id="ref-2"></a>[2] MDPI Sensors, [The Potential of E-Bikes to Replace Car Trips in the Netherlands](https://www.mdpi.com/1424-8220/23/24/9664)
