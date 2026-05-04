# ODiN Demand Estimation

This folder estimates PC4-period demand rates from ODiN. The output can later be aggregated to service categories and used to replace the synthetic category-period Skellam parameters with empirical estimates:

```text
lambda_arrivals_per_hour(pc4, period)
lambda_departures_per_hour(pc4, period)
```

ODiN is used as a proxy for potential movement demand, not observed shared-bike demand.

## Data Access

The script reads ODiN from the database at <https://metadata.tsn.tno.nl/dataset/odin>. Credentials can be
set through environment variables or through a local `.env` file defined like below (the real `.env` file is ignored by git and should not be committed).

```powershell
ODIN_DB_HOST=aaaaaaa
ODIN_DB_PORT=bbbbbbb
ODIN_DB_NAME=ccccccc
ODIN_DB_USER=ddddddd
ODIN_DB_PASSWORD=eeeeeee
```

The inspected database currently exposes `odin.odin2022` and `odin.odin2023`.
It does not currently expose `odin.odin2024`. The script accepts both older geography column names such as`vertpc` and 2024 DANS-style names such as `vertpc_pram`.

## Demand Definition

The default filters are:

- regular trips only (direct A-to-B movements, excluding series trips and freight records): `verpl = 1`
- Den Haag origin and/or destination: `vertgem = 518 OR aankgem = 518`
- trip distance from 0.5 km to 10 km: `5 <= afstv <= 100`

`afstv` is measured in hectometers. The lower bound removes very short trips
that are often walking trips or spatial assignment noise. The upper bound
captures short-to-medium trips for which cycling can plausibly be an option.
The 10 km upper bound is treated as a modeling choice and should be tested in
sensitivity analysis, but it is consistent with Dutch cycling-range references
that discuss short-to-medium cycling and e-bike substitution ranges [[1]](#ref-1) [[2]](#ref-2).

`khvm` is the ODiN grouped main transport mode for a trip. It is used here
because the demand scenarios are based on the main mode of the full movement,
not on each separate trip leg. In the ODiN codebook, `khvm = 1` means car driver
and `khvm = 5` means bicycle.

The first version estimates three scenarios:

- `bike_suitable_all_modes`: all regular trips in the distance window
- `bike_suitable_car_driver`: same trips with `khvm = 1`
- `bike_suitable_current_bike`: same trips with `khvm = 5`

## Spatial Mapping

ODiN gives origins and destinations as PC4 codes (`vertpc`, `aankpc`). The first
version keeps demand at PC4 level. This is the simplest spatial bridge for ODiN
and is useful for checking whether the demand-estimation pipeline works before
adding the service-zone layer.

To do later:

- map PC4s to the generated `K = 20` service zones;
- aggregate PC4 demand to the five service categories;
- compare PC4-based demand estimates with service-zone-based estimates.

## Usage

Run a small smoke test first:

```bash
uv run preliminary_studies/odin_demand_estimation/estimate_category_demand.py --years 2023 --limit 1000
```

Then run the full supported set:

```bash
uv run preliminary_studies/odin_demand_estimation/estimate_category_demand.py --years 2022 2023
```

The main output is:

```text
preliminary_studies/odin_demand_estimation/output/pc4_period_demand_rates.csv
```

Rows with fewer than 50 unique respondents are flagged with
`low_unique_person_count = true`. These cells should be treated carefully in the
thesis, following the ODiN manual's warning about small filtered samples.

## References

<a id="ref-1"></a>[1] KiM, [Cycling Facts 2023](https://english.kimnet.nl/documents/publications/2024/01/10/cycling-facts-2023)

<a id="ref-2"></a>[2] MDPI Sensors, [The Potential of E-Bikes to Replace Car Trips in the Netherlands](https://www.mdpi.com/1424-8220/23/24/9664)
