# Data Notes

This file records findings about the meaning and quality of the processed GBFS data.

## Den Haag provider context

Shared-mobility operators of interest for Den Haag:

| Operator            | Vehicle type                     | In GBFS feed?          | Notes                                                        |
| ------------------- | -------------------------------- | ---------------------- | ------------------------------------------------------------ |
| **Donkey Republic** | Shared bikes (docked + dockless) | Yes (`donkey_denHaag`) | Main data source in this folder                              |
| **HTM**             | Shared bikes                     | No                     | HTM ended shared-bike activity early 2024                    |
| **Cargoroo**        | Cargo bikes                      | No                     | Listed by municipality but no GBFS feed on the network share |
| **Bondi**           | Shared bikes                     | No                     | Not listed on municipal provider page                        |

Additionally, **NS OV-fiets** (`ns_ov_fiets`) has a small docked-only presence at Den Haag train stations.

The GBFS network share contains more providers than those used here, but only `donkey_denHaag` and `ns_ov_fiets` are relevant for Den Haag in this repo. Providers that mainly cover other cities include CKL/Cykl, check*\*, dott*\*, goabout, and other Donkey feeds.

The [Dashboard Deelmobiliteit](https://crow-smartmobility.nl/kenniscatalogus/dashboard-deelmobiliteit/) aggregates shared-mobility data across the Netherlands using GBFS, MDS, and TOMP standards. Access requires a government login via <info@deelfietsdashboard.nl>. HTM, Cargoroo, and Bondi data may be available there even though they are not present on the TNO network share used here.

## Raw-data coverage and zip files sizes

For the staged raw data used by this folder:

- data is available from `2021-04` to present
- `donkey_denHaag` starts from `2021-10`

Approximate zip files sizes for `donkey_denHaag + ns_ov_fiets`:

| Period  | 2022 per-minute | 2026 per-minute | 2026 per-hour |
| ------- | --------------- | --------------- | ------------- |
| 1 day   | ~40 MB          | ~130 MB         | ~2.2 MB       |
| 1 week  | ~280 MB         | ~900 MB         | ~15 MB        |
| 1 month | ~1.1 GB         | ~3.6 GB         | ~62 MB        |

Approximate file-size trend:

- Donkey Den Haag: about `15 KB/file` in 2021 to about `78 KB/file` in 2026 (as more stations are added thorough time)
- NS OV-fiets: about `14 KB/file` across the period

## Donkey Den Haag: docked vs dockless

For this project, Donkey Den Haag dockless bikes are excluded from the map and not used in the main analysis. Looking at the data there seems to big a lot of overlapping with docked bikes in `station_status` and free-floating bikes in `free_bike_status`, making this close enough to treat dockless supply as station-duplicative in practice:

- recent dockless rows usually carry a `station_id` (which is almost always the nearest station)
- almost all dockless bikes are very close to a station
- dockless counts near a station are not usually equal to the docked counts at that station and time, so this is not an exact row-level duplicate feed

Checks on Donkey Den Haag tables showed:

- `2023-12-12`: about `90.4%` of dockless rows were within `100 m` of a station
- `2024-12-12`: about `98.2%` within `100 m`
- `2025-03-12`: about `97.5%` within `100 m`
- `2026-03-20`: about `99.0%` within `100 m`

### `last_reported` field in dockless bikes

The `last_reported` field changes format over time:

- older snapshots such as `2023-12-12` use ISO timestamps
- newer snapshots such as `2024-12-12` onward use Unix seconds

Observed recency:

- `2023-12-12`: very stale, median age about `123` days
- `2024-12-12`: much fresher, median age about `8.5` hours
- `2025-03-12`: much fresher, median age about `8.2` hours
- `2026-03-20`: still usable, but less fresh, median age about `11.3` hours

So:

- late 2023 dockless `last_reported` values look dirty
- 2024 to 2026 values look current enough for broad spatial analysis but not entirely reliable
