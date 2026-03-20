# Empirical Analysis Folder

This folder contains small scripts used to inspect GBFS snapshots (TNO internal data) and build map artifacts.

The main current focus is **Den Haag** (although older exploration files of Amsterdam are not deleted), which has four shared-mobility operators of interest:

| Operator            | Vehicle type                     | In GBFS feed?          | Notes                                                        |
| ------------------- | -------------------------------- | ---------------------- | ------------------------------------------------------------ |
| **Donkey Republic** | Shared bikes (docked + dockless) | Yes (`donkey_denHaag`) | Main data source: 436 stations, ~900 free bikes              |
| **HTM**             | Shared bikes                     | No                     | HTM ended shared-bike activity early 2024                    |
| **Cargoroo**        | Cargo bikes                      | No                     | Listed by municipality but no GBFS feed on the network share |
| **Bondi**           | Shared bikes                     | No                     | Not listed on municipal provider page                        |

Additionally, **NS OV-fiets** (`ns_ov_fiets`) has 6 docked stations in Den Haag at train stations (HS, Centraal, Voorburg, Rijswijk, Mariahoeve, Laan van NOI).

The GBFS network share contains 13 providers total (see full list in the data exploration notes below), but only `donkey_denHaag` and `ns_ov_fiets` have data within Den Haag. Providers that cover other cities: CKL/Cykl, check*\*, dott*\*, goabout, donkey.

The [Dashboard Deelmobiliteit](https://crow-smartmobility.nl/kenniscatalogus/dashboard-deelmobiliteit/) aggregates data from shared-mobility providers across NL using GBFS/MDS/TOMP standards. Access requires a government login via <info@deelfietsdashboard.nl>. HTM, Cargoroo, and Bondi data may be available there but is not on the TNO network share.

## Scripts

- `stage_gbfs_subset.py`: stage selected providers and snapshots from TNO server into a local folder.
- `build_data_tables.py`: parse raw tar snapshots into docked/dockless/stations CSV tables.
- `map_den_haag_stations.py`: build a Den Haag station map.
- `map_den_haag_pc4.py`: build Den Haag area map with date/hour controls, visualization modes (`gradient`, `fixed`, `hotspot`, `house_proximity`), an area-level toggle (`PC4`, `PC6`, `CBS buurten`, `CBS wijken`), and an optional side-by-side compare view.
- `map_amsterdam_pc4.py`: build Amsterdam PC4 map with date/hour controls.
- `artifact_index.py`: rebuild output artifact index files manually if needed.

## Internal Helper Modules

These are used by the runnable scripts above and do not need to be run directly.

- `internal/data_utils.py`: shared data parsing and loading utilities.
- `internal/processed_data_utils.py`: helpers for reading processed CSV tables.
- `internal/paths.py`: shared output folder paths.
- `internal/pc4_visualizations/`: modules per visualization type (`gradient`, `fixed`, `hotspot`, `house_proximity`).

## How To Run

The workflow is (skip first two steps to just run visualization with already saved data):

1. **stage** raw data from the network share
2. **build** CSV tables into `output/data/`
3. open or refresh the Den Haag PC4 map over HTTP

### 1. Stage raw data

Copy selected providers from the TNO network share into `output/raw_staging/`. This staged data is not committed by default as it contains many .zip files that would take a lot of space in the repository. Uses `--start`/`--end` for date ranges and skips already-staged files:

```powershell
uv run preliminary_studies/empirical_analysis/stage_gbfs_subset.py `
  --source-root "\\\\tsn.tno.nl\\RA-Data\\SV\\sv-057767\\Feeds\\OpenOV\\GBFS" `
  --start 2026-02-01 --end 2026-02-07 `
  --providers donkey_denHaag ns_ov_fiets
```

Use `--mode first-per-hour` for lighter transfers every hour instead of every minute (24 snapshots/day instead of 1440).

**Size estimates** for `donkey_denHaag + ns_ov_fiets`:

| Period  | 2026 per-minute | 2022 per-minute | 2026 per-hour |
| ------- | --------------- | --------------- | ------------- |
| 1 day   | ~130 MB         | ~40 MB          | ~2.2 MB       |
| 1 week  | ~900 MB         | ~280 MB         | ~15 MB        |
| 1 month | ~3.6 GB         | ~1.1 GB         | ~62 MB        |

Data is available from 2021-04 to present. `donkey_denHaag` starts from 2021-10. File sizes grew over time as Donkey added more stations (15 KB/file in 2021 → 78 KB/file in 2026). NS OV-fiets stays ~14 KB/file throughout.

### 2. Build CSV tables

Parse the staged raw tars into processed CSVs in `output/data/`:

```powershell
uv run preliminary_studies/empirical_analysis/build_data_tables.py `
  --data-root preliminary_studies/empirical_analysis/output/raw_staging `
  --providers donkey_denHaag ns_ov_fiets
```

### 3. Open the Den Haag map

Do not open `den_haag_pc4.html` via `file://`. Browser fetches for the runtime-loaded data need a local HTTP server.

The map starts on `PC4`. `PC6`, `CBS buurten`, and `CBS wijken` can be selected in the area-level control. The visualization control supports `gradient`, `fixed`, `hotspot`, and `house_proximity`.

Run a local server from the repo root:

```bash
uv run python -m http.server 8000
```

Then open: <http://localhost:8000/preliminary_studies/empirical_analysis/output/maps/den_haag_pc4.html>

## Output Structure

All generated artifacts are under `output/`:

```text
output/
  data/
    docked/
      donkey_denHaag/ (and ns_ov_fiets/)
        docked_YYYYMMDD.csv  ← num_bikes_available per station (timestamp × station_id)
        docks_YYYYMMDD.csv   ← num_docks_available per station (timestamp × station_id)
    dockless/
      donkey_denHaag/
        dockless_YYYYMMDD.csv  ← free bike positions (timestamp, bike_id, lat, lon,
                                  is_reserved, is_disabled, last_reported, station_id,
                                  vehicle_type_id)
    stations/
      donkey_denHaag/ (and ns_ov_fiets/)
        stations_YYYYMMDD.csv  ← station metadata (station_id, name, lat, lon, capacity,
                                  is_virtual_station, region_id)
  maps/
    den_haag_stations.html
    den_haag_pc4.html
    amsterdam_pc4.html
  geodata/
    pc4_den_haag.geojson
    pc6_den_haag/
      part_01.geojson
      part_02.geojson
    buurten_den_haag.geojson
    wijken_den_haag.geojson
    pc4_amsterdam.geojson
    houses_den_haag.json
  index/
    artifacts.csv
    artifacts.json
```

## Artifact Indexing

The map scripts and `build_data_tables.py` rebuild the artifact index automatically at the end of each run.

You can also rebuild it manually:

```bash
uv run preliminary_studies/empirical_analysis/artifact_index.py
```

`artifacts.csv` and `artifacts.json` include:

- `artifact_type`
- `provider`
- `city`
- `date`
- `path`
- `size_bytes`
- `modified_at_utc`
