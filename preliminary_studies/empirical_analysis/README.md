# Empirical Analysis Folder

This folder contains scripts for inspecting GBFS snapshots (TNO internal data), building interactive maps, and running statistical coverage analysis.

## Project Structure

```text
empirical_analysis/
├── stage_gbfs_subset.py         # Stage raw data from TNO network share
├── build_data_tables.py         # Parse tar snapshots into CSV tables
│
├── map_den_haag_stations.py     # Den Haag station map
├── map_den_haag_pc4.py          # Den Haag interactive area map
├── map_amsterdam_pc4.py         # Amsterdam PC4 map
│
├── docs/
│   └── data_notes.md            # Notes on feed quirks and interpretation
│
├── internal/
│   ├── data_utils.py            # Raw snapshot parsing and extraction
│   ├── artifact_index.py        # Internal artifact catalog helper
│   ├── processed_data_utils.py  # CSV table loading helpers
│   ├── paths.py                 # Output directory paths
│
├── output/
│   ├── data/                    # Processed CSV tables (docked, dockless, stations)
│   ├── maps/                    # Interactive HTML maps
│   ├── geodata/                 # GeoJSON files, cached geometries, CBS income data
│   ├── index/                   # Artifact indexing
│   └── analysis/                # Statistical analysis outputs
│
├── AGENTS.md                    # Development guidelines
└── README.md
```

Operational context on Den Haag providers and GBFS coverage is documented in [docs/data_notes.md](docs/data_notes.md).

## Scripts

### Data pipeline

- `stage_gbfs_subset.py`: stage selected providers and snapshots from TNO server into a local folder.
- `build_data_tables.py`: parse raw tar snapshots into docked/dockless/stations CSV tables.

### Maps

- `map_den_haag_stations.py`: build a Den Haag station map.
- `map_den_haag_pc4.py`: build Den Haag area map with date/hour controls, visualization modes (`gradient`, `fixed`, `hotspot`, `house_proximity`), an area-level toggle (`PC4`, `PC6`, `CBS buurten`, `CBS wijken`), and an optional side-by-side compare view.
- `map_amsterdam_pc4.py`: build Amsterdam PC4 map with date/hour controls.
- `artifact_index.py`: rebuild output artifact index files manually if needed.

## Internal Helper Modules

These are used by the runnable scripts above and do not need to be run directly.

- `internal/data_utils.py`: shared data parsing and loading utilities.
- `internal/artifact_index.py`: internal helper that rebuilds `output/index/artifacts.csv` and `output/index/artifacts.json`.
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

Use `--mode first-per-hour` for lighter transfers every hour instead of every minute: 24 snapshots/day instead of 1440. See [docs/data_notes.md](docs/data_notes.md) for staging-size estimates and raw-data coverage notes.

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

`artifacts.csv` and `artifacts.json` include:

- `artifact_type`
- `provider`
- `city`
- `date`
- `path`
- `size_bytes`
- `modified_at_utc`
