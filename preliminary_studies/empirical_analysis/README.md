# Empirical Analysis Folder

This folder contains small scripts used to inspect Donkey GBFS snapshots and build map artifacts for Den Haag and Amsterdam.

## Scripts

- `inspect_snapshot.py`: print and inspect one raw snapshot archive.
- `check_temporal_coverage.py`: scan available dates/hours and report missing minutes.
- `map_den_haag_stations.py`: build a Den Haag station map.
- `map_den_haag_pc4_timeslider.py`: build Den Haag PC4 map with date/hour controls.
- `map_amsterdam_pc4_timeslider.py`: build Amsterdam PC4 map with date/hour controls.
- `artifact_index.py`: rebuild output artifact index files.
- `build_data_tables.py`: parse raw tar snapshots into docked/dockless/stations CSV tables.
- `data_utils.py`: shared data parsing and loading utilities.
- `processed_data_utils.py`: helpers for reading processed CSV tables.
- `paths.py`: shared output folder paths.

## How To Run

From repository root (after cloning):

Raw-processing scripts (`inspect_snapshot.py`, `check_temporal_coverage.py`, `build_data_tables.py`) look for raw snapshots in this order:
1. `--data-root` argument (per command)
2. `DONKEY_DATA_ROOT` environment variable (session/global override)
3. default `./data` (repo-relative fallback)

Data root means the folder that contains the date tree:

```text
<data-root>/
  YYYY/MM/DD/HH/<provider>_fietsData_YYYYMMDDHHMM.tar.gz
```

If raw data is in `./data`, you do not need extra flags:

```bash
uv run python preliminary_studies/empirical_analysis/inspect_snapshot.py
uv run python preliminary_studies/empirical_analysis/check_temporal_coverage.py
uv run python preliminary_studies/empirical_analysis/build_data_tables.py
uv run python preliminary_studies/empirical_analysis/map_den_haag_stations.py
uv run python preliminary_studies/empirical_analysis/map_den_haag_pc4_timeslider.py
uv run python preliminary_studies/empirical_analysis/map_amsterdam_pc4_timeslider.py
```

If data is outside the repo, set one override for all scripts:

```bash
export DONKEY_DATA_ROOT=/path/to/snapshots
uv run python preliminary_studies/empirical_analysis/check_temporal_coverage.py
```

If you only want to override one run:

```bash
uv run python preliminary_studies/empirical_analysis/check_temporal_coverage.py --data-root /path/to/snapshots
```

To parse all available raw snapshots into processed tables in one command:

```bash
uv run python preliminary_studies/empirical_analysis/build_data_tables.py --data-root /full/path/to/snapshots
```

After preprocessing, map scripts use only `output/data` (not raw snapshots).
Optional override if processed tables are elsewhere:

```bash
uv run python preliminary_studies/empirical_analysis/map_den_haag_pc4_timeslider.py --data-dir /path/to/processed/output/data
```

## Output Structure

All generated artifacts are under `output/`:

```text
output/
  data/
    docked/
      donkey_denHaag/
        docked_YYYYMMDD.csv
      donkey_am/
        docked_YYYYMMDD.csv
    dockless/
      donkey_denHaag/
        dockless_YYYYMMDD.csv
    stations/
      donkey_denHaag/
        stations_YYYYMMDD.csv
      donkey_am/
        stations_YYYYMMDD.csv
  maps/
    den_haag_stations.html
    den_haag_pc4_timeslider.html
    amsterdam_pc4_timeslider.html
  geodata/
    pc4_den_haag.geojson
    pc4_amsterdam.geojson
  index/
    artifacts.csv
    artifacts.json
```

Category meaning:
- `docked`: station-based bike counts from `station_status`
- `dockless`: free-floating bike positions from `free_bike_status`
- `stations`: station metadata (`station_id`, `name`, `lat`, `lon`, `capacity`)

## Artifact Indexing

The map scripts rebuild the artifact index automatically at the end of each run.

You can also rebuild it manually:

```bash
uv run python preliminary_studies/empirical_analysis/artifact_index.py
```

`artifacts.csv` and `artifacts.json` include:

- `artifact_type`
- `provider`
- `city`
- `date`
- `path`
- `size_bytes`
- `modified_at_utc`
