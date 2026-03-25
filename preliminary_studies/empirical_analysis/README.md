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
├── analysis_statistics.py       # Statistical coverage analysis
│
├── docs/
│   └── data_notes.md            # Notes on data feed and interpretation
│
├── internal/
│   ├── data_utils.py            # Raw snapshot parsing and extraction
│   ├── artifact_index.py        # Internal artifact catalog helper
│   ├── processed_data_utils.py  # CSV table loading helpers
│   ├── paths.py                 # Output directory paths
│   ├── coverage_utils.py        # Coverage computation (nearest-bike distances, aggregation)
│   ├── cbs_income.py            # CBS StatLine income/WOZ data fetcher
│   └── pc4_visualizations/      # Map visualization mode modules
│
├── output/
│   ├── data/                    # Processed CSV tables (docked, dockless, stations)
│   ├── maps/                    # Interactive HTML maps
│   ├── geodata/                 # GeoJSON files, cached geometries, CBS income data
│   ├── index/                   # Artifact indexing
│   └── analysis/                # Statistical analysis outputs by date range
│       └── <run_tag>/           #   One folder per temporal date filter
│           ├── figures/         #   PNG plots
│           ├── tables/          #   CSV/JSON summary outputs
│           └── buurt_hour_coverage/  #   Per-day buurt-hour coverage used by spatial step
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

### Statistical analysis

`uv run preliminary_studies/empirical_analysis/analysis_statistics.py` runs statistical analysis scripts in sequence.

Step-wise runs:

1. `uv run preliminary_studies/empirical_analysis/analysis_statistics.py --step temporal`: a raw daily-hour table, year-month-hour summaries, weekday/weekend comparison by year and month, and covered addresses within 500m per bike. Produces per-buurt-per-hour coverage used by the spatial step.
2. `uv run preliminary_studies/empirical_analysis/analysis_statistics.py --step spatial`: per-buurt mean docked-bike coverage correlated with demographic variables, plus Gini and Theil inequality metrics. Also includes comparisons after splitting neighborhoods into low, middle, and high thirds for a demographic variable. Produces summary tables, a car-ownership scatter plot, a choropleth map, grouped comparison tables, and one grouped boxplot.

Options for temporal step:

- `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD`: restrict analysis to a specific date range.
- `--max-workers N`: process daily temporal files in parallel.

Each run writes to its own folder under `output/analysis/`, based on the date filter. For example, `--start-date 2026-01-01 --end-date 2026-12-31` writes to `output/analysis/20260101_20261231/`. The processed dates used in that run are recorded in `tables/processed_dates.json`.

## Internal Helper Modules

These are used by the runnable scripts above and do not need to be run directly.

- `internal/data_utils.py`: shared data parsing and loading utilities.
- `internal/artifact_index.py`: internal helper that rebuilds `output/index/artifacts.csv` and `output/index/artifacts.json`.
- `internal/processed_data_utils.py`: helpers for reading processed CSV tables.
- `internal/paths.py`: shared output folder paths.
- `internal/coverage_utils.py`: coverage computation (house/buurt loading, coordinate conversion, nearest-bike distances, buurt aggregation).
- `internal/cbs_income.py`: CBS StatLine income/WOZ data fetcher (table 85618NED) with local caching.
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
