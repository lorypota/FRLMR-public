# Empirical Analysis Guide

## First section - DO NOT EDIT THIS

This file is in part a duplication of AGENTS.md in root because some agents might be opened directly in this folder and need this specific information without the need to read the previous file.

### IMPORTANT NOTES - DO NOT EDIT THIS

Ask for clarification whenever you're unsure, your goal is to communicate and iterate, there will never be a perfect response, but it's important to keep improving it as much as possible. Try to be straightforward, concise but not overly. Avoid using an excessive amount of adjectives and use simple words instead of overly complex ones unless necessary. Never use dashes (—) for parenthetical statements and use ":" only if necessary, not to add emphasis. Try not to be sensationalistic but instead try to generally be humble.

I want targeted and concise changes, always look for trimmings to simplify things. Feel free to edit other sections of this document if necessary, but always tell me in chat before doing so. NEVER EDIT THIS FIRST SECTION. Before making or asking for changes, ask yourself: Why would this edit be useful to an agent working on the code?

### WRITING TIPS - DO NOT EDIT THIS

When the user makes an error or misunderstands something:

- Correct directly and clearly without being overly supportive
- Don't use phrases like "Good catch!", "Great question!", "You're right to question this"
- Simply state "No, that's incorrect" and explain the correct interpretation

Do NOT use these words unless extremely necessary and try to use simpler terms without adjectives: delve, delves, delved, delving, intricate, intricacies, underscore, underscores, underscoring, showcasing, showcases, realm, pivotal, crucial, comprehensive, meticulous, meticulously, groundbreaking, advancements, aligns, boasts, comprehending, surpassing, surpasses, emphasizing, garnered, noteworthy, notable, commendable, innovative, invaluable, versatile, potent, ingenious, landscape, unparalleled, multifaceted, nuanced, robust, streamline, transformative, leverage, harness, utilize, navigate, foster, enhance, facilitate, Furthermore, Moreover, Additionally, Notably, Importantly, tapestry, interplay, paradigm, cornerstone, holistic, synergy, ever-evolving.

For Writing Structure:

1. State your main point first - what's your argument or finding?
2. Put your thesis in the opening paragraph, not after background information
3. Skip the "throughout history" or "in recent years" openings

For Supporting Arguments: 4. Signal your evidence clearly with phrases like "This is supported by..." or "The evidence shows..." 5. Preview your main evidence points in your introduction 6. Your supporting points should build on each other, not stand alone as separate justifications -> VERY IMPORTANT! 7. Avoid listing multiple independent lines of evidence that don't connect -> VERY IMPORTANT! (stems from 6.)

For Evidence and Citations: 8. Each claim needs backing. When you cite something, explain why it matters to your argument 9. Define specialized terms when you introduce them. If there's debate about a definition, state which one you're using and why

For Counterarguments: 10. Address real challenges to your argument, not weak strawman versions 11. Counterarguments should target your evidence or reasoning, not just restate an opposing view

Writing examples:
Bad Opening:
"Research into climate patterns has been conducted for decades. Scientists have used various methods to study temperature changes. This paper will explore some of these findings..."
Good Openings:
"Global temperatures have risen 1.1°C since pre-industrial times, primarily due to human CO2 emissions. This conclusion rests on three converging lines of evidence: ice core data, satellite measurements, and atmospheric modeling."
"The peer review system improves research quality less than commonly assumed. While it catches obvious errors, studies show it fails to detect more serious methodological flaws, and may actually slow scientific progress."

Bad Counterargument:
"Some might disagree with this interpretation, but the data clearly supports my view."
Good Counterargument:
"However, Smith (2022) argues that satellite measurements may contain systematic errors due to orbital decay. If true, this would reduce the confidence interval for temperature estimates by approximately 0.2°C."

Good Response:
"Yet orbital decay corrections have been standard practice since 2015 (Jones et al., 2015), and are already incorporated into the datasets cited."

### Use UV - DO NOT EDIT THIS

Use `uv` for anything related to this project.

- Create or sync the environment with `uv sync` (only run this if necessary).
- Run scripts with `uv run ...`.
- Add or update dependencies with `uv add ...` or `uv remove ...`.
- Run tooling with `uv run ruff check .` and `uv run ruff format .`.
- Do not introduce ad hoc `pip install`, bare `python script.py`.

## Scope Of This Folder

This folder is a small data-processing and map-building subproject inside `FairMSS`.
The purpose here is:

- inspect raw Donkey GBFS snapshot archives
- check temporal coverage of the raw dataset
- convert raw snapshot archives into processed day-level CSV tables
- build interactive Folium maps for Den Haag
  (I initially researched also Amsterdam but now I want to focus on Den Haag)
- keep an index of generated artifacts under `output/`

This folder is script-first. It is not a packaged library. Most changes should keep the current script entry points and output layout stable.

## Mental Model

There are two main data layers in this folder:

1. Raw snapshots
2. Processed outputs

Raw snapshots live outside or beside the repo in a date tree like:

```text
<data-root>/
  YYYY/MM/DD/HH/<provider>_fietsData_YYYYMMDDHHMM.tar.gz
```

Processed outputs live in this folder under `output/`:

```text
output/
  data/
  maps/
  geodata/
  index/
```

The normal flow is:

1. inspect raw snapshots if needed
2. check coverage if needed
3. build processed CSV tables from raw data
4. build maps from processed CSV tables
5. rebuild or verify the artifact index

Do not mix raw-data assumptions with processed-data assumptions. The map scripts mostly read `output/data`, not tar archives directly.

## Main Scripts

Top-level runnable scripts:

- `inspect_snapshot.py`: opens one raw snapshot archive and prints a structured summary
- `check_temporal_coverage.py`: scans the raw data tree and reports dates, providers, and missing minutes
- `build_data_tables.py`: converts raw snapshots into day-level docked, dockless, and station CSVs
- `map_den_haag_stations.py`: builds a station map for Den Haag from processed station metadata
- `map_den_haag_pc4.py`: builds the richer Den Haag PC4 map with date and hour controls and multiple visualization modes
- `map_amsterdam_pc4.py`: builds the Amsterdam PC4 map from processed station and docked data
- `artifact_index.py`: rebuilds `output/index/artifacts.csv` and `output/index/artifacts.json`

These scripts are the public interface of this folder. Prefer keeping them simple and readable over abstracting logic too early.

## Internal Modules

Helper code lives under `internal/`.

- `internal/data_utils.py`: raw snapshot parsing, tar extraction, provider constants, bbox filtering, and day-level table loading from raw archives
- `internal/processed_data_utils.py`: discovery and loading helpers for processed CSV tables
- `internal/paths.py`: output paths and default raw data root
- `internal/pc4_visualizations/`: visualization-mode registry and JavaScript builders used by the Den Haag PC4 map

Important split:

- `data_utils.py` deals with raw archives and parsing
- `processed_data_utils.py` deals with already-built CSV tables

Keep that split clean. Do not turn processed-data helpers into raw-data readers or the other way around unless there is a strong reason.

## Data Pipeline Rules

### Processed Tables

`build_data_tables.py` writes day-level CSVs under:

- `output/data/docked/<provider>/`
- `output/data/dockless/<provider>/`
- `output/data/stations/<provider>/`

Naming matters. The rest of the folder expects filenames like:

- `docked_YYYYMMDD.csv`
- `dockless_YYYYMMDD.csv`
- `stations_YYYYMMDD.csv`

`processed_data_utils.py` also supports some older legacy names. Do not remove that compatibility lightly.

## Den Haag PC4 Map

`map_den_haag_pc4.py` is the richest script in this folder.

It:

- loads processed station, docked, and dockless data
- downloads or reuses cached PC4 polygons
- spatially joins stations and bikes to PC4 areas
- builds a Folium map with custom HTML, CSS, and JavaScript
- supports multiple visualization modes through `internal/pc4_visualizations/`
- supports compare mode and a more complex control layout

This file is large and UI-heavy. Before editing it:

- decide whether the change belongs in the top-level script or in `internal/pc4_visualizations/`
- avoid mixing data loading changes with UI refactors in one patch
- keep the generated page behavior stable unless the user asked for a visible UI change

## Output Rules

Generated artifacts in this folder are real outputs, not scratch files.

Important outputs:

- processed CSV tables in `output/data/`
- HTML maps in `output/maps/`
- cached PC4 geojson files in `output/geodata/`
- artifact indexes in `output/index/`

Do not rename or relocate these outputs without updating all readers and the README.
