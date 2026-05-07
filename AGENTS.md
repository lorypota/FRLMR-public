# FairMSS Agent Guide

## First section - DO NOT EDIT THIS

Read `README.md` first before making changes in this folder. It is the source of truth for folder structure, scripts, workflow, outputs, and run commands. Do not repeat README content here.

This file should only hold guidance that helps with understanding, editing or reviewing code.

### IMPORTANT NOTES - DO NOT EDIT THIS

Ask for clarification whenever you're unsure, your goal is to communicate and iterate, there will never be a perfect response, but it's important to keep improving it as much as possible. Try to be straightforward, concise but not overly. Avoid using an excessive amount of adjectives and use simple words instead of overly complex ones unless necessary. Never use dashes (—) for parenthetical statements and use ":" only if necessary, not to add emphasis. Try not to be sensationalistic but instead try to generally be humble.

I want targeted and concise changes, always look for trimmings to simplify things. Feel free to edit other sections of this document if necessary, but always tell me in chat before doing so. NEVER EDIT THIS FIRST SECTION. Before making or asking for changes, ask yourself: Why would this edit be useful to an agent working on the code or is it just repetitive and understandable from the code at first glance?

Try to understand fully requests by user and ask clarifying questions. Aim for small targeted changes ONLY ONCE you understand the final goal of user and if what you're doing would specifically answer the user question concretely.

### WRITING TIPS - DO NOT EDIT THIS

When the user makes an error or misunderstands something:

- Correct directly and clearly without being overly supportive ("No, that's incorrect" and explain the correct interpretation)
- Don't use phrases like "Good catch!", "Great question!", "You're right to question this"

Do NOT use these words unless extremely necessary and try to use simpler terms without adjectives: delve, delves, delved, delving, intricate, intricacies, underscore, underscores, underscoring, showcasing, showcases, realm, pivotal, crucial, comprehensive, meticulous, meticulously, groundbreaking, advancements, aligns, boasts, comprehending, surpassing, surpasses, emphasizing, garnered, noteworthy, notable, commendable, innovative, invaluable, versatile, potent, ingenious, landscape, unparalleled, multifaceted, nuanced, robust, streamline, transformative, leverage, harness, utilize, navigate, foster, enhance, facilitate, Furthermore, Moreover, Additionally, Notably, Importantly, tapestry, interplay, paradigm, cornerstone, holistic, synergy, ever-evolving.

For Writing Structure:
w1. State your main point first - what's your argument or finding?
w2. Put your thesis in the opening paragraph, not after background information
w3. Skip the "throughout history" or "in recent years" openings

For Supporting Arguments:
s1. Signal your evidence clearly with phrases like "This is supported by..." or "The evidence shows..."
s2. Preview your main evidence points in your introduction
s3. Your supporting points should build on each other, not stand alone as separate justifications -> VERY IMPORTANT!
s4. Avoid listing multiple independent lines of evidence that don't connect -> VERY IMPORTANT! (stems from s2.)

For Evidence and Citations:
e1. Each claim needs backing. When you cite something, explain why it matters to your argument
e2. Define specialized terms when you introduce them. If there's debate about a definition, state which one you're using and why

For Counterarguments:
c1. Address real challenges to your argument, not weak strawman versions
c2. Counterarguments should target your evidence or reasoning, not just restate an opposing view

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

## Project Overview

The project is not a general-purpose library. It is an experiment repository with code, generated arrays, pickled Q-tables, plots, and empirical-analysis utilities all living together. Most changes should preserve reproducibility and filename conventions rather than chase heavy abstraction.

## Mental Model

The simulator works at station level, but learning is organized by station category.

- Stations belong to ordered categories from remote/peripheral to central.
- A scenario selects 2, 3, 4, or 5 active categories and specifies how many stations exist in each.
- Demand is synthetic and generated per category using Skellam distributions in `common/demand.py`.
- Each day is split into two decision periods: morning and evening.
- The agent observes per-station state `[bikes, time_period]` and chooses a discrete rebalancing action.
- Demand is then rolled forward hour by hour until the next rebalancing point.
- Failures occur when demand would make station inventory negative.

Shared scenario definitions live in `common/config.py`. If a result looks surprising, start there before debugging the trainers.

## Core Modules

`common/` contains the shared simulation and learning primitives.

- `common/config.py`: scenario definitions, station-category parameters, train/eval day counts, and helper expansion logic
- `common/network.py`: builds the station graph and assigns category labels
- `common/demand.py`: generates category-wise daily demand and transforms it into event lists used by the environments
- `common/agent.py`: tabular Q-learning agent used by both formulations
- `common/av_actions.py`: valid action set logic for a state

The Q-learning design is intentionally simple:

- one tabular agent per category
- Q-values keyed by `((bikes, period), action)`
- epsilon-greedy exploration during training
- greedy evaluation by forcing epsilon to `0.0`

## Fairness Formulations

### Beta Formulation

`beta/environment.py` applies fairness directly in the reward:

- every failure is penalized
- the penalty is additionally weighted by `beta * chi`
- `chi` depends on category and is defined in `common/config.py`
- rebalancing is penalized through `gamma * phi`
- deviations from category-specific target bike levels are also penalized

`beta/config.py` defines the sweep values as `BETAS = [0.0, 0.1, ..., 1.0]`.

Training outputs are keyed by a `b...` token, for example:

- Q-tables: `beta/q_tables/cat5/seed100/q_table_b0.5_cat3.pkl`
- diagnostics: `beta/results/cat5/seed100/train_diag_b0.5.npz`
- metadata: `beta/results/cat5/seed100/meta_b0.5.json`

### CMDP Formulation

`cmdp/environment.py` separates the base reward from a Lagrangian penalty:

- base reward can optionally include a direct failure cost via `--failure-cost-coef`
- dual variables `lambda[category][period]` penalize failures in constrained categories
- thresholds come from `cmdp/config.py::compute_failure_thresholds`
- dual variables are updated every `n_dual` days using average category-period failures

This means CMDP training is non-stationary by construction. Do not assume training behavior mirrors the beta setup.

CMDP outputs are keyed by both an `r...` token and a `bf...` token, for example:

- Q-tables: `cmdp/q_tables/cat5/seed100/q_table_r0.15_bf0.0_cat4.pkl`
- diagnostics: `cmdp/results/cat5/seed100/train_diag_r0.15_bf0.0.npz`
- dual history: `cmdp/results/cat5/seed100/dual_history_r0.15_bf0.0.pkl`
- final lambdas: `cmdp/results/cat5/seed100/final_lambdas_r0.15_bf0.0.pkl`

## Scenario Definitions

The project uses a small number of fixed scenario families rather than arbitrary runtime composition.

- Supported category counts are `2`, `3`, `4`, and `5`.
- `get_scenario(...)` in `common/config.py` is the source of truth.
- Scenario data includes demand parameters, number of stations per active category, active category IDs, and occupancy target parameters.
- Category IDs are semantic and ordered. In reduced scenarios, the active set skips some intermediate labels rather than renumbering them.

That last point matters. A 3-category scenario uses active categories `[0, 2, 4]`, not `[0, 1, 2]`. Any new analysis code must respect `active_cats` and `boundaries` instead of assuming contiguous labels.

## How Experiments Are Run

Typical workflow:

1. Train policies for one formulation
2. Evaluate saved Q-tables across seeds
3. Generate plots from saved evaluation arrays

Common commands:

```bash
uv run beta/training.py --beta 0.5 --categories 5 --seed 100
uv run beta/run_training.py
uv run beta/evaluation.py --categories 5 --seeds 100 110
uv run beta/plots/generate_all.py --categories 5

uv run cmdp/training.py --r-max 0.15 --categories 5 --seed 100 --failure-cost-coef 0.0
uv run cmdp/run_training.py
uv run cmdp/evaluation.py --categories 5 --failure-cost-coef 0.0 --seeds 100 110
uv run cmdp/plots/generate_all.py --categories 5
```

If you change filenames, tokens, or directory layouts, expect downstream evaluation and plotting scripts to break.

## Research Support

`research_support/` holds work that supports the thesis narrative and data exploration.

Subareas currently visible in the repo:

- `baseline/`: earlier baseline reproduction outputs and scripts
- `failure_rate_analysis/`: scripts and arrays for comparing failure-rate behavior
- `empirical_analysis/`: utilities for inspecting data snapshots, generating maps to visualize available data and generating useful statistics on this data to ground the research approach. This folder tries to justify and guide the research approach with the available TNO data.
- `service_zone_calculation/`: scripts for deriving empirical Den Haag service zones from docked-bike station coverage and address density.
- `odin_demand_estimation/`: scripts for estimating Den Haag movement-demand rates from ODiN and mapping them to PC4, service-zone, and service-category outputs.

## Conventions To Preserve

- Prefer small, direct scripts over large framework-style refactors.
- Preserve tokenized filenames (`b...`, `r...`, `bf...`) because plotting and evaluation scripts rely on them.
- Preserve `cat{n}/seed{seed}` directory structure.
- Respect `active_cats` and `boundaries`; do not infer category slices with hard-coded indices.
- Keep comments focused on experimental logic, not boilerplate narration.
- Prefer keeping script entry points and output layout stable.
- If you change output names, paths, or generated-page behavior, check `README.md` and update it if needed.
- When doing changes, do not say "this version does this, next version will do this" unless explicitly asked. Just explain what we're currently doing and note what has to be improved/worked on. Just be direct on what is done without saying "now we do bla bla". Just say "This works by doing ..." etc. (not same wording but an example)

## Working Assumptions

- The repo is optimized for local research iteration, not packaging or deployment.
- Existing generated outputs may represent expensive prior runs.
- Some directories are large; avoid operations that rescan or rewrite them unnecessarily.
