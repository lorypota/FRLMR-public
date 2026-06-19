# Fairness in Reinforcement Learning for Shared bikes Rebalancing

## Project Structure

```text
FRLSR/
├── common/                        # Shared modules
│   ├── agent.py                   # Q-learning RebalancingAgent
│   ├── av_actions.py              # Available actions
│   ├── config.py                  # Scenario definitions, constants, helpers
│   ├── demand.py                  # Synthetic demand generation
│   └── network.py                 # Network graph generation
│
├── beta/                          # Beta-weighted MDP formulation
│   ├── environment.py
│   ├── evaluation.py
│   ├── run_training.py
│   ├── training.py
│   ├── plots/
│   ├── q_tables/
│   └── results/
├── cmdp/                          # Constrained MDP lagrangian formulation
│   └── ...
├── cmdp_den_haag_case/            # CMDP lagrangian empirical implementation
│   └── ...
│
├── research_support/              # Supporting studies and analyses
│   ├── baseline/                  # Reproduction of baseline (start of project)
│   ├── empirical_analysis/        # Data aggregation and visualization
│   ├── failure_rate_analysis/     # Analysis of failure rates (start of project)
│   ├── ODiN_demand_estimation/    # ODiN demand estimation for CMDP case study
│   └── service_zone_calculation/  # Service-zone generation for CMDP case study
│
├── pyproject.toml                 # uv setup of dependencies, linter, dev. preferences
└── README.md                      # 👋👋
```

## Setup

1. Clone the repository
2. Install dependencies with: `uv sync` (also needed for `research_support/`)

## Beta formulation

### Beta Training

```bash
# Train a single configuration
uv run beta/training.py --beta 0.5 --categories 3 --seed 100

# Train all configurations
uv run beta/run_training.py
```

### Beta Evaluation

```bash
# Evaluate a scenario (uses pre-trained Q-tables from beta/q_tables/)
uv run beta/evaluation.py --categories 3

# With detailed cost breakdowns
uv run beta/evaluation.py --categories 5 --seeds 100 110 --save-detailed

# Evaluate all categories
uv run beta/run_evaluation.py
```

### Beta Plotting

```bash
# All plots for every category
uv run beta/plots/generate_all.py

# Individual plots
uv run beta/plots/boxplots.py --categories 5 --save
uv run beta/plots/paretoplots.py --categories 5 --save
uv run beta/plots/learning_curves.py --categories 5 --save
```

## CMDP formulation

Replaces the fixed beta fairness weight with adaptive Lagrange multipliers that enforce explicit failure-rate constraints.

### CMDP Training

```bash
# Train a single configuration (r_max = max allowed failure rate percentage)
uv run cmdp/training.py --r-max 0.15 --categories 2 --seed 100

# Train all configurations
uv run cmdp/run_training.py
```

### CMDP Evaluation

```bash
# Evaluate a scenario
uv run cmdp/evaluation.py --categories 2

# With custom r_max values
uv run cmdp/evaluation.py --categories 2 --r-max-values 0.05 0.10 0.15 0.20 0.25

# Evaluate all categories for both failure_cost_coef values (0.0 and 1.0)
uv run cmdp/run_evaluation.py
```

### CMDP Plotting

```bash
# All plots for every category and both bf values (0.0 and 1.0)
uv run cmdp/plots/generate_all.py

# Individual plots
uv run cmdp/plots/boxplots.py --categories 5 --save
uv run cmdp/plots/paretoplots.py --categories 5 --save
uv run cmdp/plots/lambda_convergence.py --categories 5 --save
```

## Den Haag CMDP case

Use case to ground CMDP approach on models estimated with available real-data of Den Haag:

- ODiN category-period demand rates (`research_support/odin_demand_estimation/`);
- Donkey stations with capacity and inventory (`research_support/empirical_analysis/output/data`);
- service zones (`research_support/service_zone_calculation`).

### Den Haag CMDP Training

```bash
# Train a single configuration
uv run cmdp_den_haag_case/training.py --r-max 0.01 --seed 100 --demand-scale 0.01

# Train all configurations
uv run cmdp_den_haag_case/run_training.py
```

### Den Haag CMDP Evaluation

```bash
# Evaluate trained policies
uv run cmdp_den_haag_case/evaluation.py

# With selected demand scales
uv run cmdp_den_haag_case/evaluation.py --demand-scales 0.005 0.01 0.02
```

### Den Haag CMDP Plotting

```bash
uv run cmdp_den_haag_case/plots/generate_all.py
uv run cmdp_den_haag_case/plots/generate_all.py --demand-scales 0.005 0.01
uv run cmdp_den_haag_case/plots/demand_scale_comparison.py --save
```
