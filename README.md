# Fairness in Reinforcement Learning for Shared bikes Rebalancing

## Project Structure

```text
FairMSS/
├── common/                        # Shared modules
│   ├── config.py                  # Scenario definitions, constants, helpers
│   ├── agent.py                   # Q-learning RebalancingAgent
│   ├── demand.py                  # Demand generation (Skellam distribution)
│   ├── network.py                 # Network graph generation
│   └── av_actions.py              # Available actions
│
├── beta/                          # Beta-weighted MDP formulation
│   ├── environment.py             # FairEnv with beta fairness parameter
│   ├── training.py                # Training script
│   ├── evaluation.py              # Evaluation across beta values
│   ├── run_training.py            # Batch training runner
│   ├── plots/                     # Plotting scripts + generated figures
│   ├── q_tables/                  # Trained Q-tables
│   └── results/                   # Evaluation outputs (.npy)
│
├── cmdp/                          # Lagrangian CMDP formulation
│   ├── environment.py             # CMDPEnv with adaptive dual variables
│   ├── training.py                # Training with Lagrangian dual updates
│   ├── evaluation.py              # Evaluation with constraint checking
│   ├── run_training.py            # Batch training runner
│   ├── plots/                     # Plotting scripts + generated figures
│   ├── q_tables/                  # Trained Q-tables
│   └── results/                   # Evaluation outputs (.npy, .pkl)
│
├── cmdp_den_haag_case/            # Empirical Den Haag CMDP case study
│   ├── config.py                  # ODiN category-period demand loader
│   ├── training.py                # Lagrangian CMDP training with empirical demand
│   ├── evaluation.py              # Evaluation with empirical demand
│   ├── run_training.py            # Batch sweep over r_max and demand scales
│   └── plots/                     # Plotting scripts + generated figures
│
├── research_support/              # Supporting studies and analyses
│   ├── baseline/                  # Reproduction of baseline (start of project)
│   ├── empirical_analysis/        # Map visualizations and statistics for TNO data
│   ├── failure_rate_analysis/     # Analysis of failure rates (start of project)
│   ├── odin_demand_estimation/    # ODiN demand-rate estimation for real-data CMDP inputs
│   └── service_zone_calculation/  # Empirical Den Haag service-zone generation
│
├── pyproject.toml                 # uv setup of dependencies, linter, dev. preferences
└── README.md                      # 👋👋
```

## Setup

1. Clone the repository
2. Install dependencies with: `uv sync` (also needed for `research_support`)

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
```

### Beta Plotting

```bash
uv run beta/plots/generate_all.py --categories 5
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
```

### CMDP Plotting

```bash
uv run cmdp/plots/generate_all.py --categories 5
uv run cmdp/plots/boxplots.py --categories 5 --save
uv run cmdp/plots/paretoplots.py --categories 5 --save
uv run cmdp/plots/lambda_convergence.py --categories 5 --save
```

## Den Haag CMDP case

Uses ODiN category-period demand rates from `research_support/odin_demand_estimation/output/category_period_demand_rates.csv` while preserving the existing CMDP constraint structure. Batch training sweeps `r_max` values and calibrated demand scales.

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
