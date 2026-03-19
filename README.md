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
├── preliminary_studies/           # Supporting studies and analyses
│   ├── baseline/                  # Reproduction of baseline (start of project)
│   ├── empirical_analysis/        # Map visualizations of TNO data
│   └── failure_rate_analysis/     # Analysis of failure rates (start of project)
│
├── pyproject.toml
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies with: `uv sync` (also needed for preliminary_studies)

## Beta formulation

### Training

```bash
# Train a single configuration
uv run beta/training.py --beta 0.5 --categories 3 --seed 100

# Train all configurations
uv run beta/run_training.py
```

### Evaluation

```bash
# Evaluate a scenario (uses pre-trained Q-tables from beta/q_tables/)
uv run beta/evaluation.py --categories 3

# With detailed cost breakdowns
uv run beta/evaluation.py --categories 5 --seeds 100 110 --save-detailed
```

### Plotting

```bash
uv run beta/plots/generate_all.py --categories 5
uv run beta/plots/boxplots.py --categories 5 --save
uv run beta/plots/paretoplots.py --categories 5 --save
uv run beta/plots/learning_curves.py --categories 5 --save
```

## CMDP formulation

Replaces the fixed beta fairness weight with adaptive Lagrange multipliers that enforce explicit failure-rate constraints.

### Training

```bash
# Train a single configuration (r_max = max allowed failure rate percentage)
uv run cmdp/training.py --r-max 0.15 --categories 2 --seed 100

# Train all configurations
uv run cmdp/run_training.py
```

### Evaluation

```bash
# Evaluate a scenario
uv run cmdp/evaluation.py --categories 2

# With custom r_max values
uv run cmdp/evaluation.py --categories 2 --r-max-values 0.05 0.10 0.15 0.20 0.25
```

### Plotting

```bash
uv run cmdp/plots/generate_all.py --categories 5
uv run cmdp/plots/boxplots.py --categories 5 --save
uv run cmdp/plots/paretoplots.py --categories 5 --save
uv run cmdp/plots/lambda_convergence.py --categories 5 --save
```
