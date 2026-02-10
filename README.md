# European City Inference (ECI)

[![Tests](https://img.shields.io/badge/tests-passing-green?style=flat-square&logo=github)](https://github.com/sylvainestebe/european-city-inference/actions)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/astral-sh/ruff)

**Agent-Based Political Election Simulator using Active Inference.**

ECI is a high-performance simulation framework designed to model collective decision-making in volatile political environments. Built on top of **JAX** and **PyHGF** (Hierarchical Gaussian Filter), it simulates how thousands of voters update their beliefs and cast votes under different systems (Plurality, Quadratic Voting).

## 🚀 Key Features

* **Cognitive Agents:** Voters are not static; They employ **Bayesian belief updating** as implemented in **hierarchical predictive coding networks**, together with an **uncertainty-informed decision-making process** to select the most relevant candidate. Furthermore, they observe events per **preference dimension**.
* **JAX Accelerated:** Fully vectorized simulation logic for high-performance computing (GPU/TPU ready).
* **Multi-Voting Systems:** Compare outcomes between **Plurality Voting** and **Quadratic Voting**.

## 🛠️ Installation

This project uses `uv` for modern Python package management.

```bash
# 1. Clone the repository
git clone [https://github.com/sylvainestebe/european-city-inference.git](https://github.com/sylvainestebe/european-city-inference.git)
cd european-city-inference

# 2. Install dependencies
make install

```

## ⚡ Quick Start

Here is a minimal example to run a simulation with 200 voters and 6 candidates:

```python
import jax
from eci.environment import Environment, EnvConfig
from eci.voting_system.random_voting import _vote_random

# 1. Configure the Environment
config = EnvConfig(
    num_voters=200, 
    num_candidates=6, 
    num_preferences=4,
    seed=42
)

# 2. Initialize Simulation
env = Environment(config)

# 3. Run Simulation (100 iterations)
key = jax.random.PRNGKey(42)
results = env.run_n_simulation(_vote_random, key, n_simulations=100)

# 4. Access Data (e.g., first voter's trajectory)
print(f"Winner ID: {env.winner_id}")
print(f"Voter 0 Trajectory: {env.voters[0].trajectory}")

```

## 📂 Project Structure

```text
├── src/eci/
│   ├── agents.py           # Voter and Candidate definitions
│   ├── environment.py      # Main simulation loop
│   └── voting_system/      # Plurality, Quadratic, and Decision logic
├── notebooks/              # Interactive tutorials
├── tests/                  # Pytest suite
└── docs/                   # Documentation sources

```

## 📚 Documentation

Full documentation and tutorials are available at:

👉 **[https://sylvainestebe.github.io/european-city-inference/](https://sylvainestebe.github.io/european-city-inference/)**

## Citation

If you use this software in your research, please cite:

```bibtex
@software{political_abm2025,
  author    = {Sylvain Estebe, Nicolas Legrand},
  title     = {Collective Decision-Making in Volatile Political Environments}, 
  year      = {2025},
  url       = {[https://github.com/sylvainestebe/european-city-inference](https://github.com/sylvainestebe/european-city-inference)},
}

```