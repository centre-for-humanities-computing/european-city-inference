C'est une excellente idée. Le `README.md` et le `index.md` ont des rôles différents :

* **README.md (GitHub)** : Doit être concis, technique et convaincre le développeur de mettre une étoile ⭐ ou de cloner le repo.
* **docs/index.md (Site Web)** : Doit être accueillant, visuel et guider l'utilisateur (débutant ou expert) vers la bonne section.

Voici les versions "Pro" rédigées en anglais (standard open-source) pour ces deux fichiers.

---

### 1. Le `README.md` (Pour la racine du projet)

Ce fichier est optimisé pour GitHub : badges, installation rapide, et citation académique.

```markdown
# European City Inference (ECI)

[![Tests](https://img.shields.io/badge/tests-passing-green?style=flat-square&logo=github)](https://github.com/sylvainestebe/european-city-inference/actions)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/astral-sh/ruff)

**Agent-Based Political Election Simulator using Active Inference.**

ECI is a high-performance simulation framework designed to model collective decision-making in volatile political environments. Built on top of **JAX** and **PyHGF** (Hierarchical Gaussian Filter), it simulates how thousands of voters update their beliefs and cast votes under different systems (Plurality, Quadratic Voting).

## 🚀 Key Features

* **Cognitive Agents:** Voters are not static; they use Active Inference to minimize surprise and update beliefs based on candidate policies.
* **JAX Accelerated:** Fully vectorized simulation logic for high-performance computing (GPU/TPU ready).
* **Multi-Voting Systems:** Compare outcomes between **Plurality Voting** and **Quadratic Voting**.
* **Polarization Metrics:** Built-in tools to measure and visualize social polarization and dissatisfaction.

## 🛠️ Installation

This project uses `uv` for modern Python package management.

```bash
# 1. Clone the repository
git clone [https://github.com/sylvainestebe/european-city-inference.git](https://github.com/sylvainestebe/european-city-inference.git)
cd european-city-inference

# 2. Install dependencies (via Makefile)
make install

```

## ⚡ Quick Start

Here is a minimal example to run a simulation with 200 voters and 6 candidates:

```python
import jax
from eci.environment import Environment
from eci.voting_system.random_voting import _vote_random

# Initialize Environment
env = Environment(num_voters=200, num_candidates=6, num_preferences=4)
env.initialize_network()

# Run Simulation (100 iterations)
key = jax.random.PRNGKey(42)
results = env.run_n_simulation(_vote_random, key, n_simulations=100)

# Analyze Results
env._update_agents()
df = env.create_data_frame()
print(df.head())

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

👉 **[https://sylvainestebe.github.io/european-city-inference/](https://www.google.com/search?q=https://sylvainestebe.github.io/european-city-inference/)**

## citation

If you use this software in your research, please cite:

```bibtex
@software{political_abm2025,
  author    = {Sylvain Estebe, Nicolas Legrand},
  title     = {Collective Decision-Making in Volatile Political Environments}, 
  year      = {2025},
  url       = {[https://github.com/sylvainestebe/european-city-inference](https://github.com/sylvainestebe/european-city-inference)},
}

```