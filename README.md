# Agent-Based Political Election Simulator

This project provides an agent-based model (ABM) for simulating political elections. It uses the **Hierarchical Gaussian Filter (HGF)** via the `pyhgf` library to model how individual voters update their beliefs over time.

## Project Structure

```text
├── src/
│   ├── eci/
│   │   └── voting_system/       # Core voting logic and mechanisms
│   │       ├── beliefs.py       # Handling agent beliefs and preferences
│   │       ├── decision.py      # Decision-making algorithms
│   │       ├── plurality.py     # Plurality voting implementation
│   │       ├── quadratic.py     # Quadratic voting implementation
│   │       └── random_voting.py # Randomized voting for baseline comparisons
│   ├── adapter.py               # Adapters for data transformation/interfaces
│   ├── agents.py                # Agent definitions and behaviors
│   ├── environment.py           # Simulation environment logic
│   ├── utils.py                 # Helper functions and shared utilities
│   └── visualizer.py            # Tools for rendering simulation results
├── notebooks/                   # Tutorials
├── tests/                       
├── pyproject.toml               # Project configuration and dependencies (uv)
└── Makefile                     # Automation commands
## Quick Start

### Installation

This project uses `uv` for package management.

```bash
# Install all dependencies from pyproject.toml
make install
```

## How it Work

You can use notebook: tutorial_1 to understand how the decision making process work and tutorial 2 to understand how it work under different agents. 

### 1. Create the Environment

The `Environment` is the container for all agents (Voters and Candidates).

**Code:**

```python
from eci.environment import Environment

# 1. Define the dimensions of the simulation
NUM_VOTERS = 200        # Total number of voters
NUM_CANDIDATES = 6      # Total number of candidates running
NUM_PREFERENCES = 6     # Number of policy dimensions (e.g., Economy, Social, etc.)

# Set the number of simulation
NUM_SIMULATIONS = 100
env.num_simulations = NUM_SIMULATIONS

# 2. Initialize the environment object
env = Environment(
    num_voters=NUM_VOTERS,
    num_candidates=NUM_CANDIDATES,
    num_preferences=NUM_PREFERENCES,
)

```

### 2. Configure Agents (Voters & Candidates)

Once the environment is created, you can change the "preference" and "policies" of the agents. 

* **Candidates:** Defined by their policy stance (`mean`) and how precise they are (`precision`).
* **Voters:** Defined by their preference (`mean`) and how strongly they hold those preferences (`precision`).

**Code Example:**

```python
import jax.numpy as jnp

# We loop through the first 5 voters and give them specific preferences
for i in range(5):
    env.voters[i].preferences["mean"] = jnp.array([-1.0, -1.0, -1.0, -1.0])
    
    # Higher value = higher certainty
    env.voters[i].preferences["precision"] = jnp.array([0.4, 0.2, 0.6, 0.2])

# Configure a Candidate
env.candidates[0].policy["mean"] = jnp.array([1, 1, 1, 1])
env.candidates[0].policy["precision"] = jnp.array([0.4, 0.4, 0.4, 0.4])

```

### 3. Launch Simulation

To run a simulation, you must initialize the agents' internal models.

```python
import jax
from eci.voting_system.random import _vote_random

# 1. Initialize the HGF network for all agents
env.initialize_network()

# 2. Initialize a JAX PRNGKey for reproducibility
key = jax.random.PRNGKey(42)

# 3. Run N simulations with a specific voting system
# The simulator supports: _vote_plurality, _vote_quadratic, and _vote_random
sim = env.run_n_simulation(_vote_random, key, NUM_SIMULATIONS)

# 4. Update the agents' internal states based on simulation results
env._update_agents()

# 5. Create a DataFrame for analysis and plotting
env.df = env.create_data_frame()

```

The simulator supports various ways to aggregate agent decisions:

* **Plurality Voting (`_vote_plurality`)**: Standard winner-take-all system.
* **Quadratic Voting (`_vote_quadratic`)**: Allows voters to express the intensity of their preferences.
* **Random Voting (`_vote_random`)**: Used as a baseline for comparison.

---

### 4. Plot Results 

The simulation setup involves defining how many time steps (or trajectories) the agents will experience. You also initialize the **Adapter** (handles data) and **Visualizer** (handles plotting).

**Code:**


```python
from eci.adapter import SimulationAdapter
from eci.visualizer import SimulationVisualizer

# 2. Initialize helpers
viz = SimulationVisualizer()  # For plotting graphs
adapter = SimulationAdapter() # For processing simulation data

```

## 📚 Documentation and Tutorials

> **Note:** Full official documentation and detailed interactive tutorials are currently being drafted and will be available very soon to guide you.
> 

```

If you use this code or data in a scientific publication, please cite:

@software{political_abm2025,
  author    = {Sylvain Estebe, Nicolas Legrand},
  title     = {collective decision-making in volatile political environments using quadratic voting}, 
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/SylvainEstebe/votre_repo},
}
