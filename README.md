Of course. Here is a revised `README.md` that accurately reflects the classes and functionality in the code you provided, focusing on the voting simulation.

-----

# Agent-Based Political Election Simulator 🗳️

This project provides a high-performance, agent-based model (ABM) for simulating political elections. It uses the **Hierarchical Gaussian Filter (HGF)** via the `pyhgf` library to model how individual voters update their beliefs over time. The simulation core is vectorized with **JAX** for exceptional speed, allowing for large-scale experiments.

## Key Features

  * **Agent-Based Framework:** Simulates distinct **Voter** and **Candidate** agents, each with unique preferences and policy platforms.
  * **High-Performance Core:** The voting and belief-update logic is JIT-compiled and vectorized using JAX, enabling thousands of agents to be simulated efficiently.
  * **Flexible Election Rules:** Easily swap out different voting systems, including:
      * Plurality Voting (First-Past-the-Post)
      * Ranking Voting (with Borda Count)
      * Support for single-round or two-round runoff elections.
  * **Rich Data Analysis:** A comprehensive `DataCollector` and `SimulationVisualizer` are included to analyze election outcomes, voter preferences, and individual belief trajectories.

-----

## Quick Start

### Installation

This project uses `uv` for fast package management.

```bash
# Install all dependencies from pyproject.toml
make install
```

### Running a Simulation

The following example sets up and runs a 5-step, two-round election with 50 voters, 4 candidates, and a Plurality voting system.

```python
# main.py

from environment import Environment
from voting_systems import PluralityVoting
from analysis import SimulationVisualizer

# 1. Configure the simulation
NUM_VOTERS = 50
NUM_CANDIDATES = 4
NUM_PREFERENCES = 2
NUM_STEPS = 5

# 2. Initialize the environment with a chosen voting system
plurality_system = PluralityVoting()
env = Environment(
    num_voters=NUM_VOTERS,
    num_candidates=NUM_CANDIDATES,
    num_preferences=NUM_PREFERENCES,
    voting_system=plurality_system,
    rounds=2  # Use a two-round election
)

# 3. Run the simulation
env.run(num_steps=NUM_STEPS)

# 4. Get the results
results_df = env.datacollector.get_long_dataframe()
print("Simulation Results:")
print(results_df.tail())
```

-----

## Visualization 📊

The `SimulationVisualizer` provides several methods to explore the simulation results.

```python
# --- Continuing the script from above ---

# 5. Initialize the visualizer
visualizer = SimulationVisualizer(env, env.datacollector)


# Plot the distribution of preferences for candidates and a sample of voters
visualizer.plot_preference_distributions()

# Plot the belief trajectory for a specific voter (e.g., the first one)
first_voter = env.voters[0]
visualizer.plot_belief_trajectory(voter=first_voter)

# Plot the distribution of vote proportions across all simulation steps
visualizer.plot_simulation_results_distribution(plot_kind='histogram')
```
