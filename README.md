# Multi-Agent Simulation

A Python package to simulate and analyze multiple agents using the PyHGF (Hierarchical Gaussian Filter) model. This framework supports customizable simulation scenarios, including environmental shocks and trends, and allows for detailed visualization of agent behavior and belief dynamics.

## Quick Start

### Install with `uv`
You can install the project and its dependencies using uv:

```
make install
```

### Run a Simulation
```
from multi_agent_simulation import MultiAgentSimulation

# Define simulation parameters
n_steps = 100
n_nodes = 2
n_agents_per_population = 10

# Initialize the simulation
simulation = MultiAgentSimulation(
    n_steps=n_steps,
    n_nodes=n_nodes,
    n_agents_per_population=n_agents_per_population
)

# Run scenario 1 (baseline)
simulation.run_simulation(scenario=1)

# Run scenario 2 with a sudden shock
simulation.run_simulation(
    scenario=2,
    shock_pattern='sudden',
    shock_time=50,
    recovery_time=100
)

# Run scenario 2 with a trend-shaped shock (e.g., sigmoid)
simulation.run_simulation(
    scenario=2,
    shock_pattern='trend',
    recovery_time=100,
    trend_shape='sigmoid'
)simulation.plot_agent_data()
```

# Visualisation
```
# Plot agent belief trajectories
simulation.plot_trajectories()

# Plot environmental observations
simulation.plot_observations()

# Plot agent-level data: surprise and preference comparison
simulation.plot_agent_data()
```

⸻