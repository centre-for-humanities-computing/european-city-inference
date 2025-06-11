# Multi-Agent Simulation

A simple Python project to simulate multiple agents using the PyHGF model.

## 🚀 Quick Start

### 1. Install with `uv`


### 1. Run a simulation

from multi_agent_simulation import MultiAgentSimulation

# Initialize the simulation (100 time steps, 3 nodes)
sim = MultiAgentSimulation(n_steps=100, n_nodes=3)

# Run the simulation with 5 agents
sim.run_simulation(n_agents=5)

# Plot agent trajectories
sim.plot_trajectories()

⸻