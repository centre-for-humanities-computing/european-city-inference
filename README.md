# Agent-Based Political Election Simulator

This project provides a high-performance, agent-based model (ABM) for simulating political elections. It uses the **Hierarchical Gaussian Filter (HGF)** via the `pyhgf` library to model how individual voters update their beliefs over time. The simulation core is vectorized with **JAX** for speed, allowing for large-scale experiments.

## Quick Start

### Installation

This project uses `uv` for fast package management.

```bash
# Install all dependencies from pyproject.toml
make install
```