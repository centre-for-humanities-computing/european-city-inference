# European City Inference (ECI)

[![CI](https://github.com/centre-for-humanities-computing/european-city-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/centre-for-humanities-computing/european-city-inference/actions/workflows/ci.yml)
[![Docs](https://github.com/centre-for-humanities-computing/european-city-inference/actions/workflows/documentation.yml/badge.svg)](https://centre-for-humanities-computing.github.io/european-city-inference/)
[![Python](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/astral-sh/ruff)

**Agent-based political election simulator using Hierarchical Gaussian Filter (HGF) and JAX.**

ECI models thousands of voters as **predictive-coding agents** — each one
runs a Hierarchical Gaussian Filter over a stream of noisy world
observations, then casts a vote whose decisiveness is shaped by their
posterior precision. The package compares how different voting rules
(plurality, quadratic voting, …) aggregate those precision-weighted
ballots into a collective outcome.

---

## Why should you care?

ECI turns verbal hypotheses about voter behaviour into formal models. Three things it lets you do:

- **Sweep parameters at scale.** Vary belief precision, world
  volatility, electorate size, and watch how the winning candidate
  shifts under each voting rule.
- **Compare voting rules on the same population.** Same voters, same
  beliefs — only the aggregation changes. Disagreements between rules
  become directly attributable to the rule, not to the data.
- **Calibrate against real experiments** *(coming in v0.2)*. Fit ECI
  parameters to ballots collected from real participants.

---

## Install

ECI uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/centre-for-humanities-computing/european-city-inference.git
cd european-city-inference
make install
```

That command installs `uv` (if missing), creates a `.venv/`, and
installs ECI in editable mode with all dev dependencies.

For Jupyter:

```bash
make jupyterlab
```

---

## A 30-second example

```python
import jax
from eci.environment import Environment, EnvConfig
from eci.utils import _extract_env_data_vectorized
from eci.decision import response_function
from eci.voting import _vote_plurality

# 1. Configure the population.
config = EnvConfig(
    num_voters=200,
    num_candidates=6,
    num_preferences=4,
    seed=42,
)

# 2. Build the environment and run HGF belief inference for every voter.
env = Environment(config)
env._run_multi_agent_inference()

# 3. Extract a vectorised view of the agent state.
data = _extract_env_data_vectorized(env)

# 4. Run 100 plurality elections on this population.
key = jax.random.PRNGKey(42)
results = env.run_n_simulation(
    _vote_plurality, data, response_function, key, n_simulations=100
)
```

Replace `_vote_plurality` with `_vote_quadratic` for quadratic voting,
or pass your own [response function](https://centre-for-humanities-computing.github.io/european-city-inference/extending_response_functions/)
to model a different decision rule.

---

## Project layout

```text
src/eci/
├── agents.py            # Voter / Candidate dataclasses
├── environment.py       # EnvConfig + Environment (HGF wiring)
├── perceptual.py        # HGF perceptual model wrapper
├── population.py        # voter / candidate parameter sampling
├── world.py             # observation-stream generation
├── observations.py      # synthetic observation generators
├── metrics.py           # collective-outcome metrics
├── utils.py             # KL divergence + env-extraction helpers
├── decision/            # response functions + scoring + utilities
├── voting/              # voting rules (plurality, quadratic)
└── plots/               # belief / preference / voting plot helpers
notebooks/               # six didactic tutorials + poster figures
tests/                   # pytest suite (mirrors src/eci/)
docs/                    # MkDocs Material site
```

---

## Documentation

Full docs, tutorials and API reference:
**<https://centre-for-humanities-computing.github.io/european-city-inference/>**

- [Tutorial 1 — Decision making](https://centre-for-humanities-computing.github.io/european-city-inference/tutorials/tutorial_1_decision_making/)
- [Tutorial 2 — Voting systems (plurality vs quadratic)](https://centre-for-humanities-computing.github.io/european-city-inference/tutorials/tutorial_2_voting_system/)
- [Tutorial 3 — Environment & world dynamics](https://centre-for-humanities-computing.github.io/european-city-inference/tutorials/tutorial_3_environment/)
- [Write your own response function](https://centre-for-humanities-computing.github.io/european-city-inference/extending_response_functions/)

---

## Contributing

We welcome bug reports, feature requests and pull requests.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow
and [`CHANGELOG.md`](CHANGELOG.md) for what's new.

---

## Citation

If you use ECI in academic work, please cite:

```bibtex
@software{estebe_legrand_eci_2026,
  author    = {Estebe, Sylvain and Legrand, Nicolas},
  title     = {ECI: European City Inference — Agent-based political election
               simulator with hierarchical Bayesian voter beliefs},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/centre-for-humanities-computing/european-city-inference},
  note      = {Add Zenodo DOI here once the v0.1.0 release is archived.}
}
```

A Zenodo DOI will replace the `note` field as soon as the first tagged
release is archived — see [`CHANGELOG.md`](CHANGELOG.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
