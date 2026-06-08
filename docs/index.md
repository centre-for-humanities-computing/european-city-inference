---
hide:
  - navigation
  - toc
---

**European City Inference (ECI)** simulates how thousands of voters **update their beliefs from noisy observations** (via a Hierarchical Gaussian Filter) and **cast votes** under different rules — Plurality and Quadratic Voting — letting you study how *uncertainty* shapes collective outcomes.

## Install

Get up and running using `make` and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/centre-for-humanities-computing/european-city-inference.git
cd european-city-inference
make install
```

## 30-second example

```python
import jax
from eci.environment import Environment, EnvConfig
from eci.utils import _extract_env_data_vectorized
from eci.decision import response_function
from eci.voting import _vote_plurality

# Build a population of Bayesian voters and run HGF belief inference.
env = Environment(EnvConfig(num_voters=200, num_candidates=6, num_preferences=4, seed=42))
env._run_multi_agent_inference()
data = _extract_env_data_vectorized(env)

# Run 100 plurality elections on this population.
results = env.run_n_simulation(
    _vote_plurality, data, response_function, jax.random.PRNGKey(42), n_simulations=100
)
```

Swap `_vote_plurality` for `_vote_quadratic` to compare voting rules on the *same* voters.

## Explore

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---
    Hands-on notebooks: decision making, voting systems, environment dynamics.

    [:octicons-arrow-right-24: Start here](tutorials/tutorial_1_decision_making.ipynb)

-   :material-function-variant:{ .lg .middle } **Algorithms**

    ---
    The math: belief updating, KL dissatisfaction, scoring, vote aggregation.

    [:octicons-arrow-right-24: How it works](algorithms.md)

-   :material-code-json:{ .lg .middle } **API Reference**

    ---
    Full module and function documentation for developers.

    [:octicons-arrow-right-24: Browse the API](api.md)

-   :material-book-open-page-variant:{ .lg .middle } **Glossary**

    ---
    Unsure about a term (precision, volatility, HGF…)?

    [:octicons-arrow-right-24: Definitions](glossary.md)

</div>
