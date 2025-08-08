# voting_simulation/utils/voting_logic.py

import jax
import jax.numpy as jnp
from jax import Array, jit
from functools import partial

# We assume these types are defined elsewhere, e.g., in a typing.py file:
# from pyhgf.typing import Attributes, Edges
# For this example, we define them as 'Any'.
from typing import Any, List, Dict, Tuple
Attributes = Any
Edges = Any

from .metrics import calculate_kl_divergence

@partial(jit, static_argnames=("edges", "input_idxs", "candidates"))
def get_votes(
    key: jax.random.PRNGKey,
    attributes: Attributes,
    edges: Edges,
    node_trajectories: Dict,
    input_idxs: Tuple,
    candidates: List[Tuple[Array, Array]],
) -> Array:
    """Select a candidate based on the reduction in dissatisfaction."""
    # Get the indices of the preference nodes
    preferences_idx = [edges[idx].value_parents[0] for idx in input_idxs]

    # Get beliefs from the network (trajectories)
    expected_mean = jnp.array([node_trajectories[i]["expected_mean"][-1] for i in preferences_idx])
    expected_precision = jnp.array([node_trajectories[i]["expected_precision"][-1] for i in preferences_idx])

    # Compute the current dissatisfaction
    current_dissatisfaction = calculate_kl_divergence(
        expected_mean,
        expected_precision,
        attributes[-1]["preferences"]["mean"],
        attributes[-1]["preferences"]["precision"],
    )
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)

    # Compute each candidate’s desirability (reduction in dissatisfaction)
    candidate_desirability = []
    for candidate_mean, candidate_precision in candidates:
        expected_dissatisfaction = calculate_kl_divergence(
            expected_mean,
            expected_precision,
            candidate_mean,
            candidate_precision,
        )
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_desirability.append(total_current_dissatisfaction - total_expected_dissatisfaction)

    candidate_desirability = jnp.array(candidate_desirability)

    # Use softmax to obtain a probability distribution over choices
    # and sample a vote
    vote_probabilities = jax.nn.softmax(candidate_desirability)
    vote_log_probs = jnp.log(vote_probabilities)
    
    return jax.random.categorical(key, vote_log_probs)
