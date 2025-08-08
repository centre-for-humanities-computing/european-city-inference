import jax
import jax.numpy as jnp
from jax import Array, jit
from pyhgf.typing import Attributes, Edges
from jax.typing import ArrayLike
from functools import partial

def calculate_kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> Array:
    """Calculate the KL divergence between two Gaussian distributions.

    Parameters
    ----------
    mu_belief :
        Mean of the belief distribution.
    prec_belief :
        Precision of the belief distribution.
    mean_pref :
        Mean of the preference distribution.
    precision_pref :
        Precision of the preference distribution.

    Returns
    -------
        KL divergence.
    """
    # Convert precision to variance
    var_belief = 1 / precision_belief
    var_pref = 1 / precision_pref

    # Calculate KL divergence using the analytical formula for Gaussian distributions
    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )
    return kl


@partial(jit, static_argnames=("edges", "input_idxs"))
def get_votes(
    key: jax.random.PRNGKey,
    attributes: Attributes,
    edges: Edges,
    node_trajectories: dict,
    input_idxs: tuple,
    candidates: list,
) -> Array:
    """Get votes based on network attributes and input data.

    Parameters
    ----------
    key :
        Random key for JAX.
    Network:
        The network containing the attributes and node trajectories.
    candidates :
        List of candidate preferences.

    Returns
    -------
        The index of the selected candidate.

    """
    # get continuous nodes matching preferences
    preferences_idx = [
        edges[idx].value_parents[0] for idx in input_idxs
    ]

    # Get the beliefs from the network
    expected_mean = jnp.array(
        [node_trajectories[i]["expected_mean"][-1] for i in preferences_idx]
    )
    expected_precision = jnp.array(
        [
            node_trajectories[i]["expected_precision"][-1]
            for i in preferences_idx
        ]
    )

    # compute the dissatisfaction based on the current beliefs
    current_dissatisfaction = calculate_kl_divergence(
        expected_mean,
        expected_precision,
        attributes[-1]["preferences"]["mean"],
        attributes[-1]["preferences"]["precision"],
    )
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)

    # Initialize a list to store the voting decisions for each candidate
    candidate_preferences = []

    # For each candidate, calculate the expected dissatisfaction in a vectorized manner
    for candidate in candidates:

        candidate_mean_pref, candidate_precision_pref = candidate

        # Calculate the expected dissatisfaction for this candidate
        expected_dissatisfaction = calculate_kl_divergence(
            expected_mean,
            expected_precision,
            candidate_mean_pref,
            candidate_precision_pref
        )
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_preferences.append(
            total_current_dissatisfaction - total_expected_dissatisfaction
        )

    # Convert candidate_preferences to JAX array
    candidate_preferences = jnp.array(candidate_preferences)

    # Softmax of candidate_preferences to get probabilities
    softmax_probs = jax.nn.softmax(candidate_preferences)

    # Log of softmax_probs for the voting distribution
    log_softmax_probs = jnp.log(softmax_probs)
    vote = jax.random.categorical(key, log_softmax_probs)

    return vote
