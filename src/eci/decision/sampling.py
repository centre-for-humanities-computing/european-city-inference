from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _sample_choice(
    key: ArrayLike, preferences: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Sample a categorical vote per agent using softmax probabilities.

    Parameters
    ----------
    key
        JAX PRNG key.
    preferences
        Utilities score per agent and candidate.

    Returns
    -------
    vote : ArrayLike, shape (n_agents,)
        Sampled candidate index per agent.
    softmax_probs : ArrayLike, shape (n_agents, n_candidates)
        The softmax distribution.
    """
    softmax_probs = jax.nn.softmax(preferences, axis=1)
    vote = jax.random.categorical(key, preferences, axis=1)
    return vote, softmax_probs


def _sample_from_utilities(
    utilities: ArrayLike,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Sample a categorical vote per agent from utilities, optionally applying a mask.

    Parameters
    ----------
    utilities        Utilities scores for each agent-candidate pair.
    key              JAX PRNG key.
    mask             Optional boolean array of shape.

    Returns
    -------
    vote             Sampled candidate index per agent.
    softmax_probs    The softmax distribution over candidates.
    utilities        The input utilities.
    next_key         The next JAX PRNG key after splitting.
    """
    if mask is not None:
        utilities = jnp.where(mask, utilities, -jnp.inf)
    sample_key, next_key = jax.random.split(key)
    vote, softmax_probs = _sample_choice(sample_key, utilities)
    return vote, softmax_probs, utilities, next_key
