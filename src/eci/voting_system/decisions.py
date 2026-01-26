import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.voting_system.beliefs import _get_pref_belief_gap, _get_pref_candidate_gap


def _sample_choice(
    key: ArrayLike, preferences: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample a choice using a softmax probabilities over preferences.

    Parameters
    ----------
    key:
        The JAX Pseudo-Random Number Generator key for sampling.
    preferences:
        The preferences for each option.

    Returns
    -------
        vote: An array containing the chosen option.
        softmax_probs: The resulting softmax probability distribution.
    """
    # Calculate softmax per preference (for the return)
    softmax_probs = jax.nn.softmax(preferences, axis=1)

    # Sample a choice
    vote = jax.random.categorical(key, preferences, axis=1)

    return vote, softmax_probs


def _compute_preferences(
    data: dict,
) -> ArrayLike:
    """Evaluate the preference score for each possible choice.

    Parameters
    ----------
    data:
        The dictionary returned by _extract_env_data_vectorized containing
    preference_means:
        the mean preference of agents for each preference.
    preference_precisions:
        the precision preference of agents for each preference.
    pref_belief_gap:
        The gap between preference of agent and belief of agent for each preferences.

    Returns
    -------
        preference for each choice
    """
    # Shape: (n_agents,)
    pref_belief_gap = _get_pref_belief_gap(data)

    # Shape: (n_agents, n_candidates)
    pref_candidate_gap = _get_pref_candidate_gap(data)

    # Pour soustraire, on aligne les dimensions :
    # (n_agents, 1) - (n_agents, n_candidates) -> (n_agents, n_candidates)
    preference_score_per_agent = pref_belief_gap[:, jnp.newaxis] - pref_candidate_gap

    # Stack the candidate scores
    return preference_score_per_agent
