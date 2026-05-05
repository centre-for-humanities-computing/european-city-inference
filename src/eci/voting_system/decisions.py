import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.voting_system.beliefs import (
    _get_belief_preference_gap,
    _get_pref_candidate_gap,
)


def _sample_choice(
    key: ArrayLike, preferences: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample a choice using a softmax probabilities over preferences.

    Parameters
    ----------
    key:
        Pseudo-Random Number Generator key for sampling.
    preferences:
        The preferences for each option.

    Returns
    -------
        vote: An array containing the chosen option.
        softmax_probs: The resulting softmax probability distribution.
    """
    # Calculate softmax per preference
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
        Dictionary containing preference_beliefs_gap and preference_candidate_gap.

    Returns
    -------
        preference for each choice
    """
    # Compute the gap between belief and preference, and candidate and preference
    belief_preference_gap = _get_belief_preference_gap(data)

    # Compute the gap between candidate and preference
    pref_candidate_gap = _get_pref_candidate_gap(data)

    # Compute the preference score for each candidate.
    preference_score_per_agent = response_function(
        belief_preference_gap, pref_candidate_gap
    )

    return preference_score_per_agent, pref_candidate_gap, belief_preference_gap


def response_function(belief_preference_gap, pref_candidate_gap):
    """Compute the response for the agent.

    Parameters
    ----------
    belief_preference_gap: The KL divergence between beliefs and preferences.
    pref_candidate_gap: The KL divergence between candidate policies and preferences.

    Returns
    -------
        preference for each choice
    """
    # Response function options one

    # KL(BELIEF || PREF) - KL(POLICY || PREF)
    # preference_score_per_agent =
    # belief_preference_gap[:, jnp.newaxis] - pref_candidate_gap

    # Response function options two

    # KL(POLICY || PREF) / KL(BELIEF || PREF)
    # preference_score_per_agent =
    # pref_candidate_gap - belief_preference_gap[:, jnp.newaxis]

    # Response function options three

    # (KL(POLICY || PREF) - KL(BELIEF || PREF)) / KL(BELIEF || PREF)
    preference_score_per_agent = (
        pref_candidate_gap - belief_preference_gap[:, jnp.newaxis]
    ) / belief_preference_gap[:, jnp.newaxis]

    return preference_score_per_agent
