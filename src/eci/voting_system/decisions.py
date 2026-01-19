import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import kl_divergence


def _sample_choice(
    key: ArrayLike, preferences: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample a choice using a softmax distribution over preferences.

    Parameters
    ----------
    key:
        The JAX Pseudo-Random Number Generator key for sampling.
    preferences:
        The preferences for each option.

    Returns
    -------
        vote: An array of chosen option.
        softmax_probs: The resulting softmax probability distribution.
    """
    # Calculate softmax per preference (for the return)
    softmax_probs = jax.nn.softmax(preferences, axis=1)

    # Sample a choice
    vote = jax.random.categorical(key, preferences, axis=1)

    return vote, softmax_probs


def _compute_option_preferences(
    env,
    preference_means: ArrayLike,
    preference_precisions: ArrayLike,
    pref_belief_gap: ArrayLike,
) -> ArrayLike:
    """Evaluate the preference score for each possible choice.

    Parameters
    ----------
    env:
        The environment object.
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
    candidate_preferences_per_agent = []
    candidate_list = [(c.policy["mean"], c.policy["precision"]) for c in env.candidates]

    # Calculate KL for all agents against candidates
    for mean_policy, precision_policy in candidate_list:
        preference_policy_gap = kl_divergence(
            preference_means,
            preference_precisions,
            mean_policy.reshape(-1),
            precision_policy.reshape(-1),
        )

        # Sum over preferences
        preference_policy_gap = jnp.sum(preference_policy_gap, axis=1)

        # agent prefer choice what reduce their preference_belief gap
        preference_score_per_agent = pref_belief_gap - preference_policy_gap
        candidate_preferences_per_agent.append(preference_score_per_agent)

    # Stack the candidate scores
    return jnp.stack(candidate_preferences_per_agent).T
