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


# TODO: Verify with a manual example that it work
# TODO: How to remove the loops for each candidate (to git the function)
def _compute_option_preferences(
    env,
    preference_means: ArrayLike,
    preference_precisions: ArrayLike,
    pref_belief_gap: ArrayLike,
) -> ArrayLike:
    """Evaluate the preference score for each possible choice.

    Parameters
    ----------
    beliefs_mean: ArrayLike
        the mean belief of agents for each preference.
    beliefs_precision: ArrayLike
        the precision belief of agents for each preference.
    pref_belief_gap: ArrayLike
        The gap between preference of agent and belief of agent for each preferences.

    Returns
    -------
    ArrayLike
        preference for each choice
    """
    candidate_preferences_per_agent = []
    candidate_list = [(c.policy["mean"], c.policy["precision"]) for c in env.candidates]

    # TODO: Changer les beliefs_mean/precision avec préférence des agents
    for mean_policy, precision_policy in candidate_list:
        # Calculate KL for all agents against this one candidate
        preference_policy_gap = kl_divergence(
            preference_means,  # preference
            preference_precisions,  # preference
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
