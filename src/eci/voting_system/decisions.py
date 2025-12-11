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
    key: jax.random.PRNGKey
        The JAX Pseudo-Random Number Generator key for sampling.
    preferences: ArrayLike (float)
        The preferences for each option.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
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
    beliefs_mean: ArrayLike,
    beliefs_precision: ArrayLike,
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

    for mean_policy, precision_policy in candidate_list:
        # Calculate KL for all agents against this one candidate
        belief_policy_gap = kl_divergence(
            beliefs_mean,
            beliefs_precision,
            mean_policy.reshape(-1),
            precision_policy.reshape(-1),
        )

        # Sum over preferences
        belief_policy_gap = jnp.sum(belief_policy_gap, axis=1)

        # agent prefer choice what reduce their preference_belief gap
        preference_score_per_agent = pref_belief_gap - belief_policy_gap
        candidate_preferences_per_agent.append(preference_score_per_agent)

    # Stack the candidate scores
    return jnp.stack(candidate_preferences_per_agent).T
