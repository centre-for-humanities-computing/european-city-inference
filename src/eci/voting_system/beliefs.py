import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import (
    kl_divergence,
)


def _extract_candidates_data(env) -> tuple[ArrayLike, ArrayLike]:
    """
    Extract and stack policy data for all candidates.

    Parameters
    ----------
    env :
        The environment object containing the list of candidates.

    Returns
    -------
    candidates_mean_stack :
        the mean policy values for all candidates.
    candidates_precision_stack : ArrayLike
        the precision policy values for all candidates.
    """
    # Initialize lists
    c_means = []
    c_precisions = []

    # Loop through candidates and extract policy data
    for c in env.candidates:
        c_means.append(c.policy["mean"].reshape(-1))
        c_precisions.append(c.policy["precision"].reshape(-1))

    # Stack the lists into arrays
    candidates_mean_stack = jnp.array(c_means)
    candidates_precision_stack = jnp.array(c_precisions)

    return candidates_mean_stack, candidates_precision_stack


def _get_current_beliefs(env) -> dict:
    """
    Extract current beliefs, preferences, and candidate policies for all agents.

    Parameters
    ----------
    env :
        The environment object.

    Returns
    -------
    all_agent_data :
        dictionary containing beliefs, preferences, and policy data.
    """
    # Initialize dictionary to hold all agent data
    all_agent_data = {}

    # Extract candidate policy data
    policy_mean, policy_prec = _extract_candidates_data(env)

    # Loop through each agent
    for agent in range(len(env.voters)):
        means_belief_by_preference = []
        precisions_belief_by_preference = []
        agent_pref_means = []
        agent_pref_precisions = []
        # Loop through each preference index for the agent
        for preference_idx in env.preferences_idx:
            means_belief_by_preference.append(
                env.last_attributes[preference_idx]["expected_mean"][agent]
            )
            precisions_belief_by_preference.append(
                env.last_attributes[preference_idx]["expected_precision"][agent]
            )
            agent_pref_means.append(
                env.last_attributes[-1]["preferences"]["mean"][agent][preference_idx]
            )
            agent_pref_precisions.append(
                env.last_attributes[-1]["preferences"]["precision"][agent][
                    preference_idx
                ]
            )

        all_agent_data[agent] = {
            "means_belief": jnp.array(means_belief_by_preference),
            "precisions_belief": jnp.array(precisions_belief_by_preference),
            "means_preference": jnp.array(agent_pref_means),
            "precision_preference": jnp.array(agent_pref_precisions),
            "mean_policy": jnp.array(policy_mean),
            "precision_policy": jnp.array(policy_prec),
        }

    return all_agent_data


def _get_pref_belief_gap(all_agent_data: dict) -> ArrayLike:
    """
    Compute the gap between current beliefs and preferences for each agent.

    Parameters
    ----------
    all_agent_data :

    Returns
    -------
    pref_belief_gap :
        The computed preference-belief gap for each agent.
    """
    # Extract and stack arrays from each agent
    beliefs_mean = jnp.stack(
        [agent_data["means_belief"] for agent_data in all_agent_data.values()]
    )
    beliefs_precision = jnp.stack(
        [agent_data["precisions_belief"] for agent_data in all_agent_data.values()]
    )
    agent_pref_mean = jnp.stack(
        [agent_data["means_preference"] for agent_data in all_agent_data.values()]
    )
    agent_pref_precision = jnp.stack(
        [agent_data["precision_preference"] for agent_data in all_agent_data.values()]
    )

    # Calculate dissatisfaction
    pref_belief_gap = kl_divergence(
        agent_pref_mean,
        agent_pref_precision,
        beliefs_mean,
        beliefs_precision,
    )

    # Sum (return without sum if you want kl per prefence)
    pref_belief_gap = jnp.sum(pref_belief_gap, axis=1)

    # Return everything needed for subsequent steps
    return pref_belief_gap
