import jax.numpy as jnp

from eci.utils import (
    kl_divergence,
)


def _get_pref_belief_gap(data: dict) -> jnp.ndarray:
    """
    Compute the gap between current beliefs and preferences for all agents.

    Parameters
    ----------
    data : dict
        The dictionary returned by _extract_env_data_vectorized containing
        'beliefs' and 'preferences' matrices.

    Returns
    -------
    pref_belief_gap : jnp.ndarray
        The computed KL divergence summed per agent. Shape: (n_voters,).
    """
    # Extract belief parameters
    # Shape: (n_agents, n_dim)
    beliefs_mean = data["beliefs"]["mean"]
    beliefs_precision = data["beliefs"]["precision"]

    pref_mean = data["preferences"]["mean"]
    pref_precision = data["preferences"]["precision"]

    # Compute KL(Preferences || Beliefs)
    gap_per_preference = kl_divergence(
        pref_mean,
        pref_precision,
        beliefs_mean,
        beliefs_precision,
    )

    # Sum over the preference axis (axis=1) to get a score per agent
    return jnp.sum(gap_per_preference, axis=1)


def _get_pref_candidate_gap(data: dict) -> jnp.ndarray:
    """
    Compute the KL divergence between agent preferences and candidate policies.

    Parameters
    ----------
    data : dict
        Contains 'preferences' and 'candidates' keys, each with 'mean'
        and 'precision' arrays of shape (n_agents, n_dim).

    Returns
    -------
    pref_cand_gap : jnp.ndarray
        The summed KL divergence per agent. Shape: (n_agents,).
    """
    # Extract preference parameters
    pref_mean = data["preferences"]["mean"]
    pref_precision = data["preferences"]["precision"]

    # Extract candidate parameters
    cand_mean = data["candidates"]["mean"]
    cand_precision = data["candidates"]["precision"]

    # Compute KL(Preferences || Candidate)
    gap_per_dim = kl_divergence(
        pref_mean[:, None, :],
        pref_precision[:, None, :],
        cand_mean[None, :, :],
        cand_precision[None, :, :],
    )
    # Sum across the dimensions (axis=1) to get a total gap per agent
    return jnp.sum(gap_per_dim, axis=-1)
