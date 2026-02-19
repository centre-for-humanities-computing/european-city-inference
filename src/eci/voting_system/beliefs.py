import jax.numpy as jnp

from eci.utils import (
    kl_divergence,
)


def _get_pref_belief_gap(data: dict) -> jnp.ndarray:
    """Compute the KL divergence between agent beliefs and preferences.

    Parameters
    ----------
    data : dict
        agent beliefs (mean,precision) and preference (mean,precision).

    Returns
    -------
    pref_belief_gap : jnp.ndarray
        The computed KL divergence summed per agent.
    """
    # Extract belief parameters
    beliefs_mean = data["beliefs"]["mean"]  # Shape: (n_agents, n_dim)
    beliefs_precision = data["beliefs"]["precision"]

    # Extract preference parameters
    pref_mean = data["preferences"]["mean"]
    pref_precision = data["preferences"]["precision"]

    # Compute KL(Preferences || Beliefs)
    gap_per_preference = kl_divergence(
        pref_mean,
        pref_precision,
        beliefs_mean,
        beliefs_precision,
    )

    # Sum over the preference axis.
    return jnp.sum(gap_per_preference, axis=-1)


def _get_pref_candidate_gap(data: dict) -> jnp.ndarray:
    """Compute the KL divergence between agent preferences and candidate policies.

    Parameters
    ----------
    data : dict
        agent beliefs (mean,precision) and candidate policies (mean,precision).

    Returns
    -------
    pref_cand_gap : jnp.ndarray
        The computed KL divergence summed per agent.
    """
    # Extract preference parameters
    pref_mean = data["preferences"]["mean"]
    pref_precision = data["preferences"]["precision"]

    # Extract candidate parameters
    cand_mean = data["candidates"]["mean"]
    cand_precision = data["candidates"]["precision"]

    # Compute KL(Preferences || Policy)
    gap_per_dim = kl_divergence(
        pref_mean[:, None, :],  # broadcasting (n_agents, n_candidates, n_dim)
        pref_precision[:, None, :],
        cand_mean[None, :, :],
        cand_precision[None, :, :],
    )
    # Sum across the dimensions.
    return jnp.sum(gap_per_dim, axis=-1)
