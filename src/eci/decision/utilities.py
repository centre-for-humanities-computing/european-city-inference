import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.decision.scoring import ScoringFn, score_normalized
from eci.utils import kl_divergence


def _get_belief_preference_gap(
    beliefs_mean: ArrayLike,
    beliefs_precision: ArrayLike,
    pref_mean: ArrayLike,
    pref_precision: ArrayLike,
) -> jnp.ndarray:
    """Compute KL(beliefs || preferences) summed across preference dims."""
    gap_per_dim = kl_divergence(
        beliefs_mean,
        beliefs_precision,
        pref_mean,
        pref_precision,
    )
    return jnp.sum(gap_per_dim, axis=-1)


def _get_pref_candidate_gap(
    cand_mean: ArrayLike,
    cand_precision: ArrayLike,
    pref_mean: ArrayLike,
    pref_precision: ArrayLike,
) -> jnp.ndarray:
    """Compute KL(candidate || preference) summed across dims → (n_agents, n_cand)."""
    gap_per_dim = kl_divergence(
        cand_mean[None, :, :],
        cand_precision[None, :, :],
        pref_mean[:, None, :],
        pref_precision[:, None, :],
    )
    return jnp.sum(gap_per_dim, axis=-1)


def _get_expected_future_belief_gap(
    beliefs_mean: ArrayLike,
    beliefs_precision: ArrayLike,
    pref_mean: ArrayLike,
    pref_precision: ArrayLike,
    cand_mean: ArrayLike,
    cand_precision: ArrayLike,
) -> jnp.ndarray:
    """Compute KL(Future_Belief || Preferences) using precision-weighted combination."""
    # broadcasting: (n_agents, n_candidates, n_dims)
    b_mean = beliefs_mean[:, None, :]
    b_prec = beliefs_precision[:, None, :]
    c_mean = cand_mean[None, :, :]
    c_prec = cand_precision[None, :, :]
    p_mean = pref_mean[:, None, :]
    p_prec = pref_precision[:, None, :]

    # Compute the new belief parameters (Bayesian Update)
    future_prec = b_prec + c_prec
    future_mean = (b_mean * b_prec + c_mean * c_prec) / future_prec

    # Compute KL(Future_Belief || Preference) per dimension
    gap_per_dim = kl_divergence(
        future_mean,
        future_prec,
        p_mean,
        p_prec,
    )

    # Sum across dimensions
    return jnp.sum(gap_per_dim, axis=-1)


def _compute_candidate_utilities(
    data: dict,
    scoring_fn: ScoringFn = score_normalized,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Evaluate per-agent utility score for each candidate.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    scoring_fn
        Strategy from :mod:`eci.decision.scoring` that turns the two KL
        gaps into per-agent, per-candidate utilities.

    Returns
    -------
    preference_score_per_agent : ArrayLike, shape (n_agents, n_candidates)
    pref_candidate_gap         : ArrayLike, shape (n_agents, n_candidates)
    belief_preference_gap      : ArrayLike, shape (n_agents,)
    """
    belief_preference_gap = _get_belief_preference_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )
    pref_candidate_gap = _get_pref_candidate_gap(
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    # For the expected future belief gap, we treat the candidate as if it were
    # pref_future_belief_gap_gap = _get_expected_future_belief_gap(
    #    data["beliefs"]["mean"],
    #    data["beliefs"]["precision"],
    #    data["preferences"]["mean"],
    #    data["preferences"]["precision"],
    #    data["candidates"]["mean"],
    #    data["candidates"]["precision"],
    # )
    preference_score_per_agent = scoring_fn(belief_preference_gap, pref_candidate_gap)
    return preference_score_per_agent, pref_candidate_gap, belief_preference_gap
