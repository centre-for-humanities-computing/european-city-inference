import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.decision.scoring import ScoringFn, score_normalized
from eci.decision.types import ElectionData
from eci.utils import cross_entropy, kl_divergence


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
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
    pref_mean: jnp.ndarray,
    pref_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL(candidate || preference) summed across dims → (n_agents, n_cand)."""
    gap_per_dim = kl_divergence(
        cand_mean[None, :, :],
        cand_precision[None, :, :],
        pref_mean[:, None, :],
        pref_precision[:, None, :],
    )
    return jnp.sum(gap_per_dim, axis=-1)


def _get_pref_candidate_cross_entropy(
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
    pref_mean: jnp.ndarray,
    pref_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-entropy H(candidate || preference) summed across dims → (n_agents, n_cand).

    The cross-entropy counterpart of :func:`_get_pref_candidate_gap` (which uses
    KL). Equals KL plus the candidate's entropy, so it also penalises diffuse
    (low-precision) candidates.
    """
    cross_entropy_per_dimension = cross_entropy(
        cand_mean[None, :, :],
        cand_precision[None, :, :],
        pref_mean[:, None, :],
        pref_precision[:, None, :],
    )
    return jnp.sum(cross_entropy_per_dimension, axis=-1)


def _get_expected_future_belief_gap(
    beliefs_mean: jnp.ndarray,
    beliefs_precision: jnp.ndarray,
    pref_mean: jnp.ndarray,
    pref_precision: jnp.ndarray,
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL(Future_Belief || Preferences) using precision-weighted combination."""
    # Broadcast all distributions to (n_agents, n_candidates, n_dimensions).
    belief_means_by_candidate = beliefs_mean[:, None, :]
    belief_precisions_by_candidate = beliefs_precision[:, None, :]
    candidate_means_by_agent = cand_mean[None, :, :]
    candidate_precisions_by_agent = cand_precision[None, :, :]
    preference_means_by_candidate = pref_mean[:, None, :]
    preference_precisions_by_candidate = pref_precision[:, None, :]

    future_precision = belief_precisions_by_candidate + candidate_precisions_by_agent
    future_mean = (
        belief_means_by_candidate * belief_precisions_by_candidate
        + candidate_means_by_agent * candidate_precisions_by_agent
    ) / future_precision

    gap_per_dim = kl_divergence(
        future_mean,
        future_precision,
        preference_means_by_candidate,
        preference_precisions_by_candidate,
    )

    return jnp.sum(gap_per_dim, axis=-1)


def _compute_candidate_utilities(
    data: ElectionData,
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
    preference_candidate_gap   : ArrayLike, shape (n_agents, n_candidates)
    belief_preference_gap      : ArrayLike, shape (n_agents,)
    """
    belief_preference_gap = _get_belief_preference_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )
    preference_candidate_gap = _get_pref_candidate_gap(
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    preference_score_per_agent = scoring_fn(
        belief_preference_gap,
        preference_candidate_gap,
    )
    return (
        preference_score_per_agent,
        preference_candidate_gap,
        belief_preference_gap,
    )
