import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.decision.scoring import ScoringFn, score_normalized
from eci.utils import cross_entropy, kl_divergence


def _fuse_belief_candidate(
    beliefs_mean: jnp.ndarray,
    beliefs_precision: jnp.ndarray,
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precision-weighted Bayesian fusion of a belief with each candidate platform.

    Treats the current belief as a Gaussian prior and the candidate's platform as
    a Gaussian observation, returning the conjugate posterior parameters: the
    precisions add and the mean is their precision-weighted average.

    Returns ``(future_mean, future_precision)``, each broadcast to shape
    ``(n_agents, n_candidates, n_dims)``.
    """
    b_mean = beliefs_mean[:, None, :]
    b_prec = beliefs_precision[:, None, :]
    c_mean = cand_mean[None, :, :]
    c_prec = cand_precision[None, :, :]

    future_prec = b_prec + c_prec
    future_mean = (b_mean * b_prec + c_mean * c_prec) / future_prec
    return future_mean, future_prec


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
    ce_per_dim = cross_entropy(
        cand_mean[None, :, :],
        cand_precision[None, :, :],
        pref_mean[:, None, :],
        pref_precision[:, None, :],
    )
    return jnp.sum(ce_per_dim, axis=-1)


def _get_expected_future_belief_gap(
    beliefs_mean: jnp.ndarray,
    beliefs_precision: jnp.ndarray,
    pref_mean: jnp.ndarray,
    pref_precision: jnp.ndarray,
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL(Future_Belief || Preferences) using precision-weighted combination."""
    # broadcasting: (n_agents, n_candidates, n_dims)
    future_mean, future_prec = _fuse_belief_candidate(
        beliefs_mean, beliefs_precision, cand_mean, cand_precision
    )
    p_mean = pref_mean[:, None, :]
    p_prec = pref_precision[:, None, :]

    # Compute KL(Future_Belief || Preference) per dimension
    gap_per_dim = kl_divergence(
        future_mean,
        future_prec,
        p_mean,
        p_prec,
    )

    # Sum across dimensions
    return jnp.sum(gap_per_dim, axis=-1)


def _get_expected_free_energy(
    beliefs_mean: jnp.ndarray,
    beliefs_precision: jnp.ndarray,
    pref_mean: jnp.ndarray,
    pref_precision: jnp.ndarray,
    cand_mean: jnp.ndarray,
    cand_precision: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute Expected Free Energy of the post-election belief, per candidate.

    For every candidate the belief is fused with the candidate's platform
    (:func:`_fuse_belief_candidate`) to obtain the *expected future belief*
    :math:`q^c_\text{future}`. The Expected Free Energy of voting for that
    candidate is the cross-entropy of this future world relative to the
    preference, which decomposes into a *risk* and an *ambiguity* term:

    .. math::

        G(c) = H\!\left(q^c_\text{future},\, P\right)
             = D_{KL}\!\left(q^c_\text{future} \,\|\, P\right)
             + H\!\left(q^c_\text{future}\right),

    i.e. a *risk* term (the KL of the future world to the preference) plus an
    *ambiguity* term (the entropy of the future world).

    Lower ``G`` is better. Minimising ``G`` is equivalent to maximising the
    expected log-preference of the world the vote produces
    (Bayesian decision theory), and to minimising Expected Free Energy
    (active inference).

    Parameters
    ----------
    beliefs_mean, beliefs_precision, pref_mean, pref_precision,
    cand_mean, cand_precision
        Per-agent ``(n_agents, n_dims)`` belief/preference arrays and
        ``(n_candidates, n_dims)`` candidate arrays.

    Returns
    -------
    ArrayLike, shape ``(n_agents, n_candidates)``
        Expected Free Energy ``G(c)`` summed across preference dimensions.

    """
    future_mean, future_prec = _fuse_belief_candidate(
        beliefs_mean, beliefs_precision, cand_mean, cand_precision
    )
    p_mean = pref_mean[:, None, :]
    p_prec = pref_precision[:, None, :]

    # risk + ambiguity = cross-entropy of the expected future world vs. preference
    return jnp.sum(cross_entropy(future_mean, future_prec, p_mean, p_prec), axis=-1)


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
