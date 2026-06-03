from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax.typing import ArrayLike

_EPS = 1e-8


@runtime_checkable
class ScoringFn(Protocol):
    """Protocol for a candidate-utility scoring strategy.

    Implementations are pure functions that turn two KL gaps into
    per-agent, per-candidate utility scores.

    Parameters
    ----------
    belief_gap : ArrayLike, shape (n_agents,)
        Per-agent KL divergence ``KL(belief || preference)``.
    pref_candidate_gap : ArrayLike, shape (n_agents, n_candidates)
        Per-agent, per-candidate KL divergence
        ``KL(candidate_policy || preference)``.

    Returns
    -------
    utilities : ArrayLike, shape (n_agents, n_candidates)
        Higher = better candidate for that agent. Will be fed to softmax
        downstream.
    """

    def __call__(
        self,
        belief_gap: ArrayLike,
        pref_candidate_gap: ArrayLike,
    ) -> ArrayLike:
        """Score candidates from the two KL gaps; see the class docstring."""
        ...


def score_normalized(belief_gap: ArrayLike, pref_candidate_gap: ArrayLike) -> ArrayLike:
    r"""Score candidates as a fraction of current dissatisfaction (normalized)."""
    belief_gap = jnp.asarray(belief_gap)
    safe = belief_gap[:, jnp.newaxis] + _EPS
    return (safe - pref_candidate_gap) / safe


def score_absolute(belief_gap: ArrayLike, pref_candidate_gap: ArrayLike) -> ArrayLike:
    r"""Absolute score — raw reduction of dissatisfaction."""
    belief_gap = jnp.asarray(belief_gap)
    return belief_gap[:, jnp.newaxis] - pref_candidate_gap


def score_inverted(belief_gap: ArrayLike, pref_candidate_gap: ArrayLike) -> ArrayLike:
    r"""Inverted score — distance penalty (sign-flipped absolute)."""
    belief_gap = jnp.asarray(belief_gap)
    return pref_candidate_gap - belief_gap[:, jnp.newaxis]


def score_product(belief_gap: ArrayLike, pref_candidate_gap: ArrayLike) -> ArrayLike:
    r"""Product score — dissatisfaction-weighted candidate distance."""
    belief_gap = jnp.asarray(belief_gap)
    return -belief_gap[:, jnp.newaxis] * pref_candidate_gap
