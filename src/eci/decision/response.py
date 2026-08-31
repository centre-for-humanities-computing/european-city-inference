from typing import Optional, Protocol, runtime_checkable

import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.typing import ArrayLike

from eci.decision.sampling import _sample_from_utilities
from eci.decision.scoring import ScoringFn, score_normalized
from eci.decision.types import ElectionData, ResponseResult
from eci.decision.utilities import (
    _compute_candidate_utilities,
    _get_belief_preference_gap,
    _get_expected_free_energy,
    _get_expected_future_belief_gap,
    _get_pref_candidate_cross_entropy,
    _get_pref_candidate_gap,
)


@runtime_checkable
class ResponseFunction(Protocol):
    """Protocol for vote-sampling response functions.

    Anyone can implement a custom response function by writing a
    callable matching this signature.

    Parameters
    ----------
    data : dict
        Agent data dict with keys ``"beliefs"``, ``"preferences"``,
        ``"candidates"``.
    key : jax.Array
        A JAX PRNG key.
    mask : jax.Array, optional
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it.

    Returns
    -------
    vote : jax.Array, shape (n_agents,)
        Sampled candidate index per agent.
    softmax_probs : jax.Array, shape (n_agents, n_candidates)
        Vote distribution per agent.
    candidate_utilities : jax.Array, shape (n_agents, n_candidates)
        Raw scores (logits) before softmax.
    next_key : jax.Array
        A PRNG key.
    """

    def __call__(
        self,
        data: ElectionData,
        key: ArrayLike,
        mask: Optional[ArrayLike] = None,
    ) -> ResponseResult:
        """Sample a vote per agent; see the class docstring for the contract."""
        ...


def response_function(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    """Sample one vote per agent using normalised KL-based utilities.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it.

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    utilities, _, _ = _compute_candidate_utilities(data)
    return _sample_from_utilities(utilities, key, mask)


def response_function_random(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    """Sample one vote uniformly at random from the candidate list.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key for seeding categorical sampling.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False``.

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    n_agents = data["preferences"]["mean"].shape[0]
    n_candidates = data["candidates"]["mean"].shape[0]
    utilities = jnp.zeros((n_agents, n_candidates))
    return _sample_from_utilities(utilities, key, mask)


def response_function_logpdf(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    """Sample one vote per agent using Gaussian log-pdf under preferences.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key for seeding categorical sampling.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False``.

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    preference_means = data["preferences"]["mean"]
    preference_precisions = data["preferences"]["precision"]
    candidate_means = data["candidates"]["mean"]
    preference_scales = 1.0 / jnp.sqrt(preference_precisions)

    log_probability_per_dimension = norm.logpdf(
        candidate_means[None, :, :],
        loc=preference_means[:, None, :],
        scale=preference_scales[:, None, :],
    )
    utilities = jnp.sum(log_probability_per_dimension, axis=-1)
    return _sample_from_utilities(utilities, key, mask)


def response_function_pref(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    """Sample one vote per agent using negative KL(pref || candidate).

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key for seeding categorical sampling.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it (utility set to ``-inf`` before
        softmax).

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    preference_candidate_gap = _get_pref_candidate_gap(
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    utilities = -preference_candidate_gap
    return _sample_from_utilities(utilities, key, mask)


def response_function_cross_entropy(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    """Sample one vote per agent using negative cross-entropy H(candidate, pref).

    The cross-entropy counterpart of :func:`response_function_pref`: it scores
    each candidate by ``-H(candidate || preference)`` instead of ``-KL``. Since
    ``H = KL + entropy(candidate)``, a diffuse (low-precision) candidate is
    penalised even when its mean matches the preference.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key for seeding categorical sampling.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it (utility set to ``-inf`` before
        softmax).

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    cross_entropy_gap = _get_pref_candidate_cross_entropy(
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    utilities = -cross_entropy_gap
    return _sample_from_utilities(utilities, key, mask)


def response_function_precision(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    **kwargs,
) -> ResponseResult:
    r"""Sample one vote per agent with a **precision-weighted softmax**.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it (utility set to ``-inf`` before
        softmax).

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    belief_precision_weight = jnp.sum(
        data["beliefs"]["precision"], axis=-1, keepdims=True
    )
    preference_candidate_gap = _get_pref_candidate_gap(
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )
    utilities = -belief_precision_weight * preference_candidate_gap
    return _sample_from_utilities(utilities, key, mask)


def response_function_bayesian(
    data: ElectionData,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    scoring_fn: ScoringFn = score_normalized,
    **kwargs,
) -> ResponseResult:
    """Sample one vote per agent using Bayesian fusion of beliefs and candidates.

    Unlike standard response functions that compare preferences directly to
    the candidate's platform, this function models the election as an
    observation. The agent infers an expected future world by fusing their
    current belief with the candidate's platform (weighted by precision),
    and evaluates this future against their preferences.

    Parameters
    ----------
    data
        Agent data dict with keys ``"beliefs"``, ``"preferences"``, ``"candidates"``.
    key
        A JAX PRNG key.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it.
    scoring_fn
        A function to convert the current and future KL gaps into utilities.
    **kwargs
        Additional keyword arguments passed to the scoring function.

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract.
    """
    current_gap = _get_belief_preference_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    future_gap = _get_expected_future_belief_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
    )

    utilities = scoring_fn(current_gap, future_gap)

    return _sample_from_utilities(utilities, key, mask)


# TODO: Maybe give full data parameter instead of dataframe
def response_function_efe(
    data: dict,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    gamma: float = 1.0,
    **kwargs,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    r"""Sample one vote per agent by minimising **Expected Free Energy**.

    This is the active-inference / Bayesian-decision-theory response function.
    For each candidate the agent imagines the post-election world by fusing its
    current belief with the candidate's platform, then scores that world by its
    Expected Free Energy

    .. math::

        G(c) = H\!\left(q^c_\text{future},\, P\right)
             = D_{KL}\!\left(q^c_\text{future} \,\|\, P\right)
             + H\!\left(q^c_\text{future}\right),

    a *risk* term (KL to the preference) plus an *ambiguity* term (entropy of
    the future world), and votes with a precision-weighted softmax of -EFE,
    :math:`P(\text{vote}=c) \propto \exp(-\gamma\, G(c))`.

    Unlike the direct response functions, the belief enters through the fusion
    (so it changes *which* candidate is favoured, not merely the decisiveness),
    while the action precision ``gamma`` controls how sharply the agent commits
    to the lowest-EFE candidate.

    Parameters
    ----------
    data
        Agent data dict with ``beliefs``, ``preferences``, ``candidates``,
        each holding ``mean`` and ``precision`` arrays.
    key
        A JAX PRNG key.
    mask
        Boolean array of shape ``(n_candidates,)``. ``True`` keeps the
        candidate, ``False`` excludes it (utility set to ``-inf`` before
        softmax).
    gamma
        Action precision (inverse softmax temperature). Larger ``gamma`` makes
        the vote sharper; ``gamma -> 0`` makes it uniform.

    Returns
    -------
    vote, softmax_probs, candidate_utilities, next_key
        See :class:`ResponseFunction` for the full shape contract. Here
        ``candidate_utilities`` are the negative-EFE scores ``-gamma * G(c)``.
    """
    free_energy = _get_expected_free_energy(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
    )

    utilities = -gamma * free_energy
    return _sample_from_utilities(utilities, key, mask)
