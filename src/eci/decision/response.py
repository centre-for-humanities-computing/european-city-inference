from typing import Optional, Protocol, Tuple, runtime_checkable

import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.typing import ArrayLike

from eci.decision.sampling import _sample_from_utilities
from eci.decision.scoring import score_normalized
from eci.decision.utilities import (
    _compute_candidate_utilities,
    _get_belief_preference_gap,
    _get_expected_future_belief_gap,
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
        data: dict,
        key: ArrayLike,
        mask: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Sample a vote per agent; see the class docstring for the contract."""
        ...


# TODO: Maybe give full data parameter instead of dataframe
def response_function(data, key, mask=None, *args, **kwargs):
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


# TODO: Maybe give full data parameter instead of dataframe
def response_function_random(data, key, mask=None, *args, **kwargs):
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


# TODO: Maybe give full data parameter instead of dataframe
def response_function_logpdf(data, key, mask=None, *args, **kwargs):
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
    pref_mean = data["preferences"]["mean"]
    pref_precision = data["preferences"]["precision"]
    cand_mean = data["candidates"]["mean"]
    pref_scale = 1.0 / jnp.sqrt(pref_precision)

    logpdf_per_dim = norm.logpdf(
        cand_mean[None, :, :],
        loc=pref_mean[:, None, :],
        scale=pref_scale[:, None, :],
    )
    utilities = jnp.sum(logpdf_per_dim, axis=-1)
    return _sample_from_utilities(utilities, key, mask)


# TODO: Maybe give full data parameter instead of dataframe
def response_function_pref(data, key, mask=None, *args, **kwargs):
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
    _, pref_candidate_gap, _ = _compute_candidate_utilities(data)
    utilities = -pref_candidate_gap
    return _sample_from_utilities(utilities, key, mask)


# TODO: Maybe give full data parameter instead of dataframe
def response_function_precision(data, key, mask=None, *args, **kwargs):
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
    tau = jnp.sum(data["beliefs"]["precision"], axis=-1, keepdims=True)
    gap = _get_pref_candidate_gap(
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
    )
    utilities = -tau * gap
    return _sample_from_utilities(utilities, key, mask)


# TODO: Maybe give full data parameter instead of dataframe
# TODO: Split the function
def response_function_bayesian(
    data: dict,
    key: ArrayLike,
    mask: Optional[ArrayLike] = None,
    *args,
    scoring_fn=score_normalized,
    **kwargs,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
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
    # compute utilities
    current_gap = _get_belief_preference_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    # compute expected future gap after Bayesian fusion of beliefs and candidates
    future_gap = _get_expected_future_belief_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
    )

    # compute utilities via the provided scoring function
    utilities = scoring_fn(current_gap, future_gap)

    # sample votes from utilities
    return _sample_from_utilities(utilities, key, mask)
