from typing import Optional

import jax
import jax.numpy as jnp

from eci.decision.types import ElectionData
from eci.voting.types import VoteResult


# TODO: Allow positive and negative
def _vote_quadratic(
    data: ElectionData,
    response_function,
    key,
    *args,
    budget: float = 99.0,
    num_votes: Optional[int] = 5,
    **kwargs,
) -> VoteResult:
    """Perform quadratic voting.

    Parameters
    ----------
    data:
        Agent data dict (beliefs, preferences, candidates).
    response_function:
        Implements the :class:`~eci.decision.ResponseFunction` protocol.
    key:
        A JAX PRNG key for seeding random operations.
    budget:
        Token budget per voter (default 99.0).
    num_votes:
        Number of distinct candidates each voter spends on.

    Returns
    -------
    VoteResult
        See :class:`~eci.voting.types.VoteResult` for the full
        field contract. QV adds ``credits_spent``.
    """
    _, vote_probabilities, candidate_utilities, allocation_key = response_function(
        data,
        key,
    )

    votes_matrix, credits_spent = _compute_qv_allocation(
        allocation_key,
        candidate_utilities,
        budget,
        num_votes=num_votes,
    )
    votes_per_candidate = jnp.sum(votes_matrix, axis=0)
    winner = jnp.argmax(votes_per_candidate)

    return {
        # Uniform fields (preferred):
        "votes_matrix": votes_matrix,
        "votes_per_candidate": votes_per_candidate,
        "winner": winner,
        "softmax": vote_probabilities,
        "candidate_utilities": candidate_utilities,
        # QV-specific:
        "credits_spent": credits_spent,
        # Legacy aliases — will be removed in v0.2:
        "votes": votes_per_candidate,
        "qv_votes_matrix": votes_matrix,
    }


# TODO: Implement different allocation strategies.
def _gumbel_top_k(key, logits, selection_count):
    """Sample k distinct items per row with prob ∝ softmax(logits)."""
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape)))
    _, selected_indices = jax.lax.top_k(
        logits + gumbel_noise,
        selection_count,
    )
    return jnp.sum(
        jax.nn.one_hot(selected_indices, logits.shape[1]),
        axis=1,
    )


def _add_credit_jitter(credits, key, scale):
    """Add Gaussian jitter and clip to >= 0 so sqrt stays real."""
    credit_jitter = jax.random.normal(key, credits.shape) * scale
    return jnp.maximum(credits + credit_jitter, 0.0)


def _normalize_credit_budget(credits, budget, fallback_weights):
    """Normalize each agent's non-negative credits to exactly ``budget``."""
    credit_totals = jnp.sum(credits, axis=1, keepdims=True)
    fallback_weight_totals = jnp.sum(
        fallback_weights,
        axis=1,
        keepdims=True,
    )
    normalized_weights = jnp.where(
        credit_totals > 0,
        credits / jnp.maximum(credit_totals, 1e-12),
        fallback_weights / jnp.maximum(fallback_weight_totals, 1e-12),
    )
    return normalized_weights * budget


def _credits_to_votes(credits):
    """QV rule: votes = floor(sqrt(credits))."""
    return jnp.floor(jnp.sqrt(credits)).astype(jnp.int32)


def _compute_qv_allocation(key, utilities, budget, num_votes=5, noise_scale=0.05):
    """Allocate QV credits to candidates per agent, then convert to votes.

    Parameters
    ----------
    key:
        JAX PRNG key for sampling noise and/or picks.
    utilities:
        Per-agent utilities for each candidate, shape (n_agents, n_cand).
    budget:
        Total credits available per agent.
    num_votes:
        Allocate credits to the top `num_votes` candidates per agent.
    noise_scale:
        Scale of Gaussian noise added to credit allocation (as a fraction of budget).

    Returns
    -------
    ``(votes_matrix, credits_spent)``, both shape (n_agents, n_cand).
    """
    candidate_count = utilities.shape[1]

    if num_votes is None:
        # Adaptive: distribute the full budget across all candidates by utility.
        utility_weights = jax.nn.softmax(utilities, axis=1)
        perturbed_credits = _add_credit_jitter(
            utility_weights * budget,
            key,
            noise_scale * budget / candidate_count,
        )
        normalized_credits = _normalize_credit_budget(
            perturbed_credits,
            budget,
            utility_weights,
        )
        return _credits_to_votes(normalized_credits), normalized_credits

    num_votes = min(num_votes, candidate_count)
    gumbel_key, noise_key = jax.random.split(key)
    selected_candidates = _gumbel_top_k(gumbel_key, utilities, num_votes)
    selected_utility_weights = selected_candidates * jax.nn.softmax(
        utilities,
        axis=1,
    )
    perturbed_credits = selected_candidates * _add_credit_jitter(
        selected_utility_weights * budget,
        noise_key,
        noise_scale * budget / num_votes,
    )
    normalized_credits = _normalize_credit_budget(
        perturbed_credits,
        budget,
        selected_candidates,
    )
    return _credits_to_votes(normalized_credits), normalized_credits


# Backward-compatible alias for code written before the allocation stopped being
# sequential.
_compute_sequential_qv_allocation = _compute_qv_allocation
