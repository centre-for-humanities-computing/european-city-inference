from typing import Optional

import jax
import jax.numpy as jnp

from eci.voting.types import VoteResult


# TODO: Allow positive and negative
def _vote_quadratic(
    data,
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
        Implements the :class:`~eci.voting_system.ResponseFunction` protocol.
    key:
        A JAX PRNG key for seeding random operations.
    budget:
        Token budget per voter (default 99.0).
    num_votes:
        Number of distinct candidates each voter spends on.

    Returns
    -------
    VoteResult
        See :class:`~eci.voting_system.types.VoteResult` for the full
        field contract. QV adds ``credits_spent``.
    """
    # Sample round 1 preferences.
    _, softmax_probs, candidate_utilities, key = response_function(data, key)

    # Allocate credits → votes per (agent, candidate).
    votes_matrix, credits_spent = _compute_sequential_qv_allocation(
        key, candidate_utilities, budget, num_votes=num_votes
    )
    votes_per_candidate = jnp.sum(votes_matrix, axis=0)
    winner = jnp.argmax(votes_per_candidate)

    return {
        # Uniform fields (preferred):
        "votes_matrix": votes_matrix,
        "votes_per_candidate": votes_per_candidate,
        "winner": winner,
        "softmax": softmax_probs,
        "candidate_utilities": candidate_utilities,
        # QV-specific:
        "credits_spent": credits_spent,
        # Legacy aliases — will be removed in v0.2:
        "votes": votes_per_candidate,
        "qv_votes_matrix": votes_matrix,
    }


# TODO: Implement different allocation strategies.
def _gumbel_top_k(key, logits, k):
    """Sample k distinct items per row with prob ∝ softmax(logits)."""
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape)))
    _, top_idx = jax.lax.top_k(logits + gumbel, k)
    return jnp.sum(jax.nn.one_hot(top_idx, logits.shape[1]), axis=1)


def _qv_credits_per_pick(utilities, budget, num_votes):
    """Per-(agent, candidate) credit weight before pick mask."""
    per_pick = budget / num_votes
    return jax.nn.softmax(utilities, axis=1) * per_pick


def _add_credit_jitter(credits, key, scale):
    """Add Gaussian jitter and clip to >= 0 so sqrt stays real."""
    noise = jax.random.normal(key, credits.shape) * scale
    return jnp.maximum(credits + noise, 0.0)


def _credits_to_votes(credits):
    """QV rule: votes = floor(sqrt(credits))."""
    return jnp.floor(jnp.sqrt(credits)).astype(jnp.int32)


def _compute_sequential_qv_allocation(
    key, utilities, budget, num_votes=5, noise_scale=0.05
):
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
    n_cand = utilities.shape[1]

    if num_votes is None:
        # Adaptive: distribute the full budget across all candidates by utility.
        weights = jax.nn.softmax(utilities, axis=1) * budget
        credits = _add_credit_jitter(weights, key, noise_scale * budget / n_cand)
        return _credits_to_votes(credits), credits

    num_votes = min(num_votes, n_cand)
    weights = _qv_credits_per_pick(utilities, budget, num_votes)
    gumbel_key, noise_key = jax.random.split(key)
    picks = _gumbel_top_k(gumbel_key, utilities, num_votes)
    credits = picks * _add_credit_jitter(
        weights, noise_key, noise_scale * budget / num_votes
    )
    return _credits_to_votes(credits), credits
