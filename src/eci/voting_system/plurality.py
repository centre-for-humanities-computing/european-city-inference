import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import _extract_env_data_vectorized
from eci.voting_system.decisions import (
    _compute_preferences,
    _sample_choice,
)


def _vote_plurality(env, key, *args, **kwargs) -> dict:
    """Perform plurality voting.

    Parameters
    ----------
    env:
        The environment object.
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    args:
        Variable length argument list.
    kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
        vote data.
    """
    # Extract all agent beliefs and preferences
    agent_data = _extract_env_data_vectorized(env)

    # Evaluate candidate scores for each agent
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    # Split the JAX key for two separate random samples
    key_round_1, key_round_2 = jax.random.split(key)

    # Create mask for round 1 (all candidates are eligible)
    mask_round_1 = jnp.ones_like(candidate_preferences, dtype=bool)
    masked_preferences = jnp.where(mask_round_1, candidate_preferences, -jnp.inf)

    # Sample round 1 vote
    vote_1, softmax_probs_1 = _sample_choice(key_round_1, masked_preferences)

    # Find the top two winners from round 1
    top_two_winners = _find_top_two_winners(vote_1, candidate_preferences.shape[1])

    # Create mask for round 2 (only top two are eligible)
    num_candidates = candidate_preferences.shape[1]
    all_candidates_indices = jnp.arange(num_candidates)
    mask_round_2_1d = jnp.isin(all_candidates_indices, top_two_winners)
    masked_preferences = jnp.where(mask_round_2_1d, candidate_preferences, -jnp.inf)

    # Sample final (round 2) vote
    vote_2, softmax_probs_2 = _sample_choice(key_round_2, masked_preferences)

    # Find the final winner (most votes in round 2)
    final_winner = _find_top_two_winners(vote_2, candidate_preferences.shape[1])[0]

    return {
        # --- Round 1 ---
        "vote_round_1": vote_1,
        "softmax_probs_round_1": softmax_probs_1,
        "first_round_winners": top_two_winners,
        # --- Round 2 ---
        "vote_final_round_2": vote_2,
        "softmax_probs_final_round_2": softmax_probs_2,
        "final_winner": final_winner,
        # metrics
        "pref_candidate_gap": pref_candidate_gap,
        "candidate_preferences": candidate_preferences,
        "pref_belief_gap": pref_belief_gap,
    }


def _find_top_two_winners(votes_array: ArrayLike, num_candidates: int) -> ArrayLike:
    """Find the two candidates with the most votes.

    Parameters
    ----------
    votes_array:
        voting result from the simulation.
    num_candidates:
        number of candidates.

    Returns
    -------
        vote data.
    """
    # Count votes for each unique candidate
    counts = jnp.bincount(votes_array.astype(jnp.int32), length=num_candidates)

    # Get indices that would sort the counts in ascending order
    _, top_two_winners = jax.lax.top_k(counts, k=2)

    return top_two_winners


def strategic_vote(env, key, *args, **kwargs) -> dict:
    """Perform plurality strategic voting.

    Parameters
    ----------
    env:
        The environment object.
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    args:
        Variable length argument list.
    kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
        vote data.
    """
    # Simulate a classic vote to predict expected winners
    expected_results = _vote_plurality(env, key, *args, **kwargs)
    expected_winners = expected_results["first_round_winners"]  # Top 2 candidates
    expected_final_winner = expected_results["final_winner"]  # Predicted winner

    # Extract agent preferences
    agent_data = _extract_env_data_vectorized(env)
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    # Weight preferences toward the expected winner
    strategic_factor = 2.0  # Boost preference for the predicted winner
    adjusted_preferences = candidate_preferences * (
        1 + (candidate_preferences == expected_final_winner) * (strategic_factor - 1)
    )

    # Update agent data with adjusted preferences
    kwargs["custom_preferences"] = adjusted_preferences

    # Re-run the vote with the new preferences
    new_key = jax.random.PRNGKey(jax.random.randint(key, (), 0, 1000000))
    strategic_results = _vote_plurality(env, new_key, *args, **kwargs)

    # Return strategic vote results + expected outcomes

    return {
        **strategic_results,
        "expected_winners": expected_winners,
        "expected_final_winner": expected_final_winner,
        "strategy_used": "weighted_by_expected_results",
    }
