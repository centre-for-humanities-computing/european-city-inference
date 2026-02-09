import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.voting_system.decisions import _sample_choice


def _vote_random(env, key, *args, **kwargs) -> dict:
    """Perform random voting.

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
    num_agents = len(env.voters)
    num_candidates = len(env.candidates)

    # Split the JAX key for random preferences and voting
    key_prefs, key_votes = jax.random.split(key)

    # random candidate preference shape (agent, candidate)
    random_preferences = jax.random.uniform(
        key_prefs, shape=(num_agents, num_candidates)
    )

    # --- ROUND 1 ---

    # Split the JAX key for two separate random samples
    key_round_1, key_round_2 = jax.random.split(key_votes)

    # Create mask for round 1 (all candidates are eligible)
    mask_round_1 = jnp.ones_like(random_preferences, dtype=bool)
    masked_preferences = jnp.where(mask_round_1, random_preferences, -jnp.inf)

    # Sample round 1 vote
    vote_1, softmax_probs_1 = _sample_choice(key_round_1, masked_preferences)

    # Find the top two winners from round 1
    top_two_winners = _find_top_two_winners(vote_1, random_preferences.shape[1])

    # --- ROUND 2 ---

    # Create mask for round 2 (only top two are eligible)
    num_candidates = random_preferences.shape[1]
    all_candidates_indices = jnp.arange(num_candidates)
    mask_round_2_1d = jnp.isin(all_candidates_indices, top_two_winners)
    masked_preferences = jnp.where(mask_round_2_1d, random_preferences, -jnp.inf)

    # Sample final (round 2) vote
    vote_2, softmax_probs_2 = _sample_choice(key_round_2, masked_preferences)

    # Find the final winner (most votes in round 2)
    final_winner = _find_top_two_winners(vote_2, random_preferences.shape[1])[0]

    # --- RESULTS ---
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
        "pref_candidate_gap": random_preferences,
        "candidate_preferences": random_preferences,
        "pref_belief_gap": random_preferences,
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
