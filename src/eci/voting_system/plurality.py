import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import _extract_env_data_vectorized
from eci.voting_system.decisions import (
    _compute_preferences,
    _sample_choice,
)


def _vote_plurality(env, key, *args, **kwargs) -> dict:
    """Perform plurality vote for each agent."""
    # Extract all agent beliefs and preferences
    agent_data = _extract_env_data_vectorized(env)

    # Evaluate candidate scores for each agent
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    # --- ROUND 1 ---

    # Split the JAX key for two separate random samples
    key_round_1, key_round_2 = jax.random.split(key)

    # Create mask for round 1 (all candidates are eligible)
    mask_round_1 = jnp.ones_like(candidate_preferences, dtype=bool)
    masked_preferences = jnp.where(mask_round_1, candidate_preferences, -jnp.inf)

    # Sample round 1 vote
    vote_1, softmax_probs_1 = _sample_choice(key_round_1, masked_preferences)

    # Find the top two winners from round 1
    top_two_winners = _find_top_two_winners(vote_1, candidate_preferences.shape[1])

    # --- ROUND 2 ---

    # Create mask for round 2 (only top two are eligible)
    num_candidates = candidate_preferences.shape[1]
    all_candidates_indices = jnp.arange(num_candidates)
    mask_round_2_1d = jnp.isin(all_candidates_indices, top_two_winners)
    masked_preferences = jnp.where(mask_round_2_1d, candidate_preferences, -jnp.inf)

    # Sample final (round 2) vote
    vote_2, softmax_probs_2 = _sample_choice(key_round_2, masked_preferences)

    # Find the final winner (most votes in round 2)
    final_winner = _find_top_two_winners(vote_2, candidate_preferences.shape[1])[0]

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
        "pref_candidate_gap": pref_candidate_gap,
        "candidate_preferences": candidate_preferences,
        "pref_belief_gap": pref_belief_gap,
    }


def _find_top_two_winners(votes_array: ArrayLike, num_candidates: int) -> ArrayLike:
    """Find two candidates with the most votes."""
    # Count votes for each unique candidate
    counts = jnp.bincount(votes_array.astype(jnp.int32), length=num_candidates)

    # Get indices that would sort the counts in ascending order
    _, top_two_winners = jax.lax.top_k(counts, k=2)

    return top_two_winners


############# Public Poll Update IN CONSTRUCTION  #############
def _compute_poll_results(votes: jnp.ndarray, num_candidates: int) -> jnp.ndarray:
    """Calculate vote proportions."""
    # 1. Count votes for every candidate ID (0 to num_candidates-1)
    # This is much faster than a Python loop over a dictionary
    vote_counts = jnp.bincount(votes, length=num_candidates)

    # 2. Calculate proportions
    total_votes = votes.shape[0]

    # We use jnp.where to handle the edge case of 0 votes safely
    return jnp.where(total_votes > 0, vote_counts / total_votes, 0.0)


# Example of how to integrate it into your class method
def update_public_poll(self, votes: jnp.ndarray):
    """Update the public poll with the latest election results."""
    if not self.use_theory_of_mind:
        return

    # num_candidates must be known (static)
    num_candidates = len(self.candidates)

    # Compute the new poll using the JAX-native function
    self.public_poll = _compute_poll_results(votes, num_candidates)

    # Logging is fine here because this method is the entry point
    print(f"Public poll updated: {self.public_poll}")
