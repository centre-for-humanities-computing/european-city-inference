import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.voting_system.beliefs import _get_current_beliefs, _get_pref_belief_gap
from eci.voting_system.decisions import _compute_option_preferences, _sample_choice


def _vote_plurality(env, key, *args, **kwargs) -> dict:
    """Perform plurality vote for each agent."""
    # Extract all agent beliefs and preferences
    all_agent_data = _get_current_beliefs(env)

    # Get dissatifaction (gap between preference and belief) per agent
    pref_belief_gap = _get_pref_belief_gap(all_agent_data)

    # TODO: Do this part into a function.
    means_preference = jnp.stack(
        [agent_data["means_preference"] for agent_data in all_agent_data.values()]
    )
    precision_preference = jnp.stack(
        [agent_data["precision_preference"] for agent_data in all_agent_data.values()]
    )

    # Evaluate candidate scores for each agent
    candidate_preferences = _compute_option_preferences(
        env,
        means_preference,
        precision_preference,
        pref_belief_gap,
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
    top_two_winners = _find_top_two_winners(vote_1)

    # --- ROUND 2 ---

    # Create mask for round 2 (only top two are eligible)
    num_candidates = candidate_preferences.shape[1]
    all_candidates_indices = jnp.arange(num_candidates)
    mask_round_2_1d = jnp.isin(all_candidates_indices, top_two_winners)
    masked_preferences = jnp.where(mask_round_2_1d, candidate_preferences, -jnp.inf)

    # Sample final (round 2) vote
    vote_2, softmax_probs_2 = _sample_choice(key_round_2, masked_preferences)

    # Find the final winner (most votes in round 2)
    final_winner = _find_top_two_winners(vote_2)[0]

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
    }


def _find_top_two_winners(votes_array: ArrayLike) -> ArrayLike:
    """Find two candidates with the most votes."""
    # Count votes for each unique candidate
    unique_candidates, counts = jnp.unique(votes_array, return_counts=True)

    # Get indices that would sort the counts in ascending order
    sorted_indices = jnp.argsort(counts)

    # Take the last two indices (top 2 counts) and reverse them
    top_two_indices = sorted_indices[-2:][::-1]

    # Get the corresponding candidates
    top_two_winners = unique_candidates[top_two_indices]

    num_winners = top_two_winners.shape[0]
    if num_winners < 2:
        return jnp.pad(top_two_winners, (0, 2 - num_winners), mode="edge")

    return top_two_winners


def _update_public_poll(self, vote_counts: dict) -> None:
    """Update the public poll with the latest election results."""
    if not self.use_theory_of_mind or not vote_counts:
        return

    total_votes = sum(vote_counts.values())
    if total_votes == 0:
        return

    # Create a vector of vote proportions from the results dict
    candidate_ids = [c.id for c in self.candidates]
    self.public_poll = jnp.array(
        [vote_counts.get(cid, 0) / total_votes for cid in candidate_ids]
    )
    print(f"Public poll updated: {self.public_poll}")
