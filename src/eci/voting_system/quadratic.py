import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import _extract_env_data_vectorized
from eci.voting_system.decisions import _compute_preferences, _sample_choice


def _vote_quadratic(env, key, budget: float = 99.0, *args, **kwargs) -> dict:
    """Perform quadratic voting.

    Parameters
    ----------
    env:
        The environment object.
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    args:
        Variable length argument list.
    budget:
        Token for quadratic voting.
    kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
        vote data.
    """
    # Extract all agent beliefs and preferences
    agent_data = _extract_env_data_vectorized(env)

    # Evaluate candidate scores for each agent
    if "custom_preferences" in kwargs:
        candidate_preferences = kwargs["custom_preferences"]
        _, pref_candidate_gap, pref_belief_gap = _compute_preferences(agent_data)
    else:
        # Extract all agent beliefs and preferences
        candidate_preferences, pref_candidate_gap, pref_belief_gap = (
            _compute_preferences(agent_data)
        )
    candidate_ids = jnp.array([c.id for c in env.candidates])

    # Allocation QV
    votes_matrix, credits_spent = _compute_sequential_qv_allocation(
        key, candidate_preferences, budget
    )

    # Compute final winner
    total_votes = jnp.sum(votes_matrix, axis=0)
    final_winner_idx = jnp.argmax(total_votes)
    final_winner = candidate_ids[final_winner_idx]

    top_investment_indices = jnp.argmax(votes_matrix, axis=1)
    vote_choice_legacy = candidate_ids[top_investment_indices]

    row_sums = jnp.sum(votes_matrix, axis=1, keepdims=True)
    safe_row_sums = jnp.where(row_sums == 0, 1.0, row_sums)
    pseudo_probs = votes_matrix / safe_row_sums

    sorted_indices = jnp.argsort(total_votes)[::-1]
    top_two_indices = sorted_indices[:2]
    top_two_winners = candidate_ids[top_two_indices]

    if top_two_winners.shape[0] < 2:
        top_two_winners = jnp.pad(
            top_two_winners, (0, 2 - top_two_winners.shape[0]), mode="edge"
        )

    return {
        # round 1
        "vote_round_1": vote_choice_legacy,
        "softmax_probs_round_1": pseudo_probs,
        "first_round_winners": top_two_winners,
        # round 2 (duplicata)
        "vote_final_round_2": vote_choice_legacy,
        "softmax_probs_final_round_2": pseudo_probs,
        "final_winner": final_winner,
        # qv data specific
        "total_votes_per_candidate": total_votes,
        "credits_spent": credits_spent,
        "qv_votes_matrix": votes_matrix,
        "pref_candidate_gap": pref_candidate_gap,
        "candidate_preferences": candidate_preferences,
        "pref_belief_gap": pref_belief_gap,
    }


def _compute_sequential_qv_allocation(
    key: jax.random.PRNGKey, candidate_preferences: ArrayLike, budget: float
) -> tuple[ArrayLike, ArrayLike]:
    num_agents, num_candidates = candidate_preferences.shape

    # Normalize preferences to get weights for each vote
    pref_sums = jnp.sum(candidate_preferences, axis=1, keepdims=True)
    normalized_prefs = candidate_preferences / (
        pref_sums + 1e-9
    )  # Avoid division by zero

    # Use normalized preferences to determine the number of votes
    num_votes = 5  # Fixed number of votes
    vote_weights = normalized_prefs * (
        budget / num_votes
    )  # Distribute budget across votes

    credits_spent = jnp.zeros((num_agents, num_candidates))
    current_prefs = candidate_preferences
    keys = jax.random.split(key, num_votes)

    for i in range(num_votes):
        # Sample choice based on current preferences
        choice_indices, _ = _sample_choice(keys[i], current_prefs)
        # Create one-hot encoding for chosen candidates
        choice_one_hot = jax.nn.one_hot(choice_indices, num_candidates)
        # Distribute credits spent based on vote weights
        credits_spent += choice_one_hot * vote_weights
        # Remove chosen candidate from current preferences to avoid re-selection
        current_prefs -= choice_one_hot * 1e9

    votes_matrix = jnp.floor(jnp.sqrt(credits_spent)).astype(jnp.int32)
    return votes_matrix, credits_spent


def strategic_quadratic_vote(
    env, key, alpha: float = 1.0, budget: float = 99.0, *args, **kwargs
) -> dict:
    """Perform quadratic strategic voting."""
    # Use the vote function to simulate a poll.
    expected_results = _vote_quadratic(env, key, *args, **kwargs)
    expected_probs = jnp.mean(expected_results["softmax_probs_round_1"], axis=0)

    # Compute candidate preferences and gaps for all agents
    agent_data = _extract_env_data_vectorized(env)
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    # Boost viable candidates, penalize hopeless ones.
    eps = 1e-8
    viability_bonus = jnp.log(expected_probs + eps)
    adjusted_preferences = candidate_preferences - alpha * viability_bonus

    # Re-run the vote with the new preferences
    new_key = jax.random.split(key)[0]
    strategic_results = _vote_quadratic(
        env, new_key, budget, *args, **kwargs, custom_preferences=adjusted_preferences
    )

    return {
        **strategic_results,
        "expected_results": expected_probs,  # Return the array
        "strategy_used": "weighted_by_expected_results",
    }
