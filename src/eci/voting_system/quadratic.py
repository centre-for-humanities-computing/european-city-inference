import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Ensure these imports match your project structure
from eci.voting_system.beliefs import _get_current_beliefs, _get_pref_belief_gap
from eci.voting_system.decisions import _compute_option_preferences, _sample_choice


def _vote_quadratic(env, key, budget: float = 99.0, *args, **kwargs) -> dict:
    """Orchestrates the Quadratic Voting process within the simulation."""
    # 1. Retrieve beliefs and compute preferences
    all_agent_data = _get_current_beliefs(env)
    pref_belief_gap = _get_pref_belief_gap(all_agent_data)
    # TODO: Do this part into a function.
    means_preference = jnp.stack(
        [agent_data["means_preference"] for agent_data in all_agent_data.values()]
    )
    precision_preference = jnp.stack(
        [agent_data["precision_preference"] for agent_data in all_agent_data.values()]
    )

    candidate_preferences = _compute_option_preferences(
        env, means_preference, precision_preference, pref_belief_gap
    )

    # 2. Perform sequential Quadratic Voting allocation
    votes_matrix, credits_spent = _compute_sequential_qv_allocation(
        key, candidate_preferences, budget
    )

    # 3. Determine Final Winner (Sum of SQRT votes)
    total_votes = jnp.sum(votes_matrix, axis=0)
    final_winner_idx = jnp.argmax(total_votes)

    # Map indices back to Candidate IDs
    candidate_ids = jnp.array([c.id for c in env.candidates])
    final_winner = candidate_ids[final_winner_idx]
    # 4. Determine "Legacy" Vote (Agent's highest investment)
    # Used to maintain compatibility with system expecting a single choice per agent
    top_investment_indices = jnp.argmax(votes_matrix, axis=1)
    vote_choice_legacy = candidate_ids[top_investment_indices]

    # 5. Determine Top 2 Winners (for Round 2 simulation)
    sorted_indices = jnp.argsort(total_votes)[::-1]
    top_two_indices = sorted_indices[:2]
    top_two_winners = candidate_ids[top_two_indices]

    # Pad if fewer than 2 candidates exist
    if top_two_winners.shape[0] < 2:
        top_two_winners = jnp.pad(
            top_two_winners, (0, 2 - top_two_winners.shape[0]), mode="edge"
        )

    return {
        # --- Round 1 Simulation (Allocation) ---
        "vote_round_1": vote_choice_legacy,
        "softmax_probs_round_1": jnp.zeros_like(vote_choice_legacy, dtype=float),
        "first_round_winners": top_two_winners,
        # --- Round 2 Simulation (Final Tally) ---
        "vote_final_round_2": vote_choice_legacy,
        "softmax_probs_final_round_2": jnp.zeros_like(vote_choice_legacy, dtype=float),
        "final_winner": final_winner,
        # --- QV Specific Data ---
        "total_votes_per_candidate": total_votes,
        "credits_spent": credits_spent,
    }


def _compute_sequential_qv_allocation(
    key: jax.random.PRNGKey, candidate_preferences: ArrayLike, budget: float
) -> tuple[ArrayLike, ArrayLike]:
    """
    Allocates budget sequentially across the top 5 choices without replacement.

    Agents sample their top choice, allocate a weighted portion of the budget,
    remove that choice from consideration, and repeat.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for sampling choices.
    candidate_preferences : ArrayLike
        Matrix (n_agents, n_candidates) of preference scores.
    budget : float
        Total credits to distribute per agent.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        votes_matrix : The final votes (sqrt(credits_spent)).
        credits_spent : The raw matrix of credits allocated.
    """
    num_agents, num_candidates = candidate_preferences.shape

    # Define budget distribution weights for the top 5 picks
    weights = jnp.array([0.50, 0.25, 0.15, 0.07, 0.03])
    # Normalize weights to ensure exact budget exhaustion
    weights = (weights / jnp.sum(weights)) * budget

    # Initialize state
    credits_spent = jnp.zeros((num_agents, num_candidates))
    current_prefs = candidate_preferences

    # Split key for 5 stochastic sampling steps
    keys = jax.random.split(key, 5)

    # Unrolled loop for top 5 selection
    for i in range(5):
        # 1. Sample choice based on current preferences
        # Assumes _sample_choice returns (indices, probs)
        choice_indices, _ = _sample_choice(keys[i], current_prefs)

        # 2. Create one-hot mask for the selected candidate
        choice_one_hot = jax.nn.one_hot(choice_indices, num_candidates)

        # 3. Allocate budget according to current rank weight
        credits_spent = credits_spent + (choice_one_hot * weights[i])

        # 4. Remove selected choice for next iteration (Sampling without replacement)
        # Subtracting a large value pushes Softmax prob to ~0
        current_prefs = current_prefs - (choice_one_hot * 1e9)

    # Apply Quadratic Voting rule: Votes = sqrt(Credits)
    votes_matrix = jnp.sqrt(credits_spent)

    return votes_matrix, credits_spent
