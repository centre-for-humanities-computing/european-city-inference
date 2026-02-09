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
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
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
    """Compute token allocaiton.

    Parameters
    ----------
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    candidate_preferences:
        Preference for each candidate.
    budget:
        Token to allocate.

    Returns
    -------
        vote allocation and credit spent.
    """
    num_agents, num_candidates = candidate_preferences.shape

    # Weights strategy
    weights = jnp.array([0.50, 0.25, 0.15, 0.07, 0.03])
    weights = (weights / jnp.sum(weights)) * budget
    credits_spent = jnp.zeros((num_agents, num_candidates))
    current_prefs = candidate_preferences
    keys = jax.random.split(key, 5)

    # Sequentially allocate votes based on preferences
    for i in range(5):
        # Sample choice based on current preferences
        choice_indices, _ = _sample_choice(keys[i], current_prefs)
        # Create one-hot encoding for chosen candidates
        choice_one_hot = jax.nn.one_hot(choice_indices, num_candidates)
        # Distribute credits spent based on weights
        credits_spent = credits_spent + (choice_one_hot * weights[i])
        # Remove chosen candidate from current preferences
        current_prefs = current_prefs - (choice_one_hot * 1e9)

    # Convert credits spent to integer votes using quadratic cost
    votes_matrix = jnp.floor(jnp.sqrt(credits_spent)).astype(jnp.int32)

    return votes_matrix, credits_spent


def strategic_quadratic_vote(env, key, budget: float = 99.0, *args, **kwargs) -> dict:
    """Perform quadratic strategic voting.

    Parameters
    ----------
    env:
        The environment object.
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    budget:
        Token for quadratic voting.
    args:
        Variable length argument list.
    kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
        vote data.
    """
    # Simulate classic QV to predict expected winners
    expected_results = _vote_quadratic(env, key, budget, *args, **kwargs)
    expected_winner = expected_results["final_winner"]
    total_votes_per_candidate = expected_results["total_votes_per_candidate"]

    # Extract agent preferences
    agent_data = _extract_env_data_vectorized(env)
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    # Weight preferences toward expected winner or top candidates
    strategic_factor = 2.0  # Boost factor for strategic candidates
    adjusted_preferences = candidate_preferences * (
        1 + (candidate_preferences == expected_winner) * (strategic_factor - 1)
    )
    kwargs["custom_preferences"] = adjusted_preferences
    # Re-run QV with adjusted preferences
    new_key = jax.random.PRNGKey(
        jax.random.randint(key, (), 0, 1000000)
    )  # Fresh random key
    strategic_results = _vote_quadratic(env, new_key, budget, *args, **kwargs)

    # Return strategic results + expected outcomes
    return {
        **strategic_results,
        "expected_winner": expected_winner,
        "expected_total_votes": total_votes_per_candidate,
        "strategy_used": "weighted_by_expected_results",
    }
