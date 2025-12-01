import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.core_logic import (
    kl_divergence,  # Make sure this function handles JAX broadcasting
)


def _vote_plurality(env, key, *args, **kwargs) -> dict:
    """Perform plurality vote for each agent."""
    # 1. Extract all agent beliefs and preferences
    all_agent_data = _get_current_beliefs_t(env)

    # 2. Get stacked beliefs and dissatisfaction per agent
    # dissatisfaction_per_agent -> Shape (num_agents,)
    # beliefs_mean_t            -> Shape (num_agents, num_preferences)
    # beliefs_precision_t       -> Shape (num_agents, num_preferences)
    (
        dissatisfaction_per_agent,
        beliefs_mean_t,
        beliefs_precision_t,
    ) = _get_current_dissatisfaction(env, all_agent_data)

    # 3. Evaluate candidate scores FOR EACH agent
    # candidate_preferences -> Shape (num_agents, num_candidates)
    candidate_preferences = _evaluate_candidate_scores(
        env,
        beliefs_mean_t,
        beliefs_precision_t,
        dissatisfaction_per_agent,  # Pass the per-agent vector
    )

    # --- ROUND 1 ---

    # 4. Split the JAX key for two separate random samples
    key_round_1, key_round_2 = jax.random.split(key)

    # 5. Create mask for round 1 (all candidates are eligible)
    mask_round_1 = jnp.ones_like(candidate_preferences, dtype=bool)

    # 6. Sample round 1 vote
    # vote_1.shape -> (num_agents,)
    vote_1, softmax_probs_1 = _sample_vote(
        key_round_1, mask_round_1, candidate_preferences
    )

    # 7. Find the top two winners from round 1
    # top_two_winners.shape -> (2,)
    top_two_winners = _find_top_two_winners(vote_1)

    # --- ROUND 2 ---

    # 8. Create mask for round 2 (only top two are eligible)
    num_candidates = candidate_preferences.shape[1]
    all_candidates_indices = jnp.arange(num_candidates)

    # jnp.isin creates a 1D boolean mask
    mask_round_2_1d = jnp.isin(all_candidates_indices, top_two_winners)

    # Broadcast this 1D mask to all agents
    mask_round_2 = jnp.broadcast_to(mask_round_2_1d, candidate_preferences.shape)

    # 9. Sample final (round 2) vote
    vote_2, softmax_probs_2 = _sample_vote(
        key_round_2, mask_round_2, candidate_preferences
    )

    # 10. Find the final winner (most votes in round 2)
    # We reuse _find_top_two_winners and just take the first one [0]
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
        # --- General ---
        "dissatisfaction": dissatisfaction_per_agent,
    }


# --- Helper Functions ---


def _find_top_two_winners(votes_array: ArrayLike) -> ArrayLike:
    """Find two candidates with the most votes."""
    # 1. Count votes for each unique candidate
    unique_candidates, counts = jnp.unique(votes_array, return_counts=True)

    # 2. Get indices that would sort the counts in ascending order
    sorted_indices = jnp.argsort(counts)

    # 3. Take the last two indices (top 2 counts) and reverse them
    top_two_indices = sorted_indices[-2:][::-1]

    # 4. Get the corresponding candidates
    top_two_winners = unique_candidates[top_two_indices]

    # Handle edge case: if < 2 unique candidates voted for
    num_winners = top_two_winners.shape[0]
    if num_winners < 2:
        # Pad the array to ensure shape is (2,)
        # e.g., if [1] is winner, returns [1, 1]
        return jnp.pad(top_two_winners, (0, 2 - num_winners), mode="edge")

    return top_two_winners


def _get_current_beliefs_t(env) -> dict:
    """Extract the current beliefs and underlying preferences."""
    all_agent_data = {}

    # Use len(env.voters) or env.NUM_VOTERS
    for agent in range(len(env.voters)):
        means_belief_by_preference = []
        precisions_belief_by_preference = []
        agent_pref_means = []
        agent_pref_precisions = []

        for preference_idx in env.preferences_idx:
            means_belief_by_preference.append(
                env.last_attributes[preference_idx]["expected_mean"][agent]
            )
            precisions_belief_by_preference.append(
                env.last_attributes[preference_idx]["precision"][agent]
            )
            agent_pref_means.append(
                env.last_attributes[-1]["preferences"]["mean"][agent][preference_idx]
            )
            agent_pref_precisions.append(
                env.last_attributes[-1]["preferences"]["precision"][agent][
                    preference_idx
                ]
            )

        # Assign the JAX arrays for this agent to the dictionary
        all_agent_data[agent] = {
            "means_belief": jnp.array(means_belief_by_preference),
            "precisions_belief": jnp.array(precisions_belief_by_preference),
            "agent_pref_means": jnp.array(agent_pref_means),
            "agent_pref_precisions": jnp.array(agent_pref_precisions),
        }

    # Return the full dictionary after the loop
    return all_agent_data


def _get_current_dissatisfaction(
    env, all_agent_data: dict
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute dissatisfaction FOR EACH AGENT and returns stacked arrays."""
    # 1. Extract and stack arrays from each agent
    #    Result: 4 arrays of shape (num_agents, num_preferences)
    beliefs_mean_t = jnp.stack(
        [agent_data["means_belief"] for agent_data in all_agent_data.values()]
    )
    beliefs_precision_t = jnp.stack(
        [agent_data["precisions_belief"] for agent_data in all_agent_data.values()]
    )
    agent_pref_mean = jnp.stack(
        [agent_data["agent_pref_means"] for agent_data in all_agent_data.values()]
    )
    agent_pref_precision = jnp.stack(
        [agent_data["agent_pref_precisions"] for agent_data in all_agent_data.values()]
    )

    # 2. Calculate dissatisfaction (vectorized)
    #    Result: (num_agents, num_preferences)
    all_dissatisfactions = kl_divergence(
        beliefs_mean_t,
        beliefs_precision_t,
        agent_pref_mean,
        agent_pref_precision,
    )

    # 3. Sum over the preferences axis (axis=1) to get ONE score per agent
    #    Result: (num_agents,)
    dissatisfaction_per_agent = jnp.sum(all_dissatisfactions, axis=1)

    # 4. Return everything needed for subsequent steps
    return dissatisfaction_per_agent, beliefs_mean_t, beliefs_precision_t


def _evaluate_candidate_scores(
    env,
    beliefs_mean_t: ArrayLike,  # Shape: (num_agents, num_prefs)
    beliefs_precision_t: ArrayLike,  # Shape: (num_agents, num_prefs)
    dissatisfaction_per_agent: ArrayLike,  # Shape: (num_agents,)
) -> ArrayLike:
    """Evaluate the preference score for each candidate."""
    candidate_preferences_per_agent = []
    candidate_list = [(c.policy["mean"], c.policy["precision"]) for c in env.candidates]

    for candidate_mean_pref, candidate_precision_pref in candidate_list:
        candidate_mean_pref = candidate_mean_pref.reshape(-1)
        candidate_precision_pref = candidate_precision_pref.reshape(-1)

        # Calculate KL for all agents against this one candidate
        expected_dissatisfaction = kl_divergence(
            beliefs_mean_t,
            beliefs_precision_t,
            candidate_mean_pref,
            candidate_precision_pref,
        )

        # Sum over preferences (axis=1) to get a per-agent score
        # (num_agents, num_prefs) -> (num_agents,)
        expected_dissatisfaction_per_agent = jnp.sum(expected_dissatisfaction, axis=1)

        # Score (reduction) = Current (per-agent) - Expected (per-agent)
        # (num_agents,) - (num_agents,) -> (num_agents,)
        preference_score_per_agent = (
            dissatisfaction_per_agent - expected_dissatisfaction_per_agent
        )
        candidate_preferences_per_agent.append(preference_score_per_agent)

    # Stack the candidate scores
    # List of N arrays (num_agents,) -> Array (num_candidates, num_agents)
    # .T (Transpose) -> Array (num_agents, num_candidates)
    return jnp.stack(candidate_preferences_per_agent).T


def _sample_vote(
    key, mask: ArrayLike, preferences: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Sample a vote using a softmax over masked preferences."""
    # Set non-masked preferences to -infinity
    masked_preferences = jnp.where(mask, preferences, -jnp.inf)

    # Calculate softmax per agent (per-row, axis=1)
    # Result: (num_agents, num_candidates)
    softmax_probs = jax.nn.softmax(masked_preferences, axis=1)

    # Sample a vote per-row (per-agent, axis=1)
    # categorical takes logits (pre-softmax values)
    # Result: (num_agents,)
    vote = jax.random.categorical(key, masked_preferences, axis=1)

    return vote, softmax_probs


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
