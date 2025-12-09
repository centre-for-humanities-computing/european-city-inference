import jax.numpy as jnp
from jax.typing import ArrayLike

from eci.utils import kl_divergence


def _vote_quadratic(env, key, budget: float = 99.0, *args, **kwargs) -> dict:
    """Apply Quadratic Voting."""
    # Extract all agent beliefs and preferences
    all_agent_data = _get_current_beliefs_t(env)

    # Get stacked beliefs and dissatisfaction per agent
    (
        dissatisfaction_per_agent,
        beliefs_mean_t,
        beliefs_precision_t,
    ) = _get_current_dissatisfaction(env, all_agent_data)

    # Evaluate candidate scores FOR EACH agent
    candidate_preferences = _evaluate_candidate_scores(
        env,
        beliefs_mean_t,
        beliefs_precision_t,
        dissatisfaction_per_agent,
    )

    # --- QUADRATIC VOTING LOGIC ---

    # Filter negative preferences
    # Agents only spend credits on candidates that reduce dissatisfaction (score > 0).
    positive_preferences = jnp.maximum(candidate_preferences, 0.0)

    # Normalize preferences to determine budget allocation
    sum_preferences = jnp.sum(positive_preferences, axis=1, keepdims=True)

    # Avoid division by zero for agents who hate everyone (sum = 0)
    safe_sum = jnp.where(sum_preferences == 0, 1.0, sum_preferences)

    # Proportion of budget allocated to each candidate
    proportions = positive_preferences / safe_sum

    # If sum was 0, proportion should be 0, not result of division
    proportions = jnp.where(sum_preferences == 0, 0.0, proportions)

    # Spend Credits & Calculate Quadratic Votes
    # Credits spent = Budget * Proportion
    credits_spent = proportions * budget

    # Votes = SquareRoot(Credits)
    # This is the defining math of Quadratic Voting
    votes_matrix = jnp.sqrt(credits_spent)

    # Determine Winners
    # Sum the weighted votes across all agents for each candidate
    total_votes_per_candidate = jnp.sum(votes_matrix, axis=0)

    # Find indices of top winners
    sorted_indices = jnp.argsort(total_votes_per_candidate)[::-1]

    # Get top two indices
    top_two_indices = sorted_indices[:2]

    # Map indices back to candidate IDs
    candidate_ids = jnp.array([c.id for c in env.candidates])
    top_two_winners = candidate_ids[top_two_indices]

    if top_two_winners.shape[0] < 2:
        top_two_winners = jnp.pad(
            top_two_winners, (0, 2 - top_two_winners.shape[0]), mode="edge"
        )

    # --- RESULTS ---
    return {
        "vote_matrix": votes_matrix,
        "total_votes": total_votes_per_candidate,
        "winners": top_two_winners,
        "credits_spent": credits_spent,
        "proportions": proportions,
        "dissatisfaction": dissatisfaction_per_agent,
        "vote_choice_legacy": jnp.argmax(votes_matrix, axis=1),
    }


# --- Helper Functions ---


def _get_current_beliefs_t(env) -> dict:
    """Extract the current beliefs and underlying preferences."""
    all_agent_data = {}
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

        all_agent_data[agent] = {
            "means_belief": jnp.array(means_belief_by_preference),
            "precisions_belief": jnp.array(precisions_belief_by_preference),
            "agent_pref_means": jnp.array(agent_pref_means),
            "agent_pref_precisions": jnp.array(agent_pref_precisions),
        }
    return all_agent_data


def _get_current_dissatisfaction(
    env, all_agent_data: dict
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute dissatisfaction FOR EACH AGENT and returns stacked arrays."""
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

    all_dissatisfactions = kl_divergence(
        beliefs_mean_t,
        beliefs_precision_t,
        agent_pref_mean,
        agent_pref_precision,
    )
    dissatisfaction_per_agent = jnp.sum(all_dissatisfactions, axis=1)
    return dissatisfaction_per_agent, beliefs_mean_t, beliefs_precision_t


def _evaluate_candidate_scores(
    env,
    beliefs_mean_t: ArrayLike,
    beliefs_precision_t: ArrayLike,
    dissatisfaction_per_agent: ArrayLike,
) -> ArrayLike:
    """Evaluate the preference score for each candidate."""
    candidate_preferences_per_agent = []
    candidate_list = [(c.policy["mean"], c.policy["precision"]) for c in env.candidates]

    for candidate_mean_pref, candidate_precision_pref in candidate_list:
        candidate_mean_pref = candidate_mean_pref.reshape(-1)
        candidate_precision_pref = candidate_precision_pref.reshape(-1)

        expected_dissatisfaction = kl_divergence(
            beliefs_mean_t,
            beliefs_precision_t,
            candidate_mean_pref,
            candidate_precision_pref,
        )

        expected_dissatisfaction_per_agent = jnp.sum(expected_dissatisfaction, axis=1)

        preference_score_per_agent = (
            dissatisfaction_per_agent - expected_dissatisfaction_per_agent
        )
        candidate_preferences_per_agent.append(preference_score_per_agent)

    return jnp.stack(candidate_preferences_per_agent).T
