import jax
import jax.numpy as jnp

from eci.utils import _extract_env_data_vectorized, _find_top_k_winners, _find_winner
from eci.voting_system.decisions import _compute_preferences, _sample_choice


def _vote_uniform_random(env, key, *args, **kwargs) -> dict:
    """Uniform random voting — pure chance baseline.

    Each agent picks a candidate uniformly at random in both rounds,
    independently of any preference. Metrics are still computed against
    the env's real preferences/beliefs so comparisons with real voting
    systems are meaningful.
    """
    num_agents = len(env.voters)
    num_candidates = len(env.candidates)

    key_r1, key_r2 = jax.random.split(key)

    # --- ROUND 1: uniform pick over all candidates ---
    vote_1 = jax.random.randint(key_r1, (num_agents,), 0, num_candidates)
    softmax_probs_1 = jnp.full((num_agents, num_candidates), 1.0 / num_candidates)
    top_two_winners = _find_top_k_winners(vote_1, num_candidates, k=2)

    # --- ROUND 2: uniform pick between the two finalists ---
    idx = jax.random.randint(key_r2, (num_agents,), 0, 2)
    vote_2 = top_two_winners[idx]
    mask = jnp.isin(jnp.arange(num_candidates), top_two_winners)
    softmax_probs_2 = jnp.broadcast_to(
        jnp.where(mask, 0.5, 0.0), (num_agents, num_candidates)
    )
    final_winner = _find_winner(vote_2, num_candidates)

    # Real preferences from the env — used only for metric computation,
    # not for the vote itself
    agent_data = _extract_env_data_vectorized(env)
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    return {
        "vote_round_1": vote_1,
        "softmax_probs_round_1": softmax_probs_1,
        "first_round_winners": top_two_winners,
        "vote_final_round_2": vote_2,
        "softmax_probs_final_round_2": softmax_probs_2,
        "final_winner": final_winner,
        "candidate_preferences": candidate_preferences,
        "pref_candidate_gap": pref_candidate_gap,
        "pref_belief_gap": pref_belief_gap,
    }


def _vote_random_preferences(env, key, *args, **kwargs) -> dict:
    """Score-based voting on uniformly random preferences.

    Agents vote via the same softmax mechanism as the real system, but
    using random "preferences" instead of their real ones. Metrics are
    still computed against the env's real preferences/beliefs so the
    welfare comparison with other systems is meaningful.

    Answers: "does the *content* of preferences matter, holding the
    decision mechanism fixed?"
    """
    num_agents = len(env.voters)
    num_candidates = len(env.candidates)

    key_prefs, key_r1, key_r2 = jax.random.split(key, 3)

    # Fake (random) preferences — used ONLY to drive the vote
    fake_prefs = jax.random.uniform(key_prefs, shape=(num_agents, num_candidates))

    # --- ROUND 1 ---
    vote_1, softmax_probs_1 = _sample_choice(key_r1, fake_prefs)
    top_two = _find_top_k_winners(vote_1, num_candidates, k=2)

    # --- ROUND 2: mask out non-finalists ---
    mask = jnp.isin(jnp.arange(num_candidates), top_two)
    masked = jnp.where(mask, fake_prefs, -jnp.inf)
    vote_2, softmax_probs_2 = _sample_choice(key_r2, masked)
    final_winner = _find_winner(vote_2, num_candidates)

    # Real preferences from the env — used only for metric computation
    agent_data = _extract_env_data_vectorized(env)
    candidate_preferences, pref_candidate_gap, pref_belief_gap = _compute_preferences(
        agent_data
    )

    return {
        "vote_round_1": vote_1,
        "softmax_probs_round_1": softmax_probs_1,
        "first_round_winners": top_two,
        "vote_final_round_2": vote_2,
        "softmax_probs_final_round_2": softmax_probs_2,
        "final_winner": final_winner,
        "candidate_preferences": candidate_preferences,
        "pref_candidate_gap": pref_candidate_gap,
        "pref_belief_gap": pref_belief_gap,
    }
