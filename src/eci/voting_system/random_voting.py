import jax
import jax.numpy as jnp

from eci.utils import _extract_env_data_vectorized, _find_top_k_winners, _find_winner
from eci.voting_system.decisions import _compute_preferences


def _vote_uniform_random(env, key, *args, **kwargs) -> dict:
    """Uniform random voting.

    Each agent picks a candidate uniformly at random in both rounds,
    independently of any preference.
    """
    num_agents = len(env.voters)
    num_candidates = len(env.candidates)

    key_r1, key_r2 = jax.random.split(key)

    # uniform pick over all candidates
    vote_1 = jax.random.randint(key_r1, (num_agents,), 0, num_candidates)
    softmax_probs_1 = jnp.full((num_agents, num_candidates), 1.0 / num_candidates)
    top_two_winners = _find_top_k_winners(vote_1, num_candidates, k=2)

    # uniform pick between the two finalists
    idx = jax.random.randint(key_r2, (num_agents,), 0, 2)
    vote_2 = top_two_winners[idx]
    mask = jnp.isin(jnp.arange(num_candidates), top_two_winners)
    softmax_probs_2 = jnp.broadcast_to(
        jnp.where(mask, 0.5, 0.0), (num_agents, num_candidates)
    )
    final_winner = _find_winner(vote_2, num_candidates)

    # Real preferences from the env used only for metric computation
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
