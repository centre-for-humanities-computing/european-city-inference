from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from eci.voting_system.quadratic import (
    _compute_sequential_qv_allocation,
    _vote_quadratic,
)


def test_compute_sequential_qv_allocation():
    """Tests the `_compute_sequential_qv_allocation` function."""
    key = jax.random.PRNGKey(0)

    # 2 Agents, 3 Candidates
    # Agent 0 prefers C0 strongly
    # Agent 1 prefers C1 strongly

    candidate_preferences = jnp.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])

    budget = 100.0

    votes_matrix, credits_spent = _compute_sequential_qv_allocation(
        key, candidate_preferences, budget
    )

    assert votes_matrix.shape == (2, 3)
    assert credits_spent.shape == (2, 3)

    # Check total credits spent is close to budget
    spent_per_agent = jnp.sum(credits_spent, axis=1)
    assert jnp.allclose(spent_per_agent, budget, atol=1e-4)

    # Check that Agent 0 prioritized C0 heavily
    assert credits_spent[0, 0] >= 50.0


def test_vote_quadratic():
    """Tests the main `_vote_quadratic` function."""
    with (
        patch("eci.voting_system.quadratic._get_current_beliefs"),
        patch("eci.voting_system.quadratic._get_pref_belief_gap"),
        patch(
            "eci.voting_system.quadratic._compute_option_preferences"
        ) as mock_compute_prefs,
    ):
        # Setup mock preferences
        prefs = jnp.array([[10.0, 0.0], [0.0, 10.0]])
        mock_compute_prefs.return_value = prefs

        c0 = MagicMock()
        c0.id = 100
        c1 = MagicMock()
        c1.id = 101

        env = MagicMock()
        env.candidates = [c0, c1]

        key = jax.random.PRNGKey(42)

        results = _vote_quadratic(env, key, budget=100.0)

        assert "vote_round_1" in results
        assert "final_winner" in results
        assert "credits_spent" in results
        assert "total_votes_per_candidate" in results

        vote_r1 = results["vote_round_1"]
        # Agent 0 preferred C0 (ID 100)
        assert vote_r1[0] == 100
        # Agent 1 preferred C1 (ID 101)
        assert vote_r1[1] == 101
