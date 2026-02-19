from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from eci.voting_system.quadratic import (
    _compute_sequential_qv_allocation,
    _vote_quadratic,
    strategic_quadratic_vote,
)


class TestQuadraticVoting:
    """Test quadratic voting."""

    @patch("eci.voting_system.quadratic._sample_choice")
    def test_compute_sequential_qv_allocation_math(self, mock_sample):
        """Verify that credits are allocated according to the fixed weights."""
        key = jax.random.PRNGKey(42)
        budget = 100.0
        prefs = jnp.zeros((2, 2))
        mock_sample.side_effect = [
            (jnp.array([0, 1]), None),  # Iter 1
            (jnp.array([0, 1]), None),  # Iter 2
            (jnp.array([0, 1]), None),  # Iter 3
            (jnp.array([0, 1]), None),  # Iter 4
            (jnp.array([0, 1]), None),  # Iter 5
        ]

        votes, credits_spent = _compute_sequential_qv_allocation(key, prefs, budget)

        assert jnp.isclose(credits_spent[0, 0], 100.0)
        assert jnp.isclose(credits_spent[1, 1], 100.0)

        assert votes[0, 0] == 10
        assert votes[1, 1] == 10

        assert mock_sample.call_count == 5

    @patch("eci.voting_system.quadratic._compute_sequential_qv_allocation")
    @patch("eci.voting_system.quadratic._compute_preferences")
    @patch("eci.voting_system.quadratic._extract_env_data_vectorized")
    def test_vote_quadratic_winner_logic(self, mock_extract, mock_compute, mock_alloc):
        """Test that _vote_quadratic correctly identifies the winner."""
        env = MagicMock()
        c1 = MagicMock()
        c1.id = 10
        c2 = MagicMock()
        c2.id = 20
        env.candidates = [c1, c2]

        mock_extract.return_value = {}
        mock_compute.return_value = (jnp.zeros((2, 2)), None, None)
        mock_votes_matrix = jnp.array([[10, 0], [5, 0]])
        mock_credits = jnp.zeros((2, 2))
        mock_alloc.return_value = (mock_votes_matrix, mock_credits)

        results = _vote_quadratic(env, jax.random.PRNGKey(0))

        assert results["final_winner"] == 10
        assert jnp.array_equal(results["qv_votes_matrix"], mock_votes_matrix)
        assert jnp.all(results["vote_round_1"] == 10)

    @patch("eci.voting_system.quadratic._vote_quadratic")
    @patch("eci.voting_system.quadratic._compute_preferences")
    @patch("eci.voting_system.quadratic._extract_env_data_vectorized")
    def test_strategic_quadratic_vote_flow(self, _, mock_compute, mock_qv_func):
        """Test the strategic voting."""
        poll_probs = jnp.array([[0.8, 0.2], [0.8, 0.2]])

        poll_return = {
            "softmax_probs_round_1": poll_probs,
            "total_votes_per_candidate": jnp.array([100, 20]),
        }
        final_return = {"final_winner": 10}

        mock_qv_func.side_effect = [poll_return, final_return]

        base_prefs = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        mock_compute.return_value = (base_prefs, None, None)

        env = MagicMock()
        key = jax.random.PRNGKey(0)
        strategic_quadratic_vote(env, key)

        assert mock_qv_func.call_count == 2, (
            f"Expected 2 calls, got {mock_qv_func.call_count}"
        )

        _, kwargs_2 = mock_qv_func.call_args_list[1]

        assert "custom_preferences" in kwargs_2

        expected_adjusted = jnp.array([[0.8, 0.2], [0.8, 0.2]])

        assert jnp.allclose(kwargs_2["custom_preferences"], expected_adjusted)
