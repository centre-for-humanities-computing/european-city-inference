from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from eci.voting_system.plurality import (
    _find_top_two_winners,
    _vote_plurality,
    strategic_vote,
)


class TestPluralityVoting:
    """Test plurality voting."""

    def test_find_top_two_winners_clear_winner(self):
        """Test that the top 2 are found correctly when counts are distinct."""
        votes = jnp.array([1, 1, 1, 0])
        winners = _find_top_two_winners(votes, num_candidates=3)
        assert winners[0] == 1
        assert winners[1] == 0

    def test_find_top_two_winners_tie(self):
        """Test behavior when there is a tie for 2nd place."""
        votes = jnp.array([0, 0, 1, 2])
        winners = _find_top_two_winners(votes, num_candidates=3)

        assert winners[0] == 0
        assert winners[1] in [
            1,
            2,
        ]

    @patch("eci.voting_system.plurality._sample_choice")
    @patch("eci.voting_system.plurality._compute_preferences")
    @patch("eci.voting_system.plurality._extract_env_data_vectorized")
    def test_vote_plurality_round_logic(self, mock_extract, mock_compute, mock_sample):
        """Verifies the 2-Round Plurality Logic."""
        mock_extract.return_value = {}
        fake_prefs = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        mock_compute.return_value = (fake_prefs, "gap1", "gap2")

        def sample_side_effect(key, prefs):
            is_round_2 = jnp.any(prefs == -jnp.inf)

            if not is_round_2:
                votes = jnp.array([0, 1, 0])
                return votes, jnp.zeros_like(prefs)
            else:
                votes = jnp.array([0, 0, 0])
                return votes, jnp.zeros_like(prefs)

        mock_sample.side_effect = sample_side_effect

        key = jax.random.PRNGKey(42)
        env = MagicMock()
        results = _vote_plurality(env, key)

        assert results["final_winner"] == 0
        assert jnp.array_equal(
            jnp.sort(results["first_round_winners"]), jnp.array([0, 1])
        )

        assert mock_sample.call_count == 2

        args_round_2, _ = mock_sample.call_args
        prefs_passed_round_2 = args_round_2[1]

        assert jnp.all(prefs_passed_round_2[:, 2] == -jnp.inf)
        assert jnp.all(prefs_passed_round_2[:, 0] != -jnp.inf)

    @patch("eci.voting_system.plurality._vote_plurality")
    @patch("eci.voting_system.plurality._compute_preferences")
    @patch("eci.voting_system.plurality._extract_env_data_vectorized")
    def test_strategic_vote_weighting(self, _, mock_compute, mock_plurality_func):
        """Verifies that Strategic Vote."""
        poll_probs = jnp.array([[0.9, 0.1], [0.9, 0.1]])

        mock_poll_results = {"softmax_probs_round_1": poll_probs, "final_winner": 0}

        mock_final_results = {"final_winner": 0, "strategy_used": "yes"}

        mock_plurality_func.side_effect = [mock_poll_results, mock_final_results]

        base_prefs = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        mock_compute.return_value = (base_prefs, None, None)

        env = MagicMock()
        key = jax.random.PRNGKey(0)
        results = strategic_vote(env, key)

        assert mock_plurality_func.call_count == 2

        call_args_2 = mock_plurality_func.call_args_list[1]
        kwargs_2 = call_args_2.kwargs

        assert "custom_preferences" in kwargs_2

        expected_adjusted = jnp.array([[0.9, 0.1], [0.9, 0.1]])

        assert jnp.allclose(kwargs_2["custom_preferences"], expected_adjusted)
        assert results["strategy_used"] == "weighted_by_expected_results"
