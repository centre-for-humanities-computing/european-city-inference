from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from eci.voting_system.random_voting import _find_top_two_winners, _vote_random


class TestRandomVoting:
    """Class testing random voting."""

    def test_find_top_two_winners(self):
        """Test finding top 2 winners from a vote array."""
        votes = jnp.array([0, 0, 1, 2, 0])
        num_candidates = 3
        winners = _find_top_two_winners(votes, num_candidates)
        assert len(winners) == 2
        assert winners[0] == 0
        assert winners[1] in [1, 2]

    @patch("eci.voting_system.random_voting._sample_choice")
    def test_vote_random_flow(self, mock_sample):
        """Test the full flow of _vote_random."""
        env = MagicMock()
        env.voters = [MagicMock()] * 5  # 5 Voters
        env.candidates = [MagicMock()] * 3  # 3 Candidates

        r1_votes = jnp.array([0, 1, 0, 1, 0])
        r1_probs = jnp.zeros((5, 3))

        r2_votes = jnp.array([0, 0, 0, 0, 0])
        r2_probs = jnp.zeros((5, 3))

        mock_sample.side_effect = [
            (r1_votes, r1_probs),  # Call 1
            (r2_votes, r2_probs),  # Call 2
        ]

        key = jax.random.PRNGKey(42)
        results = _vote_random(env, key)

        expected_keys = [
            "vote_round_1",
            "softmax_probs_round_1",
            "first_round_winners",
            "vote_final_round_2",
            "softmax_probs_final_round_2",
            "final_winner",
            "pref_candidate_gap",
            "candidate_preferences",
            "pref_belief_gap",
        ]
        for k in expected_keys:
            assert k in results, f"Missing key: {k}"

        winners = results["first_round_winners"]
        assert 0 in winners
        assert 1 in winners
        assert results["final_winner"] == 0

        assert results["candidate_preferences"].shape == (5, 3)
        assert mock_sample.call_count == 2

        args_round_2, _ = mock_sample.call_args_list[1]
        prefs_passed_round_2 = args_round_2[1]

        assert jnp.all(prefs_passed_round_2[:, 2] == -jnp.inf)

        assert not jnp.all(prefs_passed_round_2[:, 0] == -jnp.inf)
        assert not jnp.all(prefs_passed_round_2[:, 1] == -jnp.inf)
