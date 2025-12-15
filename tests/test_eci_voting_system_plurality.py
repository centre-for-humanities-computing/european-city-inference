from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from eci.voting_system.plurality import _find_top_two_winners, _vote_plurality


class MockCandidate:
    """A mock class representing a candidate."""

    def __init__(self, id, mean, precision):
        self.id = id
        self.policy = {"mean": jnp.array(mean), "precision": jnp.array(precision)}


class MockEnv:
    """A mock environment class."""

    def __init__(self, candidates, voters_count, preferences_idx, last_attributes):
        self.candidates = candidates
        self.voters = list(range(voters_count))
        self.preferences_idx = preferences_idx
        self.last_attributes = last_attributes
        self.use_theory_of_mind = False  # Default for test


def test_find_top_two_winners():
    """Tests the `_find_top_two_winners` function."""
    # Case 1: clear winners (0: 3 votes, 1: 2 votes, 2: 1 vote)
    votes = jnp.array([0, 0, 0, 1, 1, 2])
    winners = _find_top_two_winners(votes)
    assert winners[0] == 0
    assert winners[1] == 1

    # Case 2: tie (0 and 1 both have 1 vote)
    votes = jnp.array([0, 1])
    winners = _find_top_two_winners(votes)
    assert len(winners) == 2

    # Case 3: only one candidate (0 wins, 0 is runner-up)
    votes = jnp.array([0, 0])
    winners = _find_top_two_winners(votes)
    assert winners[0] == 0
    assert winners[1] == 0


def test_vote_plurality():
    """Tests simulating a two-round plurality election."""
    with (
        patch("eci.voting_system.plurality._get_current_beliefs"),
        patch("eci.voting_system.plurality._get_pref_belief_gap"),
        patch(
            "eci.voting_system.plurality._compute_option_preferences"
        ) as mock_compute_prefs,
    ):
        # Mocking preferences: C0 wins with 2 votes, C1 gets 1 vote, C2 gets 0.
        prefs = jnp.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        mock_compute_prefs.return_value = prefs

        # Setup mock environment and candidates with distinct IDs
        c0 = MagicMock()
        c0.id = 100
        c1 = MagicMock()
        c1.id = 101
        c2 = MagicMock()
        c2.id = 102
        env = MagicMock()
        env.candidates = [c0, c1, c2]
        key = jax.random.PRNGKey(42)

        results = _vote_plurality(env, key)

        # Assert Round 1 outcomes
        assert "first_round_winners" in results
        winners_r1 = results["first_round_winners"]
        # C0 and C1 should be the winners
        assert 100 in winners_r1
        assert 101 in winners_r1

        # Assert Final Round outcomes
        assert "final_winner" in results
        # C0 should be the ultimate winner
        assert results["final_winner"] == 100
