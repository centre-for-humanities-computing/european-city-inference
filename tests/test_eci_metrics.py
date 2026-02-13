import jax.numpy as jnp
import numpy as np
import pytest

from eci.metrics import (
    _vote_efficiency,
    _winner_satisfaction,
)


class TestWinnerSatisfaction:
    """Unit tests for the _winner_satisfaction function."""

    def test_nominal_case(self):
        """Sum of preferences for the winning candidate."""
        prefs = np.array([[3, 5, 2, 3, 2, 4], [2, 1, 4, 0, 3, 5]])
        winner_idx = 2
        assert _winner_satisfaction(prefs, winner_idx) == 6

    # Boundary cases: First and last candidate
    @pytest.mark.parametrize(
        "winner_idx, expected_sum",
        [
            (0, 5),  # Column 0: 3 + 2 = 5
            (5, 9),  # Column 5: 4 + 5 = 9
            (-1, 9),  # Negative index (last element) supported by numpy/jax
        ],
    )
    def test_boundaries_indices(self, winner_idx, expected_sum):
        """Check that slicing works at the boundaries."""
        prefs = np.array([[3, 5, 2, 3, 2, 4], [2, 1, 4, 0, 3, 5]])
        assert _winner_satisfaction(prefs, winner_idx) == expected_sum

    # Input types: JAX vs NumPy compatibility
    def test_jax_array_compatibility(self):
        """Check that the function accepts both JAX and NumPy arrays."""
        prefs_np = np.array([[1, 2], [3, 4]])
        prefs_jax = jnp.array([[1, 2], [3, 4]])
        winner = 1

        res_np = _winner_satisfaction(prefs_np, winner)
        res_jax = _winner_satisfaction(prefs_jax, winner)

        assert res_np == 6
        assert res_jax == 6

    # Empty or zero values
    def test_zeros_preferences(self):
        """Check behavior if all preferences are zero."""
        prefs = np.zeros((5, 3))  # 5 agents, 3 candidates
        assert _winner_satisfaction(prefs, 0) == 0

    def test_single_agent(self):
        """Check the case with a single agent (1 row)."""
        prefs = np.array([[10, 20, 30]])
        # If only one agent, the winner's satisfaction is just their score
        assert _winner_satisfaction(prefs, 1) == 20

    # Error handling (Robustness)
    def test_index_out_of_bounds(self):
        """Check that an error is raised if the winner index does not exist."""
        prefs = np.array([[1, 2], [3, 4]])  # 2 columns (index 0 and 1)

        # Try to access column 99
        with pytest.raises(IndexError):
            _winner_satisfaction(prefs, 99)


class TestVoteEfficiency:
    """Unit tests for the _vote_efficiency function."""

    def test_single_agent_single_vote(self):
        """Test with a single voter casting a single vote for one candidate."""
        candidate_preferences = np.array([[0.1, 0.5, 0.2]])  # Preferences: [A, B, C]
        votes_matrix = np.array([[1, 0, 0]])  # 1 vote for A
        expected_efficiency = 0.1  # 0.1 (preference for A) / 1 (total votes)
        assert (
            _vote_efficiency(candidate_preferences, votes_matrix) == expected_efficiency
        )

    def test_multiple_agents_single_vote(self):
        """Test with multiple voters, each casting a single vote for one candidate."""
        candidate_preferences = np.array(
            [
                [0.1, 0.5, 0.2],  # Voter 1
                [0.3, 0.2, 0.8],  # Voter 2
                [0.4, 0.7, 0.1],  # Voter 3
            ]
        )
        votes_matrix = np.array(
            [
                [1, 0, 0],  # Voter 1: 1 vote for A
                [0, 1, 0],  # Voter 2: 1 vote for B
                [0, 0, 1],  # Voter 3: 1 vote for C
            ]
        )
        # Expected efficiency: (0.1 + 0.2 + 0.1) = 0.4
        expected_efficiency = 0.4
        assert _vote_efficiency(candidate_preferences, votes_matrix) == pytest.approx(
            expected_efficiency
        )

    def test_quadratic_voting_multiple_votes(self):
        """Test with voters distributing their votes across multiple candidates."""
        candidate_preferences = np.array(
            [
                [0.1, 0.5, 0.2],  # Voter 1
                [0.3, 0.2, 0.8],  # Voter 2
            ]
        )
        # Voter 1: 4 votes for A, 2 votes for B
        # Voter 2: 2 votes for B, 1 vote for C
        votes_matrix = np.array(
            [
                [4, 2, 0],  # Voter 1: 4 for A, 2 for B
                [0, 2, 1],  # Voter 2: 2 for B, 1 for C
            ]
        )
        expected_efficiency = 0.6333
        assert _vote_efficiency(candidate_preferences, votes_matrix) == pytest.approx(
            expected_efficiency
        )

    def test_no_votes(self):
        """Test with a voter casting no votes."""
        candidate_preferences = np.array([[0.1, 0.5, 0.2]])
        votes_matrix = np.array([[0, 0, 0]])  # No votes
        expected_efficiency = 0.0
        assert (
            _vote_efficiency(candidate_preferences, votes_matrix) == expected_efficiency
        )

    def test_zero_preferences(self):
        """Test with all candidate preferences set to zero."""
        candidate_preferences = np.array([[0.0, 0.0, 0.0]])
        votes_matrix = np.array([[1, 0, 0]])  # 1 vote for A (preference = 0)
        expected_efficiency = 0.0
        assert (
            _vote_efficiency(candidate_preferences, votes_matrix) == expected_efficiency
        )

    def test_fractional_votes(self):
        """Test with fractional votes (e.g., 0.5 vote)."""
        candidate_preferences = np.array([[0.1, 0.5, 0.2]])
        votes_matrix = np.array([[0.5, 0.3, 0.2]])
        expected_efficiency = 0.24
        assert _vote_efficiency(candidate_preferences, votes_matrix) == pytest.approx(
            expected_efficiency
        )
