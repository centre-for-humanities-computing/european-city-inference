import numpy as np
import pandas as pd
import pytest

from eci.adapter import SimulationAdapter


# --- Mock Classes to simulate your Environment ---
class MockCandidate:
    """Create MockCandidate."""

    def __init__(self, c_id, mean, precision):
        self.id = c_id
        self.policy = {"mean": mean, "precision": precision}


class MockVoter:
    """Create MockVoter."""

    def __init__(self, v_id, mean, precision):
        self.id = v_id
        self.preferences = {"mean": mean, "precision": precision}
        # Simulate the deep structure of the HGF trajectory
        self.trajectory = {
            "expected_mean": {v_id: np.array([0.1, 0.2, 0.3])},
            "precision": {v_id: np.array([1.0, 1.0, 1.0])},
        }


class MockEnvironment:
    """Create MockEnvironement."""

    def __init__(self):
        self.candidates = [
            MockCandidate(0, [0.5], [10.0]),
            MockCandidate(1, [-0.5], [5.0]),
        ]
        self.voters = [MockVoter(100, [0.2], [2.0]), MockVoter(101, [-0.2], [2.0])]
        # Simulate observations (time_steps x num_preferences)
        self.input_data = np.array([[0.5], [0.6], [0.4]])


# --- Tests ---


class TestSimulationAdapter:
    """Create SimulationAdaptater."""

    @pytest.fixture
    def mock_env(self):
        """Provide fake environment for testing."""
        return MockEnvironment()

    def test_prepare_preference_data(self, mock_env):
        """Test if preference distributions are generated correctly."""
        df = SimulationAdapter.prepare_preference_data(mock_env, num_voters=1)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        # Check Columns
        expected_cols = ["group", "id", "preference", "x", "pdf"]
        for col in expected_cols:
            assert col in df.columns

        # Check Logic:
        # We have 2 candidates + 1 voter (limit set to 1) = 3 entities
        # Each entity has 1 preference dimension.
        # The linspace generates 400 points.
        # Total rows should be 3 * 400 = 1200
        assert len(df) == 1200

        # Check Groups
        assert "Candidate" in df["group"].values
        assert "Voter" in df["group"].values

        # Check specific ID presence
        assert "C0" in df["id"].values
        assert "V100" in df["id"].values

    def test_extract_vote_counts_standard(self):
        """Test vote extraction with standard numbered rounds."""
        # Create a dummy DataFrame representing simulation output
        data = {
            "simulation_id": [1],
            "vote_round_1": [[0, 0, 1, 1]],  # Tie between C0 and C1
            "vote_round_2": [[0, 0, 0, 1]],  # C0 wins
            "irrelevant_col": ["garbage"],
        }
        env_df = pd.DataFrame(data)

        results = SimulationAdapter.extract_vote_counts(env_df)

        assert len(results) == 4  # Round 1 (2 cands) + Round 2 (2 cands)

        # Check Round 1 (50/50 split)
        r1_c0 = next(r for r in results if r["round"] == 1 and r["candidate_id"] == 0)
        assert r1_c0["proportion"] == 0.5
        assert r1_c0["total_votes"] == 4

        # Check Round 2 (75/25 split)
        r2_c0 = next(r for r in results if r["round"] == 2 and r["candidate_id"] == 0)
        assert r2_c0["proportion"] == 0.75

    def test_extract_vote_counts_named_rounds(self):
        """Test extraction when column doesn't have a number (fallback logic)."""
        data = {"vote_final": [[1, 1, 1]]}
        env_df = pd.DataFrame(data)

        results = SimulationAdapter.extract_vote_counts(env_df)

        assert len(results) == 1
        assert results[0]["round"] == "vote_final"
        assert results[0]["candidate_id"] == 1
        assert results[0]["proportion"] == 1.0

    def test_extract_vote_counts_empty_or_none(self):
        """Test robustness against None or empty vote lists."""
        data = {
            "vote_round_1": [None],  # Should be handled safely
            "vote_round_2": [[]],  # Empty list
        }
        env_df = pd.DataFrame(data)

        results = SimulationAdapter.extract_vote_counts(env_df)
        assert len(results) == 0  # Should return empty list, no crash

    def test_get_voter_trajectory_data(self, mock_env):
        """Test if trajectory data is correctly extracted for a specific voter."""
        voter_id = 100
        data = SimulationAdapter.get_voter_trajectory_data(mock_env, voter_id)

        assert isinstance(data, dict)

        # Check keys
        expected_keys = [
            "means",
            "precisions",
            "observations",
            "preference_params",
            "title_suffix",
        ]
        for k in expected_keys:
            assert k in data

        # Check values
        # Means should match the MockVoter trajectory
        np.testing.assert_array_equal(data["means"], np.array([0.1, 0.2, 0.3]))

        # Observations should match MockEnvironment input_data
        np.testing.assert_array_equal(data["observations"], np.array([0.5, 0.6, 0.4]))

        # Preference params should be the tuple (mean, precision)
        assert data["preference_params"] == (0.2, 2.0)

        assert "Voter 100" in data["title_suffix"]

    def test_get_voter_trajectory_data_not_found(self, mock_env):
        """Testfor a non-existent voter raises StopIteration."""
        with pytest.raises(StopIteration):
            SimulationAdapter.get_voter_trajectory_data(mock_env, voter_id=999)
