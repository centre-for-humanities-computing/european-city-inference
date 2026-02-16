import jax.numpy as jnp
import pandas as pd
import pytest

from eci.metrics import (
    _vote_efficiency,
    _winner_satisfaction,
    batch_compute_metrics,
    compute_metrics,
)


class TestMetricsCalculations:
    """Tests for individual metric functions."""

    def test_winner_satisfaction(self):
        """Test that winner satisfaction sums the correct column of preferences."""
        prefs = jnp.array([[10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 0.0]])

        score_0 = _winner_satisfaction(prefs, winner=0)
        assert score_0 == 30.0

        score_1 = _winner_satisfaction(prefs, winner=1)
        assert score_1 == 5.0

    def test_vote_efficiency_perfect_match(self):
        """Test efficiency when votes perfectly align with preferences."""
        prefs = jnp.array([[10.0, 0.0]])
        votes = jnp.array([[10, 0]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 10.0

    def test_vote_efficiency_split_votes(self):
        """Test efficiency when an agent splits votes between candidates."""
        prefs = jnp.array([[10.0, 5.0]])
        votes = jnp.array([[5, 5]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 7.5

    def test_vote_efficiency_zero_votes(self):
        """Test that division by zero is handled (safe_tokens)."""
        prefs = jnp.array([[10.0, 5.0]])
        votes = jnp.array([[0, 0]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 0.0
        assert not jnp.isnan(eff)

    def test_compute_metrics_integration(self):
        """Test the wrapper function returns the expected dictionary."""
        prefs = jnp.array([[10.0, 0.0]])
        votes = jnp.array([[1, 0]])
        winner = 0

        metrics = compute_metrics(prefs, votes, winner)

        assert "winner_satisfaction" in metrics
        assert "vote_efficiency" in metrics
        assert metrics["winner_satisfaction"] == 10.0


class TestBatchMetrics:
    """Tests for batch_compute_metrics with Plurality and Quadratic data."""

    @pytest.fixture
    def mock_plurality_results(self):
        """Mock results for a Plurality Voting simulation."""
        return {
            0: {
                "final_winner": 0,
                "vote_round_1": jnp.array([0, 0]),
                "candidate_preferences": jnp.array([[1.0, 0.0], [1.0, 0.0]]),
                "softmax_probs_round_1": jnp.zeros((2, 2)),
            },
            1: {
                "final_winner": 1,
                "vote_round_1": jnp.array([1, 1]),  # Both voted 1
                "candidate_preferences": jnp.array([[0.0, 1.0], [0.0, 1.0]]),
                "softmax_probs_round_1": jnp.zeros((2, 2)),
            },
        }

    @pytest.fixture
    def mock_quadratic_results(self):
        """Mock results for a Quadratic Voting simulation (has qv_votes_matrix)."""
        return {
            0: {
                "final_winner": 0,
                "qv_votes_matrix": jnp.array([[10, 0], [10, 0]]),
                "candidate_preferences": jnp.array([[1.0, 0.0], [1.0, 0.0]]),
                "softmax_probs_round_1": jnp.zeros((2, 2)),
            }
        }

    def test_batch_metrics_plurality(self, mock_plurality_results):
        """Test processing of standard Plurality results (one-hot conversion)."""
        df = batch_compute_metrics(mock_plurality_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 Simulations
        assert "winner_satisfaction" in df.columns
        assert "vote_efficiency" in df.columns
        assert "simulation_id" in df.columns

        row_0 = df[df["simulation_id"] == 0].iloc[0]
        assert row_0["winner_satisfaction"] == 2.0
        assert row_0["vote_efficiency"] == 2.0

    def test_batch_metrics_quadratic(self, mock_quadratic_results):
        """Test processing of Quadratic results (uses qv_votes_matrix)."""
        df = batch_compute_metrics(mock_quadratic_results)

        assert len(df) == 1
        row = df.iloc[0]

        assert row["winner_satisfaction"] == 2.0
        assert row["vote_efficiency"] == 2.0

    def test_batch_metrics_alt_pref_key(self):
        """Test that it handles 'pref_candidate_gap'."""
        mock_data = {
            0: {
                "final_winner": 0,
                "vote_round_1": jnp.array([0]),
                "pref_candidate_gap": jnp.array([[5.0, 0.0]]),
                "softmax_probs_round_1": jnp.zeros((1, 2)),
            }
        }

        df = batch_compute_metrics(mock_data)
        assert df.iloc[0]["winner_satisfaction"] == 5.0
