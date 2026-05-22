"""Tests for metrics.

Notes on the refactor:
- Voting functions now return dicts with keys `winner`, `votes`, `softmax`,
  `candidate_utilities`, `qv_votes_matrix` (was: `final_winner`,
  `vote_round_1`, `softmax_probs_round_1`, `candidate_preferences`).
- `batch_compute_metrics` reads `winner`, `softmax`, `votes` /
  `qv_votes_matrix`, and falls back from `pref_candidate_gap` to
  `candidate_utilities` for the per-candidate preference array.
"""

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
        """Winner satisfaction sums the correct column of preferences."""
        prefs = jnp.array([[10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 0.0]])

        assert _winner_satisfaction(prefs, winner=0) == 30.0
        assert _winner_satisfaction(prefs, winner=1) == 5.0

    def test_vote_efficiency_perfect_match(self):
        """Efficiency when votes perfectly align with preferences."""
        prefs = jnp.array([[10.0, 0.0]])
        votes = jnp.array([[10, 0]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 10.0

    def test_vote_efficiency_split_votes(self):
        """Efficiency when an agent splits votes between candidates."""
        prefs = jnp.array([[10.0, 5.0]])
        votes = jnp.array([[5, 5]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 7.5

    def test_vote_efficiency_zero_votes(self):
        """Division by zero is handled via safe_tokens."""
        prefs = jnp.array([[10.0, 5.0]])
        votes = jnp.array([[0, 0]])

        eff = _vote_efficiency(prefs, votes)
        assert eff == 0.0
        assert not jnp.isnan(eff)

    def test_compute_metrics_integration(self):
        """The wrapper returns the expected dictionary."""
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
        """Mock results matching the new _vote_plurality return shape."""
        return {
            0: {
                "winner": 0,
                "votes": jnp.array([0, 0]),  # Both voted for candidate 0
                "candidate_utilities": jnp.array([[1.0, 0.0], [1.0, 0.0]]),
                "softmax": jnp.zeros((2, 2)),
            },
            1: {
                "winner": 1,
                "votes": jnp.array([1, 1]),  # Both voted for candidate 1
                "candidate_utilities": jnp.array([[0.0, 1.0], [0.0, 1.0]]),
                "softmax": jnp.zeros((2, 2)),
            },
        }

    @pytest.fixture
    def mock_quadratic_results(self):
        """Mock results matching the new _vote_quadratic return shape."""
        return {
            0: {
                "winner": 0,
                "qv_votes_matrix": jnp.array([[10, 0], [10, 0]]),
                "candidate_utilities": jnp.array([[1.0, 0.0], [1.0, 0.0]]),
                "softmax": jnp.zeros((2, 2)),
            }
        }

    def test_batch_metrics_plurality(self, mock_plurality_results):
        """Processing of Plurality results (one-hot conversion from `votes`)."""
        df = batch_compute_metrics(mock_plurality_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 simulations
        assert "winner_satisfaction" in df.columns
        assert "vote_efficiency" in df.columns
        assert "simulation_id" in df.columns

        row_0 = df[df["simulation_id"] == 0].iloc[0]
        assert row_0["winner_satisfaction"] == 2.0
        assert row_0["vote_efficiency"] == 2.0

    def test_batch_metrics_quadratic(self, mock_quadratic_results):
        """Processing of Quadratic results (uses qv_votes_matrix)."""
        df = batch_compute_metrics(mock_quadratic_results)

        assert len(df) == 1
        row = df.iloc[0]

        assert row["winner_satisfaction"] == 2.0
        assert row["vote_efficiency"] == 2.0

    def test_batch_metrics_pref_candidate_gap_key(self):
        """batch_compute_metrics prefers `pref_candidate_gap` when present."""
        mock_data = {
            0: {
                "winner": 0,
                "votes": jnp.array([0]),
                "pref_candidate_gap": jnp.array([[5.0, 0.0]]),
                "softmax": jnp.zeros((1, 2)),
            }
        }

        df = batch_compute_metrics(mock_data)
        assert df.iloc[0]["winner_satisfaction"] == 5.0
