from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eci.plots import (
    plot_belief_trajectory,
    plot_preference,
    plot_vote_shares,
    plot_voting_metrics,
)


class TestPlots:
    """Test suite for visualization functions."""

    @pytest.fixture
    def mock_env_data(self):
        """Create mock data simulating the environment extraction."""
        return {
            "preferences": {
                "mean": np.array([[0.5, 0.5], [0.1, 0.1]]),
                "precision": np.array([[10.0, 10.0], [5.0, 5.0]]),
            },
            "candidates": {
                "mean": np.array([[0.8, 0.8]]),  # 1 Candidate
                "precision": np.array([[10.0, 10.0]]),
            },
        }

    @patch("eci.plots._extract_env_data_vectorized")
    def test_plot_preference_runs(self, mock_extract, mock_env_data):
        """Verifie that plot_preference generates a figure without error."""
        mock_extract.return_value = mock_env_data
        env = MagicMock()

        # Execution
        fig, axes = plot_preference(env)

        # Assertions
        assert isinstance(fig, plt.Figure)
        assert len(np.atleast_1d(axes)) > 0

        # Check the title of the first subplot
        ax_list = np.atleast_1d(axes)
        assert "Dimension 0" in ax_list[0].get_title()

    def test_plot_vote_shares(self):
        """Verifie that plot_vote_shares handles the DataFrame correctly."""
        # Mock Data
        df = pd.DataFrame(
            {
                "candidate": ["A", "B", "A", "B"],
                "share": [0.6, 0.4, 0.55, 0.45],
                "round": ["Round 1", "Round 1", "Round 2", "Round 2"],
            }
        )

        # Test without existing axes (creates new figure)
        fig, ax = plot_vote_shares(df)
        assert isinstance(fig, plt.Figure)
        assert ax.get_ylabel() == "Vote Share"

        # Test with existing axes (reuses it)
        _, ax2 = plt.subplots()
        _, ax_res = plot_vote_shares(df, ax=ax2)
        assert ax_res is ax2  # Must be the same object

    def test_plot_belief_trajectory(self):
        """Verifie the plotting of belief trajectory and density."""
        # Mock Data
        steps = 10
        obs = np.random.rand(steps)
        mean = np.linspace(0, 1, steps)
        prec = np.ones(steps)
        params = (0.5, 10.0)  # Mean, Precision

        # Execution
        fig, ax_main, ax_density = plot_belief_trajectory(
            expected_mean=mean,
            precisions=prec,
            observations=obs,
            preference_params=params,
            title_suffix="Test",
        )

        # Assertions
        assert isinstance(fig, plt.Figure)
        assert "Belief Trajectory Test" in ax_main.get_title()

        assert len(ax_main.collections) > 0
        assert len(ax_main.lines) > 0

    def test_plot_voting_metrics(self):
        """Verifies that plot_voting_metrics generates the two subplots."""
        df = pd.DataFrame(
            {
                "vote_efficiency": [0.9, 0.8],
                "winner_satisfaction": [0.5, 0.6],
                "voting_system": ["Plurality", "Quadratic"],
            }
        )

        # Execution
        fig, ax_array = plot_voting_metrics(df)

        assert isinstance(fig, plt.Figure)
        assert len(ax_array) == 2  # Must have 2 subplots

        # Check titles to ensure correct plotting logic
        assert "reflect preferences" in ax_array[0].get_title()
        assert "satisfy the group" in ax_array[1].get_title()

    def test_plot_metrics_empty_data(self):
        """Ensure function handles empty DataFrame without crashing."""
        df = pd.DataFrame(
            columns=["vote_efficiency", "winner_satisfaction", "voting_system"]
        )

        try:
            plot_voting_metrics(df)
        except Exception as e:
            pytest.fail(f"Plot function crashed with empty data: {e}")
