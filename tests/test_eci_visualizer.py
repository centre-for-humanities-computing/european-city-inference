import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eci.visualizer import SimulationVisualizer


class TestSimulationVisualizer:
    """Test Simulation Visual."""

    @pytest.fixture
    def visualizer(self):
        """Fixture to provide a fresh instance of the visualizer."""
        return SimulationVisualizer(style="whitegrid")

    @pytest.fixture
    def sample_preference_data(self):
        """Generate dummy data for preference distribution plots."""
        # Create data for 2 preferences, 2 candidates
        data = {
            "preference": ["Pref A"] * 10 + ["Pref B"] * 10,
            "id": (["C1"] * 5 + ["Voter"] * 5) * 2,
            "group": (["Candidate"] * 5 + ["Voter"] * 5) * 2,
            "x": np.linspace(0, 1, 5).tolist() * 4,
            "pdf": np.random.rand(20).tolist(),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_vote_data(self):
        """Generate dummy data for vote proportion plots."""
        return [
            {"round": 1, "candidate_id": "C1", "proportion": 0.4},
            {"round": 1, "candidate_id": "C2", "proportion": 0.6},
            {"round": 2, "candidate_id": "C1", "proportion": 0.45},
            {"round": 2, "candidate_id": "C2", "proportion": 0.55},
        ]

    def test_init(self, visualizer):
        """Test that the visualizer initializes correctly."""
        assert visualizer.style == "whitegrid"

    # --- Tests for plot_preference_distributions ---

    def test_plot_preference_distributions_empty(self, visualizer):
        """Test that empty data returns None."""
        empty_df = pd.DataFrame()
        fig, ax = visualizer.plot_preference_distributions(empty_df)
        assert fig is None
        assert ax is None

    def test_plot_preference_distributions_valid(
        self, visualizer, sample_preference_data
    ):
        """Test standard plotting functionality."""
        fig, axes = visualizer.plot_preference_distributions(sample_preference_data)

        assert isinstance(fig, Figure)
        # We have 2 unique preferences in the fixture, so we expect an array of size 2
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 2

        # Cleanup
        plt.close(fig)

    def test_plot_preference_distributions_custom_axes(
        self, visualizer, sample_preference_data
    ):
        """Test that we can pass existing axes."""
        # Create external axes
        fig_ex, axes_ex = plt.subplots(2, 1)

        result_fig, result_axes = visualizer.plot_preference_distributions(
            sample_preference_data, axes=axes_ex
        )

        assert result_fig is fig_ex
        assert result_axes is axes_ex
        plt.close(fig_ex)

    # --- Tests for plot_belief_trajectory ---

    def test_plot_belief_trajectory(self, visualizer):
        """Test the trajectory plot creation."""
        # Setup dummy numpy data
        steps = 10
        means = np.linspace(0, 1, steps)
        precisions = np.ones(steps) * 10
        observations = np.random.rand(steps)
        pref_params = (0.5, 10.0)  # mean, precision

        fig, ax_main, ax_density = visualizer.plot_belief_trajectory(
            means, precisions, observations, pref_params
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax_main, Axes)
        assert isinstance(ax_density, Axes)

        # Check if title suffix is applied (default is empty, so check default title)
        assert "Belief Trajectory" in ax_main.get_title()

        plt.close(fig)

    def test_plot_belief_trajectory_with_ylim(self, visualizer):
        """Test that ylim is applied correctly."""
        steps = 5
        means = np.zeros(steps)
        precisions = np.ones(steps)
        observations = np.zeros(steps)
        pref_params = (0, 1)

        custom_lim = (-5, 5)

        fig, ax_main, _ = visualizer.plot_belief_trajectory(
            means, precisions, observations, pref_params, ylim=custom_lim
        )

        # Check if the limits were actually applied
        bottom, top = ax_main.get_ylim()
        assert bottom == custom_lim[0]
        assert top == custom_lim[1]

        plt.close(fig)

    # --- Tests for plot_vote_proportions ---

    def test_plot_vote_proportions_empty(self, visualizer):
        """Test that empty list returns None."""
        assert visualizer.plot_vote_proportions([]) is None

    def test_plot_vote_proportions_histogram(self, visualizer, sample_vote_data):
        """Test histogram mode."""
        fig, axes = visualizer.plot_vote_proportions(
            sample_vote_data, plot_kind="histogram"
        )

        assert isinstance(fig, Figure)
        # There are 2 rounds in sample data, so we expect 2 subplots
        assert len(axes) == 2

        # Check title of first subplot
        assert "Round 1" in axes[0].get_title()

        plt.close(fig)

    def test_plot_vote_proportions_stripplot(self, visualizer, sample_vote_data):
        """Test stripplot mode."""
        fig, axes = visualizer.plot_vote_proportions(
            sample_vote_data, plot_kind="stripplot"
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_vote_proportions_integration(self, visualizer, sample_vote_data):
        """Check if x-axis formatting is applied (PercentFormatter)."""
        fig, axes = visualizer.plot_vote_proportions(sample_vote_data)

        # It's hard to check the exact formatter type instance,
        # but we can check if xlim was constrained to (0, 1.05) as per code
        left, right = axes[0].get_xlim()
        assert left == 0
        assert right == 1.05

        plt.close(fig)
