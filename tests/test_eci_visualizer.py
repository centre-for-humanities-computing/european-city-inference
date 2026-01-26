import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eci.plots import SimulationVisualizer


class TestSimulationVisualizer(unittest.TestCase):
    """Testing class for SimulationVisualizer methods.

    Tests plotting methods by checking if they return a matplotlib Figure
    and safely closing the figure to prevent resource leaks.
    """

    def setUp(self):
        """Set up the SimulationVisualizer instance."""
        self.viz = SimulationVisualizer()

    def test_plot_belief_trajectory(self):
        """Test plot_belief_trajectory.

        Checks if the method returns a matplotlib Figure instance.
        """
        n_steps = 50
        means = np.linspace(0, 1, n_steps)
        precisions = np.ones(n_steps)
        observations = means + np.random.normal(0, 0.1, n_steps)
        preference_params = (0.5, 1.0)

        fig, ax_main, ax_density = self.viz.plot_belief_trajectory(
            means=means,
            precisions=precisions,
            observations=observations,
            preference_params=preference_params,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_preference_distributions(self):
        """Test plot_preference_distributions.

        Checks if the method returns a matplotlib Figure instance.
        """
        data = pd.DataFrame(
            {
                "preference": ["A"] * 10 + ["B"] * 10,
                "id": ["C1"] * 5 + ["Voter1"] * 5 + ["C2"] * 5 + ["Voter2"] * 5,
                "x": np.random.rand(20),
                "pdf": np.random.rand(20),
                "group": ["Candidate"] * 5
                + ["Voter"] * 5
                + ["Candidate"] * 5
                + ["Voter"] * 5,
            }
        )

        fig, axes = self.viz.plot_preference_distributions(data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_vote_proportions_histogram(self):
        """Test plot_vote_proportions with 'histogram' plot_kind.

        Checks if the method returns a matplotlib Figure instance.
        """
        vote_counts = []
        for r in range(1, 4):
            for c in ["C1", "C2"]:
                for _ in range(10):
                    vote_counts.append(
                        {"round": r, "candidate_id": c, "proportion": np.random.rand()}
                    )

        fig, axes = self.viz.plot_vote_proportions(vote_counts, plot_kind="histogram")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_vote_proportions_stripplot(self):
        """Test plot_vote_proportions with 'stripplot' plot_kind.

        Checks if the method returns a matplotlib Figure instance.
        """
        vote_counts = []
        for r in range(1, 3):
            for c in ["C1", "C2"]:
                for _ in range(5):
                    vote_counts.append(
                        {"round": r, "candidate_id": c, "proportion": np.random.rand()}
                    )

        fig, axes = self.viz.plot_vote_proportions(vote_counts, plot_kind="stripplot")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_inputs(self):
        """Test plotting methods with empty inputs.

        Checks if methods return None when given empty data structures.
        """
        self.assertIsNone(self.viz.plot_preference_distributions(pd.DataFrame()))
        self.assertIsNone(self.viz.plot_vote_proportions([]))
