import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import DataCollector, SimulationVisualizer


# --- MOCK CLASSES ---
class MockScheduler:
    """Mock scheduler class for testing."""

    def __init__(self):
        """Initialize the mock scheduler with a default step count."""
        self.step_count = 1


class MockVoter:
    """Mock voter class for testing."""

    def __init__(self, vid):
        """Initialize the mock voter with ID, vote, and preferences."""
        self.id = vid
        self.last_vote = vid  # just use the ID
        self.preferences = {"mean": np.array([0.0]), "precision": np.array([1.0])}


class MockCandidate:
    """Mock candidate class for testing."""

    def __init__(self, cid):
        """Initialize the mock candidate with ID and policy."""
        self.id = cid
        self.policy = {"mean": np.array([0.0]), "precision": np.array([1.0])}


class MockEnvironment:
    """Mock environment class for DataCollector and SimulationVisualizer."""

    def __init__(self):
        """Initialize the mock environment."""
        self.scheduler = MockScheduler()
        self.winner_id = 0
        self.voting_system = type("VotingSystem", (), {"name": "Plurality"})()
        self.last_round1_results = {0: 5.0, 1: 3.0}
        self.last_round2_results = {0: 2.0, 1: 6.0}
        self.voters = [MockVoter(0), MockVoter(1)]
        self.candidates = [MockCandidate(0), MockCandidate(1)]
        self.num_preferences = 1
        self.input_data = np.random.randn(5, 1)


# --- TESTS ---
def test_process_round_results():
    """Test processing of round results."""
    dc = DataCollector()
    results = {0: 10, 1: 20}
    out = dc._process_round_results(results, 1)
    assert "candidate_0_score_r1" in out
    assert "candidate_1_prop_r1" in out
    assert out["candidate_1_prop_r1"] == 20 / 30


def test_collect_and_dataframe():
    """Test that collected environment data can be returned as a DataFrame."""
    env = MockEnvironment()
    dc = DataCollector()
    dc.collect(env)
    df = dc.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "winner_id" in df.columns
    assert "voter_0_vote" in df.columns


def test_long_dataframe():
    """Test that collected data can be reshaped into long-format DataFrame."""
    env = MockEnvironment()
    dc = DataCollector()
    dc.collect(env)
    long_df = dc.get_long_dataframe()
    assert isinstance(long_df, pd.DataFrame)
    assert "candidate_id" in long_df.columns
    assert "round" in long_df.columns


def test_prepare_preference_df():
    """Test that preference DataFrame is prepared."""
    env = MockEnvironment()
    dc = DataCollector()
    vis = SimulationVisualizer(env, dc)
    df = vis._prepare_preference_df(num_voters_to_show=1)
    assert not df.empty
    assert "pdf" in df.columns
    assert set(df["group"].unique()) == {"Candidate", "Voter"}


def test_plot_preference_distributions():
    """Test plotting of preference distributions produces a valid figure."""
    env = MockEnvironment()
    dc = DataCollector()
    vis = SimulationVisualizer(env, dc)
    fig, axes = vis.plot_preference_distributions(num_voters_to_show=1)
    assert isinstance(fig, plt.Figure)
    assert len(axes) >= 1
    plt.close(fig)


def test_plot_simulation_results_distribution_hist():
    """Test histogram plotting of simulation results produces valid output."""
    env = MockEnvironment()
    dc = DataCollector()
    dc.collect(env)
    vis = SimulationVisualizer(env, dc)
    fig, axes = vis.plot_simulation_results_distribution(plot_kind="histogram")
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_simulation_results_distribution_stripplot():
    """Test stripplot plotting of simulation results produces valid output."""
    env = MockEnvironment()
    dc = DataCollector()
    dc.collect(env)
    vis = SimulationVisualizer(env, dc)
    fig, axes = vis.plot_simulation_results_distribution(plot_kind="stripplot")
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    plt.close(fig)
