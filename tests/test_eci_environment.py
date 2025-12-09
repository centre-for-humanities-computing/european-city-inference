from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from src.eci.environment import Environment


@pytest.fixture
def mock_dependencies():
    """Patches external dependencies to isolate Environment logic."""
    with (
        patch("src.eci.environment.Network") as mock_net,
        patch("src.eci.environment.generate_observations") as mock_gen,
        patch("src.eci.environment.Voter") as mock_voter_cls,
        patch("src.eci.environment.Candidate") as mock_cand_cls,
        patch("src.eci.environment.vmap") as mock_vmap,
    ):
        # Setup basic mock returns
        mock_gen.return_value = jnp.zeros((10, 2))  # Dummy observations

        def dummy_vmapped_func(*args, **kwargs):
            # Return shape matching (last_attributes, node_trajectories)
            return ({"attr": "dummy"}, [{"traj": "dummy"}])

        mock_vmap.return_value = dummy_vmapped_func

        yield {
            "Network": mock_net,
            "generate_observations": mock_gen,
            "Voter": mock_voter_cls,
            "Candidate": mock_cand_cls,
            "vmap": mock_vmap,
        }


@pytest.fixture
def env(mock_dependencies):
    """Create an Environment instance with mocked dependencies."""
    return Environment(num_voters=5, num_candidates=3, num_preferences=2)


# --- Initialization Tests ---


def test_environment_initialization(mock_dependencies):
    """Test that __init__ sets up the environment correctly."""
    num_voters = 5
    num_candidates = 3
    num_preferences = 2

    env = Environment(num_voters, num_candidates, num_preferences)

    # Verify Network creation
    mock_dependencies["Network"].assert_called()
    assert env.network == mock_dependencies["Network"].return_value

    # Verify Agent Creation
    assert len(env.voters) == num_voters
    assert len(env.candidates) == num_candidates
    assert len(env.agents) == num_voters + num_candidates

    # Verify generate_observations called
    mock_dependencies["generate_observations"].assert_called_with(
        n_nodes=num_preferences, n_steps=100, scenario=1
    )


def test_gather_agent_data(env):
    """Test aggregation of voter attributes into arrays."""
    # Setup mock voters with specific data
    v1 = MagicMock()
    v1.preferences = {"mean": jnp.array([1.0]), "precision": jnp.array([2.0])}
    v1.tonic_volatility = 0.5

    v2 = MagicMock()
    v2.preferences = {"mean": jnp.array([0.0]), "precision": jnp.array([1.0])}
    v2.tonic_volatility = -0.5

    env.voters = [v1, v2]

    mus, pis, vols = env._gather_agent_data()

    assert mus.shape == (2, 1)  # 2 voters, 1 preference
    assert pis.shape == (2, 1)
    assert vols.shape == (2,)
    assert mus[0] == 1.0


def test_initialize_network(env, mock_dependencies):
    """Test that initialize_network calls vmap and sets up trajectories."""
    # Setup mock gather data
    env._gather_agent_data = MagicMock(
        return_value=(jnp.zeros((5, 2)), jnp.zeros((5, 2)), jnp.zeros(5))
    )

    # Mock network internals required for preferences_idx logic
    edge_mock = MagicMock()
    edge_mock.value_parents = [0]
    env.network.edges = {0: edge_mock}
    env.network.input_idxs = [0]

    env.initialize_network()

    # Verify vmap was called
    mock_dependencies["vmap"].assert_called()

    # Verify attributes were stored
    assert env.last_attributes is not None
    assert env.node_trajectories is not None


# --- Simulation Execution Tests ---


def test_run_n_simulation(env):
    """Test running multiple simulations and aggregating results."""

    # Mock a simulation function
    def mock_sim_func(e, k):
        return {
            "vote_round_1": jnp.zeros(len(e.voters)),
            "val": float(jax.random.randint(k, (), 0, 10)),  # unique per run
        }

    key = jax.random.PRNGKey(0)
    n_sims = 3

    results = env.run_n_simulation(mock_sim_func, key, n_sims)

    assert len(results) == n_sims
    assert 0 in results
    assert 1 in results
    assert 2 in results
    assert env.sim_result == results


def test_update_agents(env):
    """Test that simulation results are correctly appended to agents' history."""
    # 1. Setup Environment with 2 voters
    v1 = MagicMock(
        vote_round_1=[],
        vote_round_2=[],
        softmax_probs_1=[],
        softmax_probs_2=[],
        dissatisfactions=[],
    )
    v2 = MagicMock(
        vote_round_1=[],
        vote_round_2=[],
        softmax_probs_1=[],
        softmax_probs_2=[],
        dissatisfactions=[],
    )
    env.voters = [v1, v2]

    # 2. Mock Simulation Result (1 simulation run)
    # The keys must match what _update_agents expects
    env.sim_result = {
        0: {
            "vote_round_1": [10, 20],
            "vote_final_round_2": [11, 21],
            "softmax_probs_round_1": [[0.1], [0.2]],
            "softmax_probs_final_round_2": [[0.3], [0.4]],
            "dissatisfaction": [5.0, 6.0],
        }
    }

    # 3. Run Update
    env._update_agents()

    # 4. Verify V1 (Index 0)
    assert v1.vote_round_1 == [10]
    assert v1.vote_round_2 == [11]
    assert v1.dissatisfactions == [5.0]

    # 5. Verify V2 (Index 1)
    assert v2.vote_round_1 == [20]
    assert v2.vote_round_2 == [21]


# --- Analysis & Output Tests ---


def test_get_winners_simple(env):
    """Test winner determination logic."""
    # Votes: A, A, B, B, B (B wins)
    votes = ["A", "A", "B", "B", "B"]

    winners = env.get_winners(votes, top_n=1)
    assert winners == ["B"]

    winners_top2 = env.get_winners(votes, top_n=2)
    assert winners_top2 == ["B", "A"]


def test_get_winners_jax_array(env):
    """Test winner determination with JAX/Numpy arrays."""
    # Votes: 1, 1, 2 (1 wins)
    votes = [jnp.array(1), jnp.array(1), jnp.array(2)]

    winners = env.get_winners(votes, top_n=1)
    assert winners == [1]


def test_create_data_frame(env):
    """Test pandas DataFrame creation from agent history."""
    # Setup manually
    env.num_simulations = 2

    # Voter 1 History
    v1 = MagicMock()
    v1.vote_round_1 = [1, 2]  # Sim 0, Sim 1
    v1.vote_round_2 = [1, 1]

    # Voter 2 History
    v2 = MagicMock()
    v2.vote_round_1 = [1, 2]
    v2.vote_round_2 = [1, 2]

    env.voters = [v1, v2]

    df = env.create_data_frame()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 2 simulations
    assert "simulation_id" in df.columns
    assert "winners_round_1" in df.columns
    assert "winner_round_2" in df.columns

    # Check Simulation 0 logic
    # R1 Votes: [1, 1] -> Winner 1
    # R2 Votes: [1, 1] -> Winner 1
    row_0 = df.iloc[0]
    assert row_0["winner_round_2"] == 1
