from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from eci.agents import Candidate, Voter
from eci.environment import EnvConfig, Environment


class TestEnvironment:
    """Tests for the Environment class, mocking external dependencies like PyHGF."""

    @pytest.fixture
    def mock_config(self):
        """Configure for testing."""
        return EnvConfig(
            num_voters=5,
            num_candidates=2,
            num_preferences=3,
            num_steps=10,
            scenario=2,
            seed=42,
            tonic_volatility_mean=-2.0,
            tonic_volatility_std=0.01,
        )

    @patch("eci.world.generate_observations")
    @patch("eci.perceptual.Network")
    def test_initialization(self, mock_network_cls, mock_gen_obs, mock_config):
        """Test that __init__ correctly."""
        # Setup Mocks
        # mock inputs: shape (n_steps, n_preferences) -> (10, 3)
        mock_gen_obs.return_value = jnp.zeros((10, 3))

        # Initialize Environment
        env = Environment(mock_config)

        # Data Generation — assert all the kwargs we now pass through.
        mock_gen_obs.assert_called_once_with(
            n_nodes=3,
            n_steps=10,
            scenario=2,
            shock_pattern=None,
            dispersion=1.0,
            obs_low=0.0,
            obs_high=1.0,
            recover=False,
            seed=None,
        )
        assert env.input_data.shape == (10, 3)

        # Network Setup
        mock_network_cls.assert_called()
        # Verify nodes were added (1 call for state nodes + 3 calls for children = 4)
        assert env.network.add_nodes.call_count >= 4

        # Agent Initialization
        assert len(env.voters) == 5
        assert len(env.candidates) == 2
        assert len(env.agents) == 7  # 5 + 2

        # Check types
        assert isinstance(env.voters[0], Voter)
        assert isinstance(env.candidates[0], Candidate)

        # Check IDs are unique and sequential
        ids = [a.id for a in env.agents]
        assert ids == list(range(7))

    @patch("eci.world.generate_observations")
    @patch("eci.perceptual.Network")
    def test_gather_agent_data(self, mock_net, mock_gen, mock_config):
        """Test that agent parameters are correctly gathered into JAX arrays."""
        mock_gen.return_value = jnp.zeros((10, 3))
        env = Environment(mock_config)

        mus, pis, vols = env._gather_agent_data()

        # Means: (n_voters, n_preferences)
        assert mus.shape == (5, 3)
        # Precisions: (n_voters, n_preferences)
        assert pis.shape == (5, 3)
        # Volatilities: (n_voters,)
        assert vols.shape == (5,)

    @patch("eci.world.generate_observations")
    @patch("eci.perceptual.Network")
    def test_run_n_simulation_flow(self, mock_net, mock_gen, mock_config):
        """Test the simulation loop execution.

        Signature is now: run_n_simulation(func, data, response_function,
        key, n_simulations, ...) and func is called as
        func(data, response_function, subkey, *args, **kwargs).
        """
        mock_gen.return_value = jnp.zeros((10, 3))
        env = Environment(mock_config)

        mock_sim_func = MagicMock()
        mock_sim_func.return_value = {"winner": 1}
        mock_response_function = MagicMock()
        data = {"dummy": "data"}

        key = jax.random.PRNGKey(0)
        n_sims = 3

        results = env.run_n_simulation(
            mock_sim_func, data, mock_response_function, key, n_simulations=n_sims
        )

        # Verify results storage
        assert len(results) == 3
        assert results[0] == {"winner": 1}
        assert results[2] == {"winner": 1}

        # Verify call count
        assert mock_sim_func.call_count == 3

        # Verify positional args are (data, response_function, subkey, ...)
        call_args, _ = mock_sim_func.call_args_list[0]
        assert call_args[0] is data
        assert call_args[1] is mock_response_function

        # Verify env.sim_result is updated
        assert env.sim_result == results

    def test_run_single_agent_inference_logic(self, mock_config):
        """Test the logic inside the single agent inference step."""
        with (
            patch("eci.world.generate_observations") as mock_gen,
            patch("eci.perceptual.Network"),
        ):
            mock_gen.return_value = jnp.zeros((5, 2))
            env = Environment(mock_config)

            mock_net_instance = MagicMock()

            mock_attributes_list = MagicMock()
            mock_net_instance.attributes = mock_attributes_list

            mock_net_instance.input_idxs = [0, 1]

            mock_net_instance.last_attributes = "mock_last_attrs"
            mock_net_instance.node_trajectories = "mock_node_trajs"

            mu = jnp.array([0.5, 0.5])
            pi = jnp.array([1.0, 1.0])
            vol = 0.1

            last, traj = env._run_single_agent_inference(mu, pi, vol, mock_net_instance)

            mock_attributes_list.__getitem__.assert_any_call(-1)

            mock_attributes_list.__getitem__.assert_any_call(0)
            mock_attributes_list.__getitem__.assert_any_call(1)

            mock_net_instance.input_data.assert_called_with(input_data=env.input_data)

            assert last == "mock_last_attrs"
            assert traj == "mock_node_trajs"

    @patch("eci.perceptual.vmap")
    @patch("eci.world.generate_observations")
    @patch("eci.perceptual.Network")
    def test_run_multi_agent_inference_distribution(
        self, mock_net_cls, mock_gen, mock_vmap, mock_config
    ):
        """Test that results are correctlydistributed."""
        mock_gen.return_value = jnp.zeros((10, 3))
        mock_net_instance = MagicMock()
        mock_net_instance.input_idxs = [0, 1, 2]
        mock_net_cls.return_value = mock_net_instance

        env = Environment(mock_config)

        dummy_traj = {
            "mean": jnp.array(
                [10, 11, 12, 13, 14]
            ),  # Voter 0 gets 10, Voter 1 gets 11...
            "precision": jnp.array([20, 21, 22, 23, 24]),
        }

        mock_vmap.return_value.return_value = ("dummy_last_attrs", dummy_traj)

        env._run_multi_agent_inference()

        assert env.preferences_idx == [0, 1, 2]

        v0 = env.voters[0]
        assert v0.trajectory["mean"] == 10
        assert v0.trajectory["precision"] == 20

        v4 = env.voters[4]
        assert v4.trajectory["mean"] == 14
        assert v4.trajectory["precision"] == 24
