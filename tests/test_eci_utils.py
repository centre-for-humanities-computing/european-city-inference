from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from eci.utils import (
    _extract_env_data_vectorized,
    _get_parameter_trajectory,
    generate_observations,
    get_voter_trajectory_data,
    kl_divergence,
)


class TestKLDivergence:
    """Tests for the kl_divergence function."""

    def test_kl_divergence_identical(self):
        """KL divergence between identical distributions should be 0."""
        mean = jnp.array([1.0, 2.0])
        prec = jnp.array([1.0, 1.0])

        # When P == Q, KL(P||Q) = 0
        kl = kl_divergence(mean, prec, mean, prec)
        assert jnp.allclose(kl, 0.0)

    def test_kl_divergence_known_values(self):
        """Test against manual calculation."""
        m_pref = jnp.array([0.0])
        p_pref = jnp.array([1.0])  # var = 1

        m_belief = jnp.array([1.0])
        p_belief = jnp.array([1.0])  # var = 1

        kl = kl_divergence(m_pref, p_pref, m_belief, p_belief)
        assert jnp.isclose(kl, 0.5)

    def test_kl_divergence_broadcasting(self):
        """Ensure the function handles array broadcasting correctly."""
        m_pref = jnp.array([[0.0], [10.0]])  # Shape (2, 1)
        p_pref = jnp.array([[1.0], [1.0]])

        m_belief = jnp.array([0.0])  # Shape (1,)
        p_belief = jnp.array([1.0])

        kl = kl_divergence(m_pref, p_pref, m_belief, p_belief)

        assert kl.shape == (2, 1)
        assert jnp.isclose(kl[0], 0.0)
        assert kl[1] > 0.0


class TestDataGeneration:
    """Tests for generate_observations and trajectory logic."""

    def test_get_parameter_trajectory_phase(self):
        """Test 'phase' shock pattern (A -> B -> A)."""
        n_steps = 10
        s_time = 2
        r_time = 5
        params = ((1.0, 1.0), (10.0, 10.0))  # (Normal, Shock)

        alpha, beta = _get_parameter_trajectory(
            n_steps, s_time, r_time, "phase", "linear", params
        )

        # Before shock
        assert alpha[0] == 1.0
        # During shock (indices 2, 3, 4)
        assert np.all(alpha[2:5] == 10.0)
        # After recovery (index 5+)
        assert np.all(alpha[5:] == 1.0)

    def test_get_parameter_trajectory_sudden(self):
        """Test 'sudden' shock pattern (A -> B...)."""
        n_steps = 10
        s_time = 5
        r_time = 8
        params = ((1.0, 1.0), (10.0, 10.0))

        alpha, beta = _get_parameter_trajectory(
            n_steps, s_time, r_time, "sudden", "linear", params
        )

        # Before shock
        assert np.all(alpha[:5] == 1.0)
        # After shock (persists until end)
        assert np.all(alpha[5:] == 10.0)

    def test_generate_observations_shape_and_bounds(self):
        """Test that output shape is correct and values are within [0,1]."""
        n_nodes = 3
        n_steps = 20

        obs = generate_observations(n_nodes, n_steps, seed=42)

        assert obs.shape == (n_steps, n_nodes)
        assert np.min(obs) >= 0.0
        assert np.max(obs) <= 1.0

    def test_generate_observations_reproducibility(self):
        """Test that fixing the seed produces identical results."""
        obs1 = generate_observations(2, 10, seed=123)
        obs2 = generate_observations(2, 10, seed=123)
        assert np.array_equal(obs1, obs2)

    def test_generate_observations_validation(self):
        """Test error raising for invalid inputs."""
        with pytest.raises(ValueError, match="Scenario must be 1 or 2"):
            generate_observations(1, 10, scenario=99)

        with pytest.raises(ValueError, match="Invalid shock_pattern"):
            generate_observations(1, 10, scenario=2, shock_pattern="invalid_pattern")


class TestDataExtraction:
    """Tests for environment data extraction functions."""

    def test_get_voter_trajectory_data(self):
        """Verify data retrieval for a specific voter."""
        # Mock Environment and Voter
        env = MagicMock()
        voter = MagicMock()
        voter.id = 1
        voter.trajectory = [{"expected_mean": [0.5], "precision": [1.0]}]
        voter.preferences = {"mean": [0.9], "precision": [2.0]}

        env.voters = [voter]
        # Mock input_data (shape: n_steps, n_preferences)
        env.input_data = np.zeros((10, 5))

        data = get_voter_trajectory_data(env, voter_id=1, pref_idx=0)

        assert "expected_mean" in data
        assert "observations" in data
        assert data["title_suffix"] == "for Voter 1"
        assert data["preference_params"] == (0.9, 2.0)

    def test_extract_env_data_vectorized(self):
        """Test extraction of complex nested structures into JAX arrays."""
        # --- Setup Mock Environment ---
        env = MagicMock()

        # 1. Preferences Indices (e.g., 2 dimensions)
        env.preferences_idx = [0, 1]

        # 2. Candidates (2 candidates)
        c1 = MagicMock()
        c1.policy = {"mean": np.array([0.1, 0.2]), "precision": np.array([1, 1])}
        c2 = MagicMock()
        c2.policy = {"mean": np.array([0.8, 0.9]), "precision": np.array([1, 1])}
        env.candidates = [c1, c2]

        node_0 = {
            "expected_mean": jnp.array([0.5, 0.5]),
            "expected_precision": jnp.array([1.0, 1.0]),
        }
        node_1 = {
            "expected_mean": jnp.array([0.6, 0.6]),
            "expected_precision": jnp.array([1.0, 1.0]),
        }

        agent_prefs = {
            "preferences": {
                "mean": jnp.array([[0.0, 0.0], [1.0, 1.0]]),
                "precision": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            }
        }

        def get_item(index):
            if index == 0:
                return node_0
            if index == 1:
                return node_1
            if index == -1:
                return agent_prefs
            raise IndexError("Unexpected index access")

        env.last_attributes.__getitem__.side_effect = get_item

        data = _extract_env_data_vectorized(env)

        assert data["candidates"]["mean"].shape == (2, 2)
        assert data["candidates"]["mean"][1, 0] == 0.8

        assert data["beliefs"]["mean"].shape == (2, 2)

        assert data["preferences"]["mean"].shape == (2, 2)
