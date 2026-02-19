import jax.numpy as jnp
import pytest

from eci.voting_system.beliefs import _get_pref_belief_gap, _get_pref_candidate_gap


@pytest.fixture
def mock_voting_data():
    """Create a mock data dictionary."""
    n_agents = 2
    n_candidates = 2
    n_dim = 3

    return {
        "beliefs": {
            "mean": jnp.zeros((n_agents, n_dim)),
            "precision": jnp.ones((n_agents, n_dim)),
        },
        "preferences": {
            "mean": jnp.zeros((n_agents, n_dim)),
            "precision": jnp.ones((n_agents, n_dim)),
        },
        "candidates": {
            "mean": jnp.zeros((n_candidates, n_dim)),
            "precision": jnp.ones((n_candidates, n_dim)),
        },
    }


def test_get_pref_belief_gap_identical(mock_voting_data):
    """Test that KL divergence is zero when beliefs and preferences are identical."""
    gaps = _get_pref_belief_gap(mock_voting_data)

    assert gaps.shape == (2,)  # One gap per agent
    assert jnp.allclose(gaps, 0.0)  # KL divergence should be 0


def test_get_pref_belief_gap_value_check():
    """Test with specific values to ensure summation works across dimensions."""
    data = {
        "beliefs": {
            "mean": jnp.array([[1.0, 1.0]]),  # Agent 1
            "precision": jnp.array([[1.0, 1.0]]),
        },
        "preferences": {
            "mean": jnp.array([[0.0, 0.0]]),
            "precision": jnp.array([[1.0, 1.0]]),
        },
    }
    gaps = _get_pref_belief_gap(data)
    assert jnp.isclose(
        gaps[0], 1.0
    )  # KL divergence should be 0.5 per dimension, summed to 1.0


def test_get_pref_candidate_gap_broadcasting(mock_voting_data):
    """Verify that every agent is compared against every candidate."""
    gaps = _get_pref_candidate_gap(mock_voting_data)

    assert gaps.shape == (2, 2)


def test_get_pref_candidate_gap_values():
    """Verify the broadcasting."""
    data = {
        "preferences": {
            "mean": jnp.array([[0.0]]),
            "precision": jnp.array([[1.0]]),
        },
        "candidates": {
            "mean": jnp.array([[1.0], [2.0]]),
            "precision": jnp.array([[1.0], [1.0]]),
        },
    }
    gaps = _get_pref_candidate_gap(data)
    # KL(Preferences || Candidate 1) = 0.5, KL(Preferences || Candidate 2) = 2.0
    assert jnp.allclose(gaps, jnp.array([[0.5, 2.0]]))
