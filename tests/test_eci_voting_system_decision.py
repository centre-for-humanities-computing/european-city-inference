import jax
import jax.numpy as jnp
import pytest

from eci.voting_system.decisions import _compute_preferences, _sample_choice


@pytest.fixture
def mock_decision_data():
    """Prepare a dictionary of mock data for preference tests."""
    n_agents = 3
    n_candidates = 2
    n_dim = 2

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


def test_sample_choice_shapes():
    """Verifies that the output shapes of _sample_choice are correct."""
    key = jax.random.PRNGKey(42)
    # Preferences for 5 agents and 3 candidates
    preferences = jnp.array(
        [
            [1.0, 2.0, 0.5],
            [0.1, 0.8, 0.1],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, 0.0],
        ]
    )

    vote, softmax_probs = _sample_choice(key, preferences)

    assert vote.shape == (5,)
    assert softmax_probs.shape == (5, 3)
    assert jnp.allclose(jnp.sum(softmax_probs, axis=1), 1.0)


def test_sample_choice_determinism():
    """Verifies that a very high score consistently results in the same choice."""
    key = jax.random.PRNGKey(0)
    preferences = jnp.array([[100.0, 0.0, 0.0]])

    vote, softmax_probs = _sample_choice(key, preferences)

    assert vote[0] == 0
    assert softmax_probs[0, 0] > 0.99


def test_compute_preferences_logic(mock_decision_data):
    """Verifies that the score calculation (Belief Gap - Candidate Gap) is correct."""
    data = mock_decision_data

    pref_scores, cand_gap, belief_gap = _compute_preferences(data)

    assert pref_scores.shape == (3, 2)
    assert cand_gap.shape == (3, 2)
    assert belief_gap.shape == (3,)

    expected_score = belief_gap[0] - cand_gap[0, 0]
    assert jnp.isclose(pref_scores[0, 0], expected_score)


def test_compute_preferences_broadcast(mock_decision_data):
    """Verifies that the subtraction broadcasts correctly across all candidates."""
    pref_scores, cand_gap, belief_gap = _compute_preferences(mock_decision_data)

    for i in range(pref_scores.shape[0]):
        for j in range(pref_scores.shape[1]):
            assert jnp.isclose(pref_scores[i, j], belief_gap[i] - cand_gap[i, j])
