import jax
import jax.numpy as jnp
import pytest

from eci.decision import _compute_candidate_utilities, _sample_choice


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


def test_compute_candidate_utilities_logic():
    """Verifies the normalized score (belief_gap - cand_gap) / belief_gap.

    Uses non-degenerate beliefs so belief_gap > 0 and the normalization
    is well-defined (the eps guard only kicks in when belief_gap == 0).
    """
    # Beliefs deliberately offset from preferences so KL(belief‖pref) > 0.
    data = {
        "beliefs": {
            "mean": jnp.array([[1.0, 1.0]]),
            "precision": jnp.array([[1.0, 1.0]]),
        },
        "preferences": {
            "mean": jnp.array([[0.0, 0.0]]),
            "precision": jnp.array([[1.0, 1.0]]),
        },
        "candidates": {
            "mean": jnp.array([[0.0, 0.0], [2.0, 2.0]]),
            "precision": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        },
    }

    pref_scores, cand_gap, belief_gap = _compute_candidate_utilities(data)

    assert pref_scores.shape == (1, 2)
    assert cand_gap.shape == (1, 2)
    assert belief_gap.shape == (1,)

    # Normalized score = (belief_gap - cand_gap) / belief_gap
    expected_0 = (belief_gap[0] - cand_gap[0, 0]) / belief_gap[0]
    expected_1 = (belief_gap[0] - cand_gap[0, 1]) / belief_gap[0]
    assert jnp.isclose(pref_scores[0, 0], expected_0, atol=1e-5)
    assert jnp.isclose(pref_scores[0, 1], expected_1, atol=1e-5)


def test_compute_candidate_utilities_broadcast():
    """Score normalization broadcasts correctly across agents and candidates."""
    data = {
        "beliefs": {
            "mean": jnp.array([[1.0, 1.0], [2.0, 2.0]]),
            "precision": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        },
        "preferences": {
            "mean": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
            "precision": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        },
        "candidates": {
            "mean": jnp.array([[0.5, 0.5], [1.5, 1.5]]),
            "precision": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        },
    }

    pref_scores, cand_gap, belief_gap = _compute_candidate_utilities(data)

    for i in range(pref_scores.shape[0]):
        for j in range(pref_scores.shape[1]):
            expected = (belief_gap[i] - cand_gap[i, j]) / belief_gap[i]
            assert jnp.isclose(pref_scores[i, j], expected, atol=1e-5)
