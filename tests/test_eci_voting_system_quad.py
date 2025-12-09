from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from eci.voting_system.quadratic import (
    _evaluate_candidate_scores,
    _get_current_beliefs_t,
    _get_current_dissatisfaction,
    _vote_quadratic,
)


@pytest.fixture
def mock_kl_divergence():
    """Mock the math function to avoid needing real stats logic."""
    with patch("src.eci.voting_system.quad.kl_divergence") as mock_kl:
        # Default behavior: return 0.0 distance (perfect match)
        # This simplifies 'Expected Dissatisfaction' logic in tests
        mock_kl.side_effect = lambda m1, p1, m2, p2: jnp.zeros_like(m1)
        yield mock_kl


@pytest.fixture
def mock_env():
    """Create a complex mock environment with the required nested structure."""
    env = MagicMock()

    # Configuration
    num_agents = 3
    num_prefs = 2
    num_candidates = 3

    env.voters = range(num_agents)
    env.preferences_idx = range(num_prefs)

    # Mock Candidates with specific IDs
    # We use IDs 10, 20, 30 to differentiate them from array indices 0, 1, 2
    candidates = []
    for i in range(num_candidates):
        c = MagicMock()
        c.id = (i + 1) * 10
        c.policy = {
            "mean": jnp.array([float(i), float(i)]),
            "precision": jnp.array([1.0, 1.0]),
        }
        candidates.append(c)
    env.candidates = candidates

    # Mock 'last_attributes' (The nested dict source of data)
    # Structure: env.last_attributes[pref_idx]["expected_mean"][agent_idx]
    last_attributes = {}

    # 1. Fill preference-specific attributes
    for p_idx in range(num_prefs):
        last_attributes[p_idx] = {
            "expected_mean": [0.5] * num_agents,
            "precision": [1.0] * num_agents,
        }

    # 2. Fill global agent preferences (Index -1)
    # Structure: env.last_attributes[-1]["preferences"]["mean"][agent][pref]
    last_attributes[-1] = {
        "preferences": {
            "mean": [[0.8] * num_prefs for _ in range(num_agents)],
            "precision": [[2.0] * num_prefs for _ in range(num_agents)],
        }
    }

    env.last_attributes = last_attributes
    return env


# --- Unit Tests for Helper Functions ---


def test_get_current_beliefs_t_structure(mock_env):
    """Test that data is correctly extracted from the complex nested env dict."""
    data = _get_current_beliefs_t(mock_env)

    assert len(data) == 3  # 3 agents
    # Check the 4 required keys exist per agent
    required_keys = [
        "means_belief",
        "precisions_belief",
        "agent_pref_means",
        "agent_pref_precisions",
    ]
    for k in required_keys:
        assert k in data[0]
        assert isinstance(data[0][k], (jnp.ndarray, np.ndarray))
        # Check shape: (2 preferences,)
        assert data[0][k].shape == (2,)


def test_get_current_dissatisfaction_calc(mock_env, mock_kl_divergence):
    """Test that dissatisfaction stacks arrays and sums correctly."""
    # Setup
    data = _get_current_beliefs_t(mock_env)

    # Force KL divergence to return 1.0 for every element
    # This means dissatisfaction per preference is 1.0
    mock_kl_divergence.side_effect = lambda m1, p1, m2, p2: jnp.ones_like(m1)

    dissat, means, precs = _get_current_dissatisfaction(mock_env, data)

    # Assertions
    # Shape: (num_agents,)
    assert dissat.shape == (3,)
    # Value: Sum over 2 preferences = 1.0 + 1.0 = 2.0
    assert jnp.allclose(dissat, 2.0)

    # Check stacked shapes (num_agents, num_prefs)
    assert means.shape == (3, 2)


def test_evaluate_candidate_scores_logic(mock_env, mock_kl_divergence):
    """Test calculation of Marginal Utility (Score)."""
    # Score = Current Dissatisfaction - Expected Dissatisfaction

    # 1. Setup inputs
    num_agents = 3
    num_cands = 3
    beliefs_mean = jnp.zeros((num_agents, 2))
    beliefs_prec = jnp.ones((num_agents, 2))

    # Assume current dissatisfaction is HIGH (10.0)
    current_dissat = jnp.full((num_agents,), 10.0)

    # 2. Mock KL to return 1.0 per preference
    # So Expected Dissat = 1.0 * 2 prefs = 2.0
    mock_kl_divergence.side_effect = lambda m1, p1, m2, p2: jnp.ones_like(m1)

    # 3. Run
    scores = _evaluate_candidate_scores(
        mock_env, beliefs_mean, beliefs_prec, current_dissat
    )

    # 4. Assert
    # Score should be 10.0 - 2.0 = 8.0
    assert scores.shape == (num_agents, num_cands)
    assert jnp.allclose(scores, 8.0)


# --- Core Logic Tests for _vote_quadratic ---


def test_quadratic_math_specific_scenario():
    """Verifies the specific Quadratic Voting formula."""
    budget = 100.0

    # Create specific score matrix manually:
    # Agent 0: Likes Cand A (Score 10) and Cand B (Score 30). Total=40.
    # Agent 1: Hates everyone (Scores < 0).
    scores = jnp.array([[10.0, 30.0], [-5.0, -10.0]])

    # We patch the helper functions so we don't depend on Env/Math,
    # just the voting logic block.
    with (
        patch("src.eci.voting_system.quad._get_current_beliefs_t") as m_bel,
        patch("src.eci.voting_system.quad._get_current_dissatisfaction") as m_diss,
        patch("src.eci.voting_system.quad._evaluate_candidate_scores") as m_score,
    ):
        m_bel.return_value = {}
        m_diss.return_value = (None, None, None)
        m_score.return_value = scores

        # Mock env candidates to match shape (2 candidates)
        env = MagicMock()
        env.candidates = [MagicMock(id=0), MagicMock(id=1)]

        result = _vote_quadratic(env, None, budget=budget)

        # --- Check Agent 0 ---
        # Proportions: A=10/40 (0.25), B=30/40 (0.75)
        # Credits: A=25, B=75
        # Votes: A=Sqrt(25)=5, B=Sqrt(75)=~8.66
        votes_0 = result["vote_matrix"][0]
        assert jnp.isclose(votes_0[0], 5.0)
        assert jnp.isclose(votes_0[1], jnp.sqrt(75.0))

        # --- Check Agent 1 (The Hater) ---
        # Should have spent 0 credits and cast 0 votes
        votes_1 = result["vote_matrix"][1]
        assert jnp.all(votes_1 == 0.0)
        assert jnp.all(result["proportions"][1] == 0.0)


def test_vote_quadratic_zero_division_safety(mock_env):
    """Test that the code doesn't crash if an agent has 0 positive preference."""
    # Force all scores to be negative
    with (
        patch("src.eci.voting_system.quad._evaluate_candidate_scores") as m_score,
        patch("src.eci.voting_system.quad._get_current_beliefs_t"),
        patch("src.eci.voting_system.quad._get_current_dissatisfaction") as m_diss,
    ):
        m_diss.return_value = (None, None, None)
        m_score.return_value = jnp.full((3, 3), -5.0)  # All negative

        result = _vote_quadratic(mock_env, None, budget=99.0)

        # Should result in 0 votes, not NaNs
        assert not jnp.any(jnp.isnan(result["vote_matrix"]))
        assert jnp.all(result["vote_matrix"] == 0.0)


def test_winner_selection_and_ids(mock_env):
    """Test that correct winners are picked and mapped to IDs."""
    scores = jnp.array([[10.0, -5.0, 100.0], [10.0, -5.0, 100.0], [100.0, -5.0, 10.0]])

    with (
        patch(
            "src.eci.voting_system.quad._evaluate_candidate_scores", return_value=scores
        ),
        patch("src.eci.voting_system.quad._get_current_beliefs_t"),
        patch(
            "src.eci.voting_system.quad._get_current_dissatisfaction",
            return_value=(None, None, None),
        ),
    ):
        result = _vote_quadratic(mock_env, None)

        winners = result["winners"]

        # Winner 1 should be Index 2 (ID 30)
        assert winners[0] == 30
        # Winner 2 should be Index 0 (ID 10)
        assert winners[1] == 10


def test_edge_case_padding():
    """Test the padding logic when fewer than 2 candidates exist."""
    # Setup Env with only 1 candidate
    env = MagicMock()
    env.candidates = [MagicMock(id=999)]
    env.voters = range(2)
    env.preferences_idx = range(1)
    env.last_attributes = {}

    # Mock the helper to return scores for 1 candidate
    # Shape (2 agents, 1 candidate)
    scores = jnp.array([[10.0], [10.0]])

    with (
        patch(
            "src.eci.voting_system.quad._evaluate_candidate_scores", return_value=scores
        ),
        patch("src.eci.voting_system.quad._get_current_beliefs_t"),
        patch(
            "src.eci.voting_system.quad._get_current_dissatisfaction",
            return_value=(None, None, None),
        ),
    ):
        result = _vote_quadratic(env, None)

        # Should return [999, 999] due to padding
        assert result["winners"].shape == (2,)
        assert result["winners"][0] == 999
        assert result["winners"][1] == 999
