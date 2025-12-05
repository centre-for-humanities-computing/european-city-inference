from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.eci.voting_system.plurality import (
    _evaluate_candidate_scores,
    _find_top_two_winners,
    _get_current_beliefs_t,
    _get_current_dissatisfaction,
    _sample_vote,
    _update_public_poll,
    _vote_plurality,
)


@pytest.fixture
def mock_kl_divergence():
    """Mock kl_divergence to return a simple difference for predictable testing."""
    with patch("src.eci.voting_system.plurality.kl_divergence") as mock_kl:
        # Mock behavior: return squared difference of means for simplicity
        def side_effect(m1, p1, m2, p2):
            return jnp.ones_like(m1)

        mock_kl.side_effect = side_effect
        yield mock_kl


@pytest.fixture
def mock_env():
    """Create a mock environment with the complex nested structure required."""
    env = MagicMock()

    # Constants
    num_agents = 5
    num_prefs = 2
    num_candidates = 3

    env.voters = range(num_agents)
    env.preferences_idx = range(num_prefs)

    # Mock Candidates
    candidates = []
    for i in range(num_candidates):
        c = MagicMock()
        c.id = i
        c.policy = {
            "mean": jnp.array([float(i), float(i)]),
            "precision": jnp.array([1.0, 1.0]),
        }
        candidates.append(c)
    env.candidates = candidates

    last_attributes = {}

    # Fill preference attributes
    for p_idx in range(num_prefs):
        last_attributes[p_idx] = {
            "expected_mean": [0.5] * num_agents,
            "precision": [1.0] * num_agents,
        }

    # Fill global agent preferences (usually index -1)
    last_attributes[-1] = {
        "preferences": {
            "mean": [[0.8] * num_prefs for _ in range(num_agents)],
            "precision": [[2.0] * num_prefs for _ in range(num_agents)],
        }
    }

    env.last_attributes = last_attributes

    # For _update_public_poll
    env.use_theory_of_mind = True
    env.public_poll = None

    return env


# --- Unit Tests ---


def test_find_top_two_winners_standard():
    """Test finding winners when there is a clear distribution."""
    # Votes: Candidate 0 gets 3 votes, Candidate 1 gets 2 votes, Candidate 2 gets 0
    votes = jnp.array([0, 0, 0, 1, 1])

    winners = _find_top_two_winners(votes)

    # Winners should be 0 (most) and 1 (second most)
    assert winners[0] == 0
    assert winners[1] == 1
    assert winners.shape == (2,)


def test_find_top_two_winners_padding():
    """Test edge case: only one candidate received votes."""
    # Everyone voted for candidate 2
    votes = jnp.array([2, 2, 2, 2])

    winners = _find_top_two_winners(votes)

    # Should return [2, 2] due to padding logic
    assert jnp.array_equal(winners, jnp.array([2, 2]))


def test_find_top_two_winners_tie_breaking():
    """Test behavior when counts are equal."""
    # 0 gets 2 votes, 1 gets 2 votes, 2 gets 1 vote
    votes = jnp.array([0, 0, 1, 1, 2])

    winners = _find_top_two_winners(votes)

    # Should pick 0 and 1
    expected = jnp.array([0, 1])
    assert set(np.array(winners)) == set(np.array(expected))


def test_get_current_beliefs_t(mock_env):
    """Test extraction of beliefs from environment."""
    data = _get_current_beliefs_t(mock_env)

    assert len(data) == 5  # 5 agents
    assert "means_belief" in data[0]
    assert data[0]["means_belief"].shape == (2,)  # 2 preferences
    # Check value from mock setup
    assert data[0]["means_belief"][0] == 0.5


def test_get_current_dissatisfaction(mock_env, mock_kl_divergence):
    """Test stacking and summation of dissatisfaction."""
    # 1. Get data
    agent_data = _get_current_beliefs_t(mock_env)

    # 2. Run function
    dissatisfaction, means, precisions = _get_current_dissatisfaction(
        mock_env, agent_data
    )

    # Checks
    num_agents = 5
    num_prefs = 2

    assert dissatisfaction.shape == (num_agents,)
    assert means.shape == (num_agents, num_prefs)
    assert precisions.shape == (num_agents, num_prefs)

    assert jnp.all(dissatisfaction == 2.0)


def test_evaluate_candidate_scores(mock_env, mock_kl_divergence):
    """Test score calculation (Current Dissatisfaction - Expected Dissatisfaction)."""
    num_agents = 5
    num_candidates = 3
    num_prefs = 2

    # Mock inputs
    beliefs_mean = jnp.zeros((num_agents, num_prefs))
    beliefs_prec = jnp.ones((num_agents, num_prefs))
    # Let's say current dissatisfaction is high (10)
    current_dissatisfaction = jnp.full((num_agents,), 10.0)

    # Mock KL to return 1.0 per preference -> Expected dissat = 2.0
    scores = _evaluate_candidate_scores(
        mock_env, beliefs_mean, beliefs_prec, current_dissatisfaction
    )

    # Result shape should be (num_agents, num_candidates)
    assert scores.shape == (num_agents, num_candidates)

    # Score = 10.0 (current) - 2.0 (expected) = 8.0
    assert jnp.all(scores == 8.0)


def test_sample_vote():
    """Test masking and softmax voting logic."""
    key = jax.random.PRNGKey(0)

    # 2 Agents, 3 Candidates
    # Agent 0 prefers Candidate 0 (score 100)
    # Agent 1 prefers Candidate 2 (score 100)
    preferences = jnp.array([[100.0, 10.0, 10.0], [10.0, 10.0, 100.0]])

    # Mask out Candidate 1 (index 1)
    mask = jnp.array([[True, False, True], [True, False, True]])

    vote, probs = _sample_vote(key, mask, preferences)

    # Check shapes
    assert vote.shape == (2,)
    assert probs.shape == (2, 3)

    # Check Masking: Probability of masked candidate (index 1) should be 0
    assert jnp.all(probs[:, 1] == 0.0)

    # Check Logic:
    # Agent 0 should vote 0
    # Agent 1 should vote 2
    assert vote[0] == 0
    assert vote[1] == 2


def test_vote_plurality_integration(mock_env, mock_kl_divergence):
    """Full integration test of the voting process."""
    key = jax.random.PRNGKey(42)

    results = _vote_plurality(mock_env, key)

    required_keys = [
        "vote_round_1",
        "softmax_probs_round_1",
        "first_round_winners",
        "vote_final_round_2",
        "softmax_probs_final_round_2",
        "final_winner",
        "dissatisfaction",
    ]

    for k in required_keys:
        assert k in results

    # Check Shapes
    num_agents = 5
    assert results["vote_round_1"].shape == (num_agents,)
    assert results["vote_final_round_2"].shape == (num_agents,)
    assert results["first_round_winners"].shape == (2,)
    assert results["final_winner"].shape == ()  # Scalar

    # Verify Round 2 logic:
    # The final winner must be one of the top two from round 1
    top_two = results["first_round_winners"]
    final = results["final_winner"]
    assert final in top_two


def test_update_public_poll():
    """Test the public poll update method."""

    # Mock 'self' object
    class MockSelf:
        def __init__(self):
            self.use_theory_of_mind = True
            self.public_poll = None
            self.candidates = [MagicMock(id=0), MagicMock(id=1), MagicMock(id=2)]

    obj = MockSelf()
    vote_counts = {0: 50, 1: 30, 2: 20}  # Total 100

    _update_public_poll(obj, vote_counts)

    assert obj.public_poll is not None
    assert obj.public_poll.shape == (3,)
    # Check proportions
    expected = jnp.array([0.5, 0.3, 0.2])
    assert jnp.allclose(obj.public_poll, expected)


def test_update_public_poll_no_votes():
    """Test public poll update with zero votes."""

    class MockSelf:
        def __init__(self):
            self.use_theory_of_mind = True
            self.public_poll = None
            self.candidates = [MagicMock(id=0)]

    obj = MockSelf()
    _update_public_poll(obj, {})  # Empty dict
    assert obj.public_poll is None

    _update_public_poll(obj, {0: 0})  # Zero total
    assert obj.public_poll is None
