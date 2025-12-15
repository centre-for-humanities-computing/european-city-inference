import jax.numpy as jnp

from eci.voting_system.beliefs import (
    _extract_candidates_data,
    _get_current_beliefs,
    _get_pref_belief_gap,
)


class MockCandidate:
    """Represent a candidate in the voting system."""

    def __init__(self, mean, precision):
        self.policy = {"mean": jnp.array(mean), "precision": jnp.array(precision)}


class MockEnv:
    """Simulates the necessary environment attributes."""

    def __init__(self, candidates, voters_count, preferences_idx, last_attributes):
        self.candidates = candidates
        self.voters = list(range(voters_count))
        self.preferences_idx = preferences_idx
        self.last_attributes = last_attributes


def test_extract_candidates_data():
    """Verifies that the function correctly extracts the policy mean and precision."""
    c1 = MockCandidate([1.0], [10.0])
    c2 = MockCandidate([2.0], [5.0])
    env = MockEnv([c1, c2], 0, [], [])

    means, precisions = _extract_candidates_data(env)

    assert means.shape == (2, 1)
    assert precisions.shape == (2, 1)
    assert jnp.allclose(means, jnp.array([[1.0], [2.0]]))
    assert jnp.allclose(precisions, jnp.array([[10.0], [5.0]]))


def test_get_current_beliefs():
    """Checks if the function correctly parses the environment."""
    # Setup
    c1 = MockCandidate([0.0], [1.0])

    # 2 agents, 1 preference dimension
    voters_count = 2
    preferences_idx = [0]

    # Mock agent data
    attr_0 = {"expected_mean": [0.5, 0.6], "precision": [2.0, 3.0]}
    pref_data = {"mean": [[1.0], [2.0]], "precision": [[10.0], [20.0]]}

    last_attributes = {0: attr_0, -1: {"preferences": pref_data}}

    env = MockEnv([c1], voters_count, preferences_idx, last_attributes)

    data = _get_current_beliefs(env)

    assert len(data) == 2
    assert jnp.allclose(data[0]["means_belief"], jnp.array([0.5]))
    assert jnp.allclose(data[1]["means_preference"], jnp.array([2.0]))
    assert jnp.allclose(data[0]["precisions_belief"], jnp.array([2.0]))
    assert jnp.allclose(data[1]["precision_preference"], jnp.array([20.0]))


def test_get_pref_belief_gap():
    """Verifie that the function correctly calculates the gap."""
    # Setup a simple dictionary as returned by _get_current_beliefs
    # Agent 0: KL=0 (Belief=Pref)
    # Agent 1: KL=0.5 (Belief vs Pref difference of 1, precision 1)

    agent0 = {
        "means_belief": jnp.array([0.0]),
        "precisions_belief": jnp.array([1.0]),
        "means_preference": jnp.array([0.0]),
        "precision_preference": jnp.array([1.0]),
    }

    agent1 = {
        "means_belief": jnp.array([0.0]),
        "precisions_belief": jnp.array([1.0]),
        "means_preference": jnp.array([1.0]),
        "precision_preference": jnp.array([1.0]),
    }

    all_agent_data = {0: agent0, 1: agent1}

    gaps = _get_pref_belief_gap(all_agent_data)

    assert gaps.shape == (2,)
    assert jnp.allclose(gaps[0], 0.0)
    assert jnp.allclose(gaps[1], 0.5)
