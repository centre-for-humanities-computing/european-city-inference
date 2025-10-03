import numpy as np
import pytest

from agents import Agent, Candidate, Voter  # adjust path if needed


def test_abstract_agent_cannot_be_instantiated():
    """Test that the abstract Agent class cannot be instantiated."""
    with pytest.raises(TypeError):
        Agent(id=1)


def test_voter_initialization():
    """Test initialization of a Voter and its default attributes."""
    prefs = {"mean": np.array([0.5, 0.2]), "precision": np.array([1.0, 1.0])}
    v = Voter(id=1, preferences=prefs, tonic_volatility=0.3)

    assert v.id == 1
    assert v.preferences == prefs
    assert v.tonic_volatility == 0.3
    assert v.budget == 100.0
    assert v.last_vote is None
    assert isinstance(v.last_softmax_probs, dict)
    assert isinstance(v.last_dissatisfactions, dict)


def test_candidate_initialization_and_votes():
    """Test Candidate initialization and vote counting behavior."""
    policy = {"mean": np.array([0.1, 0.9]), "precision": np.array([0.5, 0.5])}
    c = Candidate(id=2, policy=policy)

    assert c.id == 2
    assert c.policy == policy
    assert c.vote_count == 0

    # simulate votes
    c.vote_count += 3
    assert c.vote_count == 3


def test_voter_step_placeholder():
    """Test placeholder voter step."""
    prefs = {"mean": np.array([0.1, 0.2]), "precision": np.array([1.0, 1.0])}
    v = Voter(id=1, preferences=prefs, tonic_volatility=0.2)

    # step does nothing for now, but should not crash
    v.step(env={})
    assert v.last_vote is None  # still None since no logic is implemented


def test_candidate_step_placeholder():
    """Test placeholder candidate step."""
    c = Candidate(id=1, policy={"mean": np.array([0.5]), "precision": np.array([1.0])})

    # step does nothing for now, but should not crash
    c.step(env={})
    assert c.vote_count == 0
