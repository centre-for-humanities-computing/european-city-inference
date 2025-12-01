import pytest

from eci.agents import Agent, Candidate, Voter


def test_cannot_instantiate_abstract_agent():
    """Ensures that the Agent class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Agent(id=1)


def test_voter_initialization_defaults():
    """Verifies that default values are set correctly."""
    voter = Voter(id=1, preferences={"mean": [0.5]}, tonic_volatility=0.1)

    # Assertions
    assert voter.budget == 100.0
    assert voter.vote_round_1 == []
    assert voter.perceived_outcome is None


def test_voter_mutable_defaults_are_independent():
    """Ensures default_factory=list is working correctly."""
    voter1 = Voter(id=1, preferences={}, tonic_volatility=0.1)
    voter2 = Voter(id=2, preferences={}, tonic_volatility=0.1)

    # Modify voter1
    voter1.vote_round_1.append(1)

    # voter2 must remain unchanged
    assert voter1.vote_round_1 == [1]
    assert voter2.vote_round_1 == []  # If this fails, they share memory pointer


def test_candidate_initialization():
    """Verifies that Candidate initializes correctly with given parameters."""
    cand = Candidate(id=10, policy={"axis": "left"})
    assert cand.vote_count == 0
