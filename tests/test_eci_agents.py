import pytest

from eci.agents import Agent, Candidate, Voter


def test_cannot_instantiate_abstract_agent():
    """Ensure that the Agent class cannot be instantiated directly."""
    with pytest.raises(TypeError) as excinfo:
        # Attempting to instantiate the abstract class
        Agent(id=1)

    # We can even check the error message to be precise
    assert "Can't instantiate abstract class Agent" in str(excinfo.value)


def test_voter_initialization_defaults():
    """Verifies that default values are set correctly."""
    voter = Voter(id=1, preferences={"mean": [0.5]}, tonic_volatility=0.1)

    # Assertions
    assert voter.id == 1
    assert voter.budget == 100.0
    assert voter.vote_round_1 == []
    assert voter.perceived_outcome is None


def test_voter_mutable_defaults_are_independent():
    """Verifies that default_factory=list is working correctly."""
    voter1 = Voter(id=1, preferences={}, tonic_volatility=0.1)
    voter2 = Voter(id=2, preferences={}, tonic_volatility=0.1)

    # Modify voter1
    voter1.vote_round_1.append(99)

    # Verifications
    assert voter1.vote_round_1 == [99]
    assert voter2.vote_round_1 == []  # voter2 must remain empty!


def test_candidate_initialization():
    """Verifies that Candidate initializes correctly with given parameters."""
    cand = Candidate(id=10, policy={"axis": "left"})

    assert cand.id == 10
    assert cand.policy == {"axis": "left"}
    assert cand.vote_count == 0


def test_voter_step_execution():
    """Call the step method on a Voter instance to ensure the 'pass' statement."""
    voter = Voter(id=1, preferences={}, tonic_volatility=0.1)

    # Calling step should not raise any errors
    voter.step(env=None)


def test_candidate_step_execution():
    """Call the step method on a Candidate instance to ensure the 'pass' statement."""
    cand = Candidate(id=10, policy={"axis": "left"})

    # Calling step should not raise any errors
    cand.step(env=None)


def test_agent_abstract_step_coverage():
    """Creates a concrete subclass of Agent solely to call super().step()."""

    class ConcreteAgentForTest(Agent):
        def step(self):
            # This explicit call triggers the code in the abstract base class
            super().step(env=None)

    # Instantiate the concrete test class
    agent = ConcreteAgentForTest(id=999)
    agent.step()
