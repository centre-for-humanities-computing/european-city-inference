import numpy as np

from eci.agents import Agent, Candidate, Voter


class TestAgents:
    """Tests for the Agent, Voter, and Candidate classes."""

    def test_agent_base_initialization(self):
        """Test that the base Agent class initializes with an ID."""
        agent = Agent(id=1)
        assert agent.id == 1

    def test_voter_initialization_defaults(self):
        """Test Voter initialization with required fields and check defaults."""
        prefs = {"mean": np.array([0.5]), "precision": np.array([1.0])}
        voter = Voter(id=10, preferences=prefs, tonic_volatility=0.5)

        # Check required fields
        assert voter.id == 10
        assert voter.preferences == prefs
        assert voter.tonic_volatility == 0.5

        # Check defaults
        assert voter.perceived_outcome is None
        assert voter.vote_round_1 == []
        assert voter.vote_round_2 == []
        assert voter.softmax_probs_1 == []
        assert voter.dissatisfactions == []
        assert voter.trajectory is None

    def test_voter_mutable_defaults_independence(self):
        """Verify list attributes are independent."""
        v1 = Voter(id=1, preferences={}, tonic_volatility=0.1)
        v2 = Voter(id=2, preferences={}, tonic_volatility=0.1)

        # Modify v1's list
        v1.vote_round_1.append(99)
        v1.softmax_probs_1.append(0.99)

        # Check v1
        assert v1.vote_round_1 == [99]
        assert v1.softmax_probs_1 == [0.99]

        # Check v2 (Should remain empty)
        assert v2.vote_round_1 == []
        assert v2.softmax_probs_1 == []
        assert v2.vote_round_1 is not v1.vote_round_1

    def test_candidate_initialization(self):
        """Test Candidate initialization and default vote count."""
        policy = {"mean": np.array([1.0]), "precision": np.array([2.0])}
        candidate = Candidate(id=5, policy=policy)

        assert candidate.id == 5
        assert candidate.policy == policy
        # Check default vote_count
        assert candidate.vote_count == 0

    def test_candidate_state_mutability(self):
        """Test that we can update the candidate's state."""
        candidate = Candidate(id=5, policy={})
        candidate.vote_count += 10
        assert candidate.vote_count == 10
