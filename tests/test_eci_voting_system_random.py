import jax
import jax.numpy as jnp

# Import your functions here.
# Assuming your file is named 'voting_module.py', adapt the import:
from eci.voting_system.random_voting import (
    _find_top_two_winners,  # Needed to mock its return if not mocking the env deeply
    _sample_vote,
    _vote_random,
)


class MockCandidate:
    """Create MockCandidate."""

    def __init__(self, id, mean, precision):
        self.id = id
        self.policy = {"mean": jnp.array(mean), "precision": jnp.array(precision)}


class MockVoter:
    """Create MockVoter."""

    def __init__(self, id):
        self.id = id


class MockEnv:
    """Create Env."""

    def __init__(self, num_voters, num_candidates, num_preferences):
        self.voters = [MockVoter(i) for i in range(num_voters)]
        self.candidates = [
            MockCandidate(i, [0.0] * num_preferences, [1.0] * num_preferences)
            for i in range(num_candidates)
        ]
        self.preferences_idx = list(range(num_preferences))

        # Mocking last_attributes structure is complex, so we might mock
        # the helper function _get_current_beliefs_t instead in the integration test.
        self.last_attributes = {}


# --- UNIT TESTS FOR HELPER FUNCTIONS ---


def test_find_top_two_winners_standard_case():
    """Test finding top two winners when there is a clear distribution."""
    # Votes: Candidate 1 gets 3 votes, Candidate 2 gets 2 votes, Candidate 0 gets 1 vote
    votes = jnp.array([1, 1, 1, 2, 2, 0])

    winners = _find_top_two_winners(votes)

    # Should be [1, 2] because 1 has most votes, 2 has second most
    assert jnp.array_equal(winners, jnp.array([1, 2]))


def test_find_top_two_winners_tie_breaking():
    """Test behavior when vote counts are equal."""
    # Candidate 0: 2 votes, Candidate 1: 2 votes.
    votes = jnp.array([0, 0, 1, 1])

    winners = _find_top_two_winners(votes)

    # Logic usually picks based on sort order (last seen or index based),
    # but strictly we just need two unique winners here.
    assert len(winners) == 2
    assert 0 in winners
    assert 1 in winners


def test_find_top_two_winners_edge_case_single_candidate():
    """Test when everyone votes for the same person."""
    votes = jnp.array([5, 5, 5, 5])

    winners = _find_top_two_winners(votes)

    # Should pad the result. Winner is 5. Result should be [5, 5].
    assert jnp.array_equal(winners, jnp.array([5, 5]))


def test_sample_vote_logic():
    """Test that masking works and shapes are correct."""
    key = jax.random.PRNGKey(0)
    num_agents = 100
    num_candidates = 3

    # Preferences are all zeros (equal probability)
    preferences = jnp.zeros((num_agents, num_candidates))

    # Mask: Candidate 1 is NOT eligible (False). 0 and 2 are eligible.
    mask = jnp.array([[True, False, True]] * num_agents)

    vote, probs = _sample_vote(key, mask, preferences)

    # Check shapes
    assert vote.shape == (num_agents,)
    assert probs.shape == (num_agents, num_candidates)

    # Check that NO ONE voted for candidate 1
    assert jnp.sum(vote == 1) == 0

    # Check that Prob for candidate 1 is 0.0
    assert jnp.all(probs[:, 1] == 0.0)

    # Check that Prob for 0 and 2 is roughly 0.5
    assert jnp.allclose(probs[:, 0], 0.5)
    assert jnp.allclose(probs[:, 2], 0.5)


# --- INTEGRATION TEST FOR _vote_random ---


def test_vote_random_integration(mocker):
    """Test the main function _vote_random."""
    # 1. Setup Mock Environment
    num_voters = 50
    num_candidates = 4
    env = MockEnv(num_voters, num_candidates, num_preferences=2)
    key = jax.random.PRNGKey(42)

    # 2. Mock the belief/dissatisfaction extraction functions
    # We don't want to test the math of KL divergence here, just the voting logic.

    # Mock _get_current_beliefs_t to return a dummy dict
    mocker.patch(
        "eci.voting_system.random_voting._get_current_beliefs_t", return_value={}
    )

    # Mock _get_current_dissatisfaction to return dummy arrays
    # Return (dissatisfaction_per_agent, beliefs_mean, beliefs_precision)
    dummy_dissatisfaction = jnp.zeros(num_voters)
    mocker.patch(
        "eci.voting_system.random_voting._get_current_dissatisfaction",
        return_value=(dummy_dissatisfaction, None, None),
    )

    # 3. Run the function
    result = _vote_random(env, key)

    # 4. Assertions

    # Check keys exist
    expected_keys = [
        "vote_round_1",
        "softmax_probs_round_1",
        "first_round_winners",
        "vote_final_round_2",
        "softmax_probs_final_round_2",
        "final_winner",
        "dissatisfaction",
    ]
    for k in expected_keys:
        assert k in result

    # Check Round 1
    vote_r1 = result["vote_round_1"]
    assert vote_r1.shape == (num_voters,)
    # In random voting, it's possible any candidate gets votes,
    # but we just check ranges.
    assert jnp.all(vote_r1 >= 0) and jnp.all(vote_r1 < num_candidates)

    # Check Round 2 Winners
    winners = result["first_round_winners"]
    assert winners.shape == (2,)

    # Check Round 2 Votes
    vote_r2 = result["vote_final_round_2"]
    # Verify that in Round 2, people ONLY voted for the top 2 winners
    # jnp.isin returns a boolean mask, all should be True
    is_valid_vote = jnp.isin(vote_r2, winners)
    assert jnp.all(is_valid_vote)

    # Check Final Winner
    final_winner = result["final_winner"]
    # Final winner must be one of the top two
    assert final_winner in winners

    # Check Probabilities (Random voting = Equal probs in Round 1)
    probs_r1 = result["softmax_probs_round_1"]
    expected_prob = 1.0 / num_candidates
    # Allow small float tolerance
    assert jnp.allclose(probs_r1, expected_prob, atol=1e-6)
