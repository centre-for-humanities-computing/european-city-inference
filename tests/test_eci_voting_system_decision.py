import jax
import jax.numpy as jnp

from eci.voting_system.decisions import _compute_option_preferences, _sample_choice


class MockCandidate:
    """A mock class representing a candidate in the voting system."""

    def __init__(self, mean, precision):
        self.policy = {"mean": jnp.array(mean), "precision": jnp.array(precision)}


class MockEnv:
    """A mock environment class."""

    def __init__(self, candidates):
        self.candidates = candidates


def test_sample_choice():
    """Verifies that the function correctly takes agent preference scores."""
    key = jax.random.PRNGKey(0)
    # 2 agents, 3 options
    # agent 0 prefers option 0; agent 1 prefers option 2
    preferences = jnp.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])

    vote, softmax_probs = _sample_choice(key, preferences)

    assert vote.shape == (2,)
    assert vote[0] == 0
    assert vote[1] == 2

    assert softmax_probs.shape == (2, 3)
    assert jnp.allclose(softmax_probs[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)


def test_compute_option_preferences():
    """Verifies that the function calculates the final preference."""
    # 2 Candidates
    c0 = MockCandidate([1.0], [10.0])
    c1 = MockCandidate([2.0], [10.0])
    env = MockEnv([c0, c1])

    # 2 Agents
    beliefs_mean = jnp.array([[1.0], [2.0]])
    beliefs_precision = jnp.array([[10.0], [10.0]])
    pref_belief_gap = jnp.array([0.5, 0.8])

    # Score = pref_belief_gap - KL(belief || policy)
    scores = _compute_option_preferences(
        env, beliefs_mean, beliefs_precision, pref_belief_gap
    )

    assert scores.shape == (2, 2)  # (agents, candidates)

    # Check Agent 0
    assert jnp.allclose(scores[0, 0], 0.5)  # Expected score for C0 (KL=0)
    assert scores[0, 1] < 0.5  # Expected score for C1 (KL>0)

    # Check Agent 1
    assert scores[1, 0] < 0.8  # Expected score for C0 (KL>0)
    assert jnp.allclose(scores[1, 1], 0.8)  # Expected score for C1 (KL=0)
