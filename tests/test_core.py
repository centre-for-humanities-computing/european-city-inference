import jax
import jax.numpy as jnp
import numpy as np

from core_logic import (
    get_votes,
    init_preferences,
    kl_divergence,
    total_dissatisfaction_per_candidate,
)


def test_kl_divergence_zero():
    """Test that KL divergence."""
    mean = jnp.array([0.0])
    precision = jnp.array([1.0])
    kl = kl_divergence(mean, precision, mean, precision)
    assert jnp.allclose(kl, 0.0)


def test_kl_divergence_asymmetry_with_different_precisions():
    """Test that KL divergence is not symmetric when variances differ."""
    mean1, mean2 = jnp.array([0.0]), jnp.array([1.0])
    prec1, prec2 = jnp.array([1.0]), jnp.array([2.0])  # different variances
    kl1 = kl_divergence(mean1, prec1, mean2, prec2)
    kl2 = kl_divergence(mean2, prec2, mean1, prec1)
    assert not jnp.allclose(kl1, kl2)


def test_init_preferences_random_shape():
    """Test initialization of random preferences has correct shape."""
    prefs = init_preferences(n_agents=3, n_preferences=2)
    assert prefs["mean"].shape == (3, 2)
    assert prefs["precision"].shape == (3, 2)


def test_init_preferences_manual():
    """Test initialization of preferences using manual means and precisions."""
    mus = np.ones((2, 3))
    pis = np.ones((2, 3))
    prefs = init_preferences(2, 3, manual_means=mus, manual_precisions=pis)
    assert np.allclose(prefs["mean"], mus)
    assert np.allclose(prefs["precision"], pis)


def test_total_dissatisfaction_per_candidate():
    """Test trivial case where belief == preference."""
    node_trajectories = {
        0: {"expected_mean": jnp.array([1.0]), "expected_precision": jnp.array([1.0])}
    }
    input_idxs = (0,)
    candidates = [(jnp.array([1.0]), jnp.array([1.0]))]
    attributes = [
        {"preferences": {"mean": jnp.array([1.0]), "precision": jnp.array([1.0])}}
    ]
    res = total_dissatisfaction_per_candidate(
        node_trajectories, input_idxs, candidates, attributes
    )
    assert res.shape == (1,)
    assert jnp.allclose(res, 0.0)


def test_get_votes_plurality():
    """Test plurality voting produces valid votes and softmax probabilities."""
    key = jax.random.PRNGKey(0)
    attributes = [
        {"preferences": {"mean": jnp.array([0.0]), "precision": jnp.array([1.0])}}
    ]

    # Use a tuple instead of dict → hashable
    class Edge:
        """Mock edge class with a parent reference."""

        def __init__(self, parent):
            """Initialize the edge with a given parent."""
            self.value_parents = [parent]

    edges = (Edge(0),)  # tuple instead of dict

    node_trajectories = {
        0: {"expected_mean": [jnp.array(0.0)], "expected_precision": [jnp.array(1.0)]}
    }
    input_idxs = (0,)
    candidates = [
        (jnp.array([0.0]), jnp.array([1.0])),
        (jnp.array([1.0]), jnp.array([1.0])),
    ]
    mask = jnp.array([True, True])

    vote, softmax_probs, _ = get_votes(
        key,
        attributes,
        edges,
        node_trajectories,
        input_idxs,
        candidates,
        mask,
        voting_system="Plurality Voting",
    )

    assert isinstance(vote, (jnp.ndarray, int))
    assert softmax_probs.shape == (2,)
    assert jnp.allclose(jnp.sum(softmax_probs), 1.0)
