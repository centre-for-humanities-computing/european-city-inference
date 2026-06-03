"""Tests for the ResponseFunction extension protocol.

These tests verify that the built-in response functions conform to the
public ``ResponseFunction`` contract, and that a user-defined custom
response function plugs in to the voting rules without any registration
step.
"""

import jax
import jax.numpy as jnp
import pytest

from eci.decision import (
    ResponseFunction,
    response_function,
    response_function_logpdf,
    response_function_pref,
)
from eci.voting import _vote_plurality, _vote_quadratic

BUILTIN_RESPONSE_FNS = [
    response_function,
    response_function_logpdf,
    response_function_pref,
]


@pytest.fixture
def small_data():
    """Synthetic dataset for testing."""
    return {
        "beliefs": {
            "mean": jnp.array([[0.1], [0.5], [0.9]]),
            "precision": jnp.array([[1.0], [1.0], [1.0]]),
        },
        "preferences": {
            "mean": jnp.array([[0.2], [0.5], [0.8]]),
            "precision": jnp.array([[1.0], [1.0], [1.0]]),
        },
        "candidates": {
            "mean": jnp.array([[0.3], [0.7]]),
            "precision": jnp.array([[1.0], [1.0]]),
        },
    }


@pytest.fixture
def key():
    """Random key for testing."""
    return jax.random.PRNGKey(0)


# -------------------------------------------------------------------------
# Protocol conformance
# -------------------------------------------------------------------------
@pytest.mark.parametrize("fn", BUILTIN_RESPONSE_FNS, ids=lambda f: f.__name__)
def test_builtin_response_fns_satisfy_protocol(fn):
    """All shipped response functions must be ResponseFunction-conformant."""
    assert isinstance(fn, ResponseFunction), (
        f"{fn.__name__} does not satisfy the ResponseFunction protocol"
    )


@pytest.mark.parametrize("fn", BUILTIN_RESPONSE_FNS, ids=lambda f: f.__name__)
def test_response_fn_return_shapes(fn, small_data, key):
    """Vote / softmax / utilities / next_key must have the documented shapes."""
    n_agents = small_data["preferences"]["mean"].shape[0]
    n_candidates = small_data["candidates"]["mean"].shape[0]

    vote, softmax_probs, utilities, next_key = fn(small_data, key)

    assert vote.shape == (n_agents,)
    assert softmax_probs.shape == (n_agents, n_candidates)
    assert utilities.shape == (n_agents, n_candidates)
    # next_key is a JAX PRNG key — modern JAX represents these as uint32 arrays
    # of shape (2,) (legacy) or unsigned key dtype. We just check it's not None
    # and is distinct from the input key.
    assert next_key is not None
    assert not jnp.array_equal(jnp.asarray(next_key), jnp.asarray(key))


@pytest.mark.parametrize("fn", BUILTIN_RESPONSE_FNS, ids=lambda f: f.__name__)
def test_softmax_rows_sum_to_one(fn, small_data, key):
    """Softmax probabilities are a valid distribution per agent."""
    _, softmax_probs, _, _ = fn(small_data, key)
    row_sums = jnp.sum(softmax_probs, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5)


@pytest.mark.parametrize("fn", BUILTIN_RESPONSE_FNS, ids=lambda f: f.__name__)
def test_mask_excludes_candidates(fn, small_data, key):
    """When mask=False for a candidate, no agent should vote for them."""
    mask = jnp.array([True, False])
    _, softmax_probs, _, _ = fn(small_data, key, mask=mask)
    # Masked candidate (col 1) should have ~0 probability everywhere.
    assert jnp.allclose(softmax_probs[:, 1], 0.0, atol=1e-5)


# -------------------------------------------------------------------------
# Custom response function plugs in without registration
# -------------------------------------------------------------------------
def custom_uniform_response(data, key, mask=None, *args, **kwargs):
    """Trivial example: every candidate is equally good for every agent."""
    n_agents = data["preferences"]["mean"].shape[0]
    n_candidates = data["candidates"]["mean"].shape[0]
    utilities = jnp.zeros((n_agents, n_candidates))
    if mask is not None:
        utilities = jnp.where(mask, utilities, -jnp.inf)
    sample_key, next_key = jax.random.split(key)
    softmax_probs = jax.nn.softmax(utilities, axis=1)
    vote = jax.random.categorical(sample_key, utilities, axis=1)
    return vote, softmax_probs, utilities, next_key


def test_custom_response_satisfies_protocol():
    """A user-defined response function satisfies the ResponseFunction protocol."""
    assert isinstance(custom_uniform_response, ResponseFunction)


def test_custom_response_works_with_plurality(small_data, key):
    """A user-defined response function works with _vote_plurality."""
    result = _vote_plurality(small_data, custom_uniform_response, key)
    assert "winner" in result
    assert "votes" in result
    assert result["votes"].shape == (small_data["preferences"]["mean"].shape[0],)


def test_custom_response_works_with_quadratic(small_data, key):
    """A user-defined response function works with _vote_quadratic."""
    result = _vote_quadratic(small_data, custom_uniform_response, key)
    assert "votes" in result
    assert result["votes"].shape == (small_data["candidates"]["mean"].shape[0],)
