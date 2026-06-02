"""Tests for the candidate-utility scoring strategies."""

import jax.numpy as jnp
import pytest

from eci.decision import (
    ScoringFn,
    _compute_candidate_utilities,
    score_absolute,
    score_inverted,
    score_normalized,
    score_product,
)

ALL_STRATEGIES = [
    score_normalized,
    score_absolute,
    score_inverted,
    score_product,
]


@pytest.fixture
def gaps():
    """Synthetic (belief_gap, pref_candidate_gap) for a 3-agent, 2-candidate case."""
    belief_gap = jnp.array([0.1, 1.0, 5.0])  # increasing dissatisfaction
    pref_cand_gap = jnp.array(
        [
            [0.05, 0.50],  # agent 0: C0 close, C1 far
            [0.50, 0.05],  # agent 1: C0 far,   C1 close
            [2.50, 2.50],  # agent 2: both equidistant
        ]
    )
    return belief_gap, pref_cand_gap


# -------------------------------------------------------------------------
# Protocol conformance + shape contract
# -------------------------------------------------------------------------
@pytest.mark.parametrize("fn", ALL_STRATEGIES, ids=lambda f: f.__name__)
def test_strategies_satisfy_protocol(fn):
    """All strategies must satisfy the ScoringFn protocol."""
    assert isinstance(fn, ScoringFn)


@pytest.mark.parametrize("fn", ALL_STRATEGIES, ids=lambda f: f.__name__)
def test_output_shape(fn, gaps):
    """Output shape matches pref_cand_gap shape (n_agents, n_candidates)."""
    belief_gap, pref_cand_gap = gaps
    u = fn(belief_gap, pref_cand_gap)
    assert u.shape == pref_cand_gap.shape


# -------------------------------------------------------------------------
# Per-strategy semantics
# -------------------------------------------------------------------------
def test_normalized_sign_convention(gaps):
    """Normalised: U > 0 when candidate beats status quo (pref_gap < belief_gap)."""
    belief_gap, pref_cand_gap = gaps
    u = score_normalized(belief_gap, pref_cand_gap)
    assert u[0, 0] > 0
    assert u[0, 1] < 0
    assert u[2, 0] > 0


def test_absolute_is_unnormalised_difference(gaps):
    """Absolute: U = belief_gap - pref_cand_gap (no normalization)."""
    belief_gap, pref_cand_gap = gaps
    u = score_absolute(belief_gap, pref_cand_gap)
    expected = belief_gap[:, None] - pref_cand_gap
    assert jnp.allclose(u, expected)


def test_inverted_is_negated_absolute(gaps):
    """Inverted: U = -(belief_gap - pref_cand_gap) = pref_cand_gap - belief_gap."""
    belief_gap, pref_cand_gap = gaps
    assert jnp.allclose(
        score_inverted(belief_gap, pref_cand_gap),
        -score_absolute(belief_gap, pref_cand_gap),
    )


def test_product_multiplies_dissatisfaction_and_distance(gaps):
    """Product: a satisfied agent (belief_gap=0) is indifferent across candidates."""
    belief_gap, pref_cand_gap = gaps
    # Override agent 0 to be perfectly satisfied.
    belief_gap = belief_gap.at[0].set(0.0)
    u = score_product(belief_gap, pref_cand_gap)
    # Agent 0 row should be all zero (no discrimination).
    assert jnp.allclose(u[0], 0.0)
    assert u[2, 0] < u[1, 0]


def test_product_argmax_matches_min_distance(gaps):
    """For any agent with belief_gap > 0, argmax(product) = argmin(pref_cand_gap)."""
    belief_gap, pref_cand_gap = gaps
    u = score_product(belief_gap, pref_cand_gap)
    # Agent 0 prefers C0 (closer); agent 1 prefers C1 (closer).
    assert jnp.argmax(u[0]) == 0
    assert jnp.argmax(u[1]) == 1


# -------------------------------------------------------------------------
# Integration with _compute_candidate_utilities
# -------------------------------------------------------------------------
@pytest.fixture
def small_data():
    """Small synthetic data dict for testing the full utility computation pipeline."""
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


def test_default_scoring_is_normalized(small_data):
    """Calling _compute_candidate_utilities with no scoring_fn = score_normalized."""
    default = _compute_candidate_utilities(small_data)
    explicit = _compute_candidate_utilities(small_data, scoring_fn=score_normalized)
    assert jnp.allclose(default[0], explicit[0])


@pytest.mark.parametrize("fn", ALL_STRATEGIES, ids=lambda f: f.__name__)
def test_compute_utilities_dispatch(small_data, fn):
    """All four strategies plug in via the scoring_fn parameter."""
    utilities, pref_gap, belief_gap = _compute_candidate_utilities(
        small_data, scoring_fn=fn
    )
    assert utilities.shape == pref_gap.shape
    assert belief_gap.shape == (small_data["preferences"]["mean"].shape[0],)
