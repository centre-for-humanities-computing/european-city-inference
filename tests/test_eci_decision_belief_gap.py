"""Tests for belief/preference/candidate KL gap helpers.

Note: these helpers used to live in `eci.voting_system.beliefs` (now deleted)
and have been merged into `eci.voting_system.decisions`. `_get_pref_belief_gap`
was renamed to `_get_belief_preference_gap` (KL(beliefs || preferences)).
Their public signatures now take raw arrays instead of a `data` dict.
"""

import jax.numpy as jnp
import pytest

from eci.decision import (
    _get_belief_preference_gap,
    _get_pref_candidate_gap,
)


@pytest.fixture
def mock_voting_arrays():
    """Create mock arrays mirroring the old `data` dict layout."""
    n_agents = 2
    n_candidates = 2
    n_dim = 3

    return {
        "beliefs_mean": jnp.zeros((n_agents, n_dim)),
        "beliefs_precision": jnp.ones((n_agents, n_dim)),
        "pref_mean": jnp.zeros((n_agents, n_dim)),
        "pref_precision": jnp.ones((n_agents, n_dim)),
        "cand_mean": jnp.zeros((n_candidates, n_dim)),
        "cand_precision": jnp.ones((n_candidates, n_dim)),
    }


def test_get_belief_preference_gap_identical(mock_voting_arrays):
    """KL divergence is zero when beliefs and preferences are identical."""
    gaps = _get_belief_preference_gap(
        mock_voting_arrays["beliefs_mean"],
        mock_voting_arrays["beliefs_precision"],
        mock_voting_arrays["pref_mean"],
        mock_voting_arrays["pref_precision"],
    )

    assert gaps.shape == (2,)  # One gap per agent
    assert jnp.allclose(gaps, 0.0)


def test_get_belief_preference_gap_value_check():
    """With unit precisions, KL per dim is 0.5*(μ_b-μ_p)^2; should sum across dims."""
    gaps = _get_belief_preference_gap(
        beliefs_mean=jnp.array([[1.0, 1.0]]),
        beliefs_precision=jnp.array([[1.0, 1.0]]),
        pref_mean=jnp.array([[0.0, 0.0]]),
        pref_precision=jnp.array([[1.0, 1.0]]),
    )
    # 0.5 per dimension, summed across 2 dims = 1.0
    assert jnp.isclose(gaps[0], 1.0)


def test_get_pref_candidate_gap_broadcasting(mock_voting_arrays):
    """Every agent is compared against every candidate."""
    gaps = _get_pref_candidate_gap(
        mock_voting_arrays["pref_mean"],
        mock_voting_arrays["pref_precision"],
        mock_voting_arrays["cand_mean"],
        mock_voting_arrays["cand_precision"],
    )

    assert gaps.shape == (2, 2)


def test_get_pref_candidate_gap_values():
    """Verify broadcasting with hand-computed KL values."""
    gaps = _get_pref_candidate_gap(
        pref_mean=jnp.array([[0.0]]),
        pref_precision=jnp.array([[1.0]]),
        cand_mean=jnp.array([[1.0], [2.0]]),
        cand_precision=jnp.array([[1.0], [1.0]]),
    )
    # KL(Preferences || Candidate 1) = 0.5, KL(Preferences || Candidate 2) = 2.0
    assert jnp.allclose(gaps, jnp.array([[0.5, 2.0]]))
