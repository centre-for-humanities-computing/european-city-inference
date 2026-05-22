"""Tests for quadratic voting.

Notes on the refactor:
- `_vote_quadratic` no longer takes `env` but `(data, response_function, key)`.
  Its return dict now uses keys `votes`, `winner`, `softmax`,
  `candidate_utilities`, `credits_spent`, `qv_votes_matrix` (no more
  `final_winner`, `vote_round_1`, `softmax_probs_round_1`).
- `_compute_sequential_qv_allocation` was rewritten with Gumbel-top-k
  sampling — it no longer calls `_sample_choice` in a loop, so the old
  `mock_sample.call_count == 5` assertion no longer applies.
- `strategic_quadratic_vote` is currently commented out.
"""

import jax
import jax.numpy as jnp
import pytest

from eci.voting_system.quadratic import (
    _compute_sequential_qv_allocation,
    _vote_quadratic,
)


class TestQuadraticVoting:
    """Test quadratic voting."""

    def test_vote_quadratic_winner_logic(self):
        """Smoke-test the new (data, response_function, key) signature.

        Candidate 0 receives much higher utility than candidate 1, so Gumbel-
        top-k allocation should put all its credits on candidate 0 and
        `winner` should be 0.
        """
        n_agents, n_cand = 3, 2
        # Strongly prefer candidate 0 for every agent.
        utilities = jnp.tile(jnp.array([10.0, -10.0]), (n_agents, 1))
        softmax = jax.nn.softmax(utilities, axis=1)
        votes = jnp.zeros((n_agents,), dtype=jnp.int32)

        def fake_response_function(data, key, mask=None, *args, **kwargs):
            sample_key, next_key = jax.random.split(key)
            return votes, softmax, utilities, next_key

        results = _vote_quadratic(
            data={}, response_function=fake_response_function, key=jax.random.PRNGKey(0)
        )

        assert set(results.keys()) == {
            "votes",
            "softmax",
            "winner",
            "credits_spent",
            "qv_votes_matrix",
            "candidate_utilities",
        }
        assert int(results["winner"]) == 0
        assert results["qv_votes_matrix"].shape == (n_agents, n_cand)
        # Candidate 0 dominates total votes.
        assert int(results["votes"][0]) >= int(results["votes"][1])

    def test_compute_sequential_qv_allocation_shapes(self):
        """Allocation produces integer vote matrices of the expected shape."""
        key = jax.random.PRNGKey(42)
        # 2 agents, 3 candidates
        candidate_utilities = jnp.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
        budget = 100.0

        votes_matrix, credits_spent = _compute_sequential_qv_allocation(
            key, candidate_utilities, budget
        )

        assert votes_matrix.shape == (2, 3)
        assert credits_spent.shape == (2, 3)
        assert votes_matrix.dtype == jnp.int32
        # Credits should be non-negative (sqrt would fail otherwise).
        assert jnp.all(credits_spent >= 0.0)

    def test_compute_sequential_qv_allocation_zero_budget(self):
        """Zero budget should yield zero votes and zero credits without NaNs."""
        key = jax.random.PRNGKey(42)
        candidate_utilities = jnp.zeros((2, 3))
        budget = 0.0

        votes_matrix, credits_spent = _compute_sequential_qv_allocation(
            key, candidate_utilities, budget
        )

        assert not jnp.isnan(credits_spent).any()
        assert not jnp.isnan(votes_matrix).any()
        assert jnp.all(votes_matrix == 0)
        assert jnp.all(credits_spent == 0.0)

    @pytest.mark.skip(
        reason="strategic_quadratic_vote is currently commented out in "
        "quadratic.py. Re-enable once strategic QV is restored."
    )
    def test_strategic_quadratic_vote_flow(self):
        """Test the strategic QV flow. TODO: restore."""
        pass
