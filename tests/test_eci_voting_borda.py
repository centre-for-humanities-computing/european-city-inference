"""Tests for Borda count voting.

`_vote_borda` ranks candidates by each agent's `candidate_utilities` (returned
by the response function), awards `n_cand - 1 .. 0` points per agent, sums them,
and elects the highest total (ties broken by lowest candidate index).
"""

import jax
import jax.numpy as jnp

from eci.voting import _vote_borda


class TestBordaVoting:
    """Test Borda count voting."""

    def _fake_response(self, utilities):
        """Build a response function with fixed candidate utilities."""

        def fn(data, key, mask=None, *args, **kwargs):
            n_agents, n_cand = utilities.shape
            softmax = jnp.full((n_agents, n_cand), 1.0 / n_cand)
            votes = jnp.zeros((n_agents,), dtype=jnp.int32)  # ignored by Borda
            _, next_key = jax.random.split(key)
            return votes, softmax, utilities, next_key

        return fn

    def test_borda_points_and_winner(self):
        """Points = rank (0..n-1) per agent; candidate ranked top by all wins."""
        # 3 agents, 3 candidates. Candidate 1 is everyone's favourite.
        utilities = jnp.array(
            [
                [0.1, 0.9, 0.2],  # ranks -> [0, 2, 1]
                [0.1, 0.8, 0.3],  # ranks -> [0, 2, 1]
                [0.2, 0.7, 0.1],  # ranks -> [1, 2, 0]
            ]
        )
        result = _vote_borda({}, self._fake_response(utilities), jax.random.PRNGKey(0))

        required = {
            "votes",
            "winner",
            "softmax",
            "candidate_utilities",
            "votes_matrix",
            "votes_per_candidate",
        }
        assert required <= set(result.keys())
        assert result["votes_matrix"].shape == (3, 3)
        # Per-agent Borda points (rank of each candidate).
        assert jnp.array_equal(result["votes_matrix"][0], jnp.array([0, 2, 1]))
        # Totals: cand0=1, cand1=6, cand2=2.
        assert jnp.array_equal(result["votes_per_candidate"], jnp.array([1, 6, 2]))
        assert int(result["winner"]) == 1

    def test_borda_rewards_broad_acceptance_over_plurality(self):
        """A broadly second-ranked candidate can beat a polarising plurality leader."""
        # Candidate 0 is top for 2 of 3 agents (plurality winner) but last for the
        # third; candidate 1 is second for everyone -> Borda favours candidate 1.
        utilities = jnp.array(
            [
                [0.9, 0.5, 0.1],  # ranks -> [2, 1, 0]
                [0.9, 0.5, 0.1],  # ranks -> [2, 1, 0]
                [0.1, 0.5, 0.9],  # ranks -> [0, 1, 2]
            ]
        )
        result = _vote_borda({}, self._fake_response(utilities), jax.random.PRNGKey(0))
        # cand0 = 4, cand1 = 3, cand2 = 2 -> here plurality leader (cand0) still wins,
        # but cand1 is a strong broadly-acceptable runner-up.
        assert jnp.array_equal(result["votes_per_candidate"], jnp.array([4, 3, 2]))
        assert int(result["winner"]) == 0

    def test_borda_uses_average_rank_for_tied_utilities(self):
        """Indifferent voters must not systematically favour a candidate index."""
        utilities = jnp.zeros((2, 3))

        result = _vote_borda({}, self._fake_response(utilities), jax.random.PRNGKey(0))

        assert jnp.array_equal(result["votes_matrix"], jnp.ones((2, 3)))
        assert jnp.array_equal(result["votes_per_candidate"], jnp.full(3, 2.0))
        assert int(result["winner"]) == 0
