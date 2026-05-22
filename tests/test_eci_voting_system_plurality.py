"""Tests for plurality voting.

Notes on the refactor:
- `_find_top_two_winners` was generalised to `_find_top_k_winners` in
  `eci.utils`. The single-winner case used in plurality is now
  `eci.utils._find_winner`.
- `_vote_plurality` no longer takes `env` but `(data, response_function, key)`
  and the second-round (top-2 runoff) is currently disabled (commented out)
  inside the function.
- `strategic_vote` is currently commented out in `plurality.py`.
"""

import jax
import jax.numpy as jnp
import pytest

from eci.utils import _find_top_k_winners, _find_winner
from eci.voting_system.plurality import _vote_plurality


class TestPluralityVoting:
    """Test plurality voting."""

    def test_find_winner_clear(self):
        """Top-1 is found when counts are distinct."""
        votes = jnp.array([1, 1, 1, 0])
        winner = _find_winner(votes, num_candidates=3)
        assert int(winner) == 1

    def test_find_top_k_winners_clear(self):
        """Top-2 is found when counts are distinct (replaces _find_top_two_winners)."""
        votes = jnp.array([1, 1, 1, 0])
        winners = _find_top_k_winners(votes, num_candidates=3, k=2)
        assert int(winners[0]) == 1
        assert int(winners[1]) == 0

    def test_find_top_k_winners_tie(self):
        """Top-2 still returns a valid 2nd place when there is a tie."""
        votes = jnp.array([0, 0, 1, 2])
        winners = _find_top_k_winners(votes, num_candidates=3, k=2)
        assert int(winners[0]) == 0
        assert int(winners[1]) in [1, 2]

    def test_vote_plurality_basic_flow(self):
        """Smoke-test the new (data, response_function, key) signature."""
        # 3 agents, 3 candidates. Use a fake response_function so we can
        # control the votes without driving full HGF/KL machinery.
        n_agents, n_cand = 3, 3
        votes = jnp.array([0, 1, 0])
        softmax = jnp.full((n_agents, n_cand), 1.0 / n_cand)
        utilities = jnp.zeros((n_agents, n_cand))

        def fake_response_function(data, key, mask=None, *args, **kwargs):
            sample_key, next_key = jax.random.split(key)
            return votes, softmax, utilities, next_key

        data = {}  # response_function ignores it here
        key = jax.random.PRNGKey(0)

        results = _vote_plurality(data, fake_response_function, key)

        assert set(results.keys()) == {
            "votes",
            "winner",
            "softmax",
            "candidate_utilities",
        }
        # Candidate 0 has 2 votes, candidate 1 has 1 vote, candidate 2 has 0.
        assert int(results["winner"]) == 0
        assert jnp.array_equal(results["votes"], votes)
        assert results["softmax"].shape == (n_agents, n_cand)

    @pytest.mark.skip(
        reason="2-round plurality (top-2 runoff) is currently commented out in "
        "_vote_plurality. Re-enable once the runoff block is restored."
    )
    def test_vote_plurality_round_logic(self):
        """Verifies the 2-Round Plurality Logic. TODO: restore."""
        pass

    @pytest.mark.skip(
        reason="strategic_vote is currently commented out in plurality.py. "
        "Re-enable once strategic plurality voting is restored."
    )
    def test_strategic_vote_weighting(self):
        """Verifies Strategic Plurality Vote weighting. TODO: restore."""
        pass
