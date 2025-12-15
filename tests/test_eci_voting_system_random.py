from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from eci.voting_system.random_voting import _vote_random


def test_vote_random():
    """Tests the `_vote_random` function."""
    # Setup
    c0 = MagicMock()
    c0.id = 100
    c1 = MagicMock()
    c1.id = 101
    c2 = MagicMock()
    c2.id = 102

    env = MagicMock()
    env.candidates = [c0, c1, c2]
    env.voters = [0, 1, 2]  # 3 voters

    key = jax.random.PRNGKey(42)

    results = _vote_random(env, key)

    assert "vote_round_1" in results
    assert "first_round_winners" in results
    assert "vote_final_round_2" in results
    assert "final_winner" in results

    vote_r1 = results["vote_round_1"]
    assert vote_r1.shape == (3,)
    # Vote values should be IDs
    assert jnp.all(jnp.isin(vote_r1, jnp.array([100, 101, 102])))

    winners_r1 = results["first_round_winners"]
    assert winners_r1.shape == (2,)
    assert jnp.all(jnp.isin(winners_r1, jnp.array([100, 101, 102])))

    final_winner = results["final_winner"]
    # Should be an ID
    assert final_winner.shape == ()
    assert final_winner in [100, 101, 102]
