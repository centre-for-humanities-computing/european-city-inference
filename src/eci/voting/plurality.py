import jax
import jax.numpy as jnp

from eci.voting.types import VoteResult
from eci.voting.utils import _find_winner


def _vote_plurality(data, response_function, key, *args, **kwargs) -> VoteResult:
    """Perform plurality voting.

    Each voter casts one vote (sampled by ``response_function``); the
    candidate with the most votes wins. Ties are broken by the lowest
    candidate index.

    Parameters
    ----------
    data:
        Agent data (beliefs, preferences, candidates).
    response_function:
        Implements the :class:`~eci.voting_system.ResponseFunction` protocol.
    key:
        A JAX PRNG key used for seeding random operations.

    Returns
    -------
    VoteResult
        See :class:`~eci.voting_system.types.VoteResult` for the full
        field contract.
    """
    # TODO: re-enable runoff (top-2) voting; the prototype lives in the git
    # history (commit before this refactor). Open an issue to track it.
    votes, softmax, candidate_utilities, _key = response_function(data, key)
    n_cand = candidate_utilities.shape[1]

    votes_matrix = jax.nn.one_hot(votes, n_cand, dtype=jnp.int32)
    votes_per_candidate = jnp.sum(votes_matrix, axis=0)
    winner = _find_winner(votes, n_cand)

    return {
        # Uniform fields (preferred):
        "votes_matrix": votes_matrix,
        "votes_per_candidate": votes_per_candidate,
        "winner": winner,
        "softmax": softmax,
        "candidate_utilities": candidate_utilities,
        # Legacy (per-agent indices) — will be removed in v0.2:
        "votes": votes,
    }
