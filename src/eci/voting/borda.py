import jax.numpy as jnp

from eci.voting.types import VoteResult


def _vote_borda(data, response_function, key, *args, **kwargs) -> VoteResult:
    """Perform Borda count voting.

    Each voter ranks the candidates by utility; a candidate scores
    ``n_candidates - 1`` points from a voter who ranks it first, down to ``0``
    for the least preferred. Candidates tied in a voter's ranking share the
    average points for their positions. Points are summed across voters and
    the highest total wins (ties broken by the lowest candidate index).

    Unlike plurality (one vote each), Borda uses each voter's full ordering, so
    it rewards broadly-acceptable candidates over narrowly-favoured ones.

    Parameters
    ----------
    data:
        Agent data (beliefs, preferences, candidates).
    response_function:
        Implements the :class:`~eci.decision.ResponseFunction` protocol. Only
        its returned ``candidate_utilities`` are used (to rank candidates).
    key:
        A JAX PRNG key used for seeding random operations.

    Returns
    -------
    VoteResult
        See :class:`~eci.voting.types.VoteResult`. ``votes_matrix`` holds the
        per-(agent, candidate) Borda points.
    """
    _, softmax, candidate_utilities, _key = response_function(data, key)

    # Borda points per (agent, candidate) = number of candidates it strictly
    # outranks plus half a point per tied peer. This is the average rank and
    # avoids introducing candidate-index bias when utilities are equal.
    pairwise = candidate_utilities[:, :, None] - candidate_utilities[:, None, :]
    lower = jnp.sum(pairwise > 0, axis=-1)
    tied_peers = jnp.sum(pairwise == 0, axis=-1) - 1
    votes_matrix = lower + 0.5 * tied_peers
    votes_per_candidate = jnp.sum(votes_matrix, axis=0)
    winner = jnp.argmax(votes_per_candidate)

    return {
        # Uniform fields (preferred):
        "votes_matrix": votes_matrix,
        "votes_per_candidate": votes_per_candidate,
        "winner": winner,
        "softmax": softmax,
        "candidate_utilities": candidate_utilities,
        # Legacy alias — will be removed in v0.2:
        "votes": votes_per_candidate,
    }
