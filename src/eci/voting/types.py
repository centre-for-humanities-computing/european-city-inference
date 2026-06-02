from typing import TypedDict

from jax.typing import ArrayLike


class VoteResult(TypedDict, total=False):
    """Uniform return type for voting rules.

    The fields under "Always present" are guaranteed by every voting
    function in :mod:`eci.voting_system`. Extra fields appear only for
    specific rules (e.g. ``credits_spent`` for quadratic).

    Always present
    --------------
    votes_matrix : ArrayLike, shape (n_agents, n_candidates)
        Per-(agent, candidate) vote count. One-hot for plurality;
        integer-valued (from QV sqrt rule) for quadratic.
    votes_per_candidate : ArrayLike, shape (n_candidates,)
        Total votes per candidate, summed over agents.
    winner : ArrayLike, scalar int
        Elected candidate index, equal to ``argmax(votes_per_candidate)``
        with deterministic tie-breaking on the lowest index.
    softmax : ArrayLike, shape (n_agents, n_candidates)
        Per-agent intended vote distribution (softmax of utilities).
    candidate_utilities : ArrayLike, shape (n_agents, n_candidates)
        Raw scores before softmax.

    QV-specific
    -----------
    credits_spent : ArrayLike, shape (n_agents, n_candidates)
        Per-(agent, candidate) credits allocated. Only present in QV.

    Legacy (deprecated — will be removed in v0.2)
    ---------------------------------------------
    votes : ArrayLike
        For plurality: per-agent chosen candidate index (n_agents,).
        For quadratic: per-candidate vote totals (n_candidates,).
        Prefer ``votes_matrix`` or ``votes_per_candidate``.
    qv_votes_matrix : ArrayLike
        Alias of ``votes_matrix`` for QV. Prefer ``votes_matrix``.
    """

    # Always present
    votes_matrix: ArrayLike
    votes_per_candidate: ArrayLike
    winner: ArrayLike
    softmax: ArrayLike
    candidate_utilities: ArrayLike

    # QV-specific
    credits_spent: ArrayLike

    # Legacy
    votes: ArrayLike
    qv_votes_matrix: ArrayLike
