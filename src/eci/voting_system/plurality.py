from eci.utils import _find_winner


def _vote_plurality(data, response_function, key, *args, **kwargs) -> dict:
    """Perform plurality voting.

    Parameters
    ----------
    data:
        Agent data dict (beliefs, preferences, candidates).
    response_function:
        Function (data, key) -> (vote, softmax_probs, candidate_preferences, key).
    key:
        A JAX PRNG key (rng) used for seeding random operations.

    Returns
    -------
        vote data.
    """
    votes, softmax, candidate_utilities, key = response_function(data, key)

    # decomment to implement second round voting (top-2)
    winner = _find_winner(votes, candidate_utilities.shape[1])
    # top_two_winners = _find_top_k_winners(vote, candidate_utilities.shape[1], k=2)

    # decomment to implement second round voting (top-2)
    # mask = jnp.isin(jnp.arange(candidate_utilities.shape[1]), top_two_winners)
    # vote_2, _, _, key = response_function(data, key, mask=mask)
    # final_winner = _find_winner(vote_2, candidate_utilities.shape[1])

    return {
        "votes": votes,
        "winner": winner,
        "softmax": softmax,
        "candidate_utilities": candidate_utilities,
    }


# def strategic_vote(data, response_function, key, *args, alpha: float = 1.0, **kwargs):
#     """Perform plurality strategic voting.

#     Parameters
#     ----------
#     data:
#         Agent data dict (beliefs, preferences, candidates).
#     response_function:
#         Function (data, key) -> (vote, softmax_probs, candidate_preferences, key).
#     key:
#         A JAX PRNG key for seeding random operations.
#     alpha:
#         Strength of the strategic adjustment.

#     Returns
#     -------
#         vote data.
#     """
#     # Poll via response_function to get preferences and expected vote shares,
#     # so any custom scorer/sampler chosen by the caller is respected.
#     key, poll_key = jax.random.split(key)
#     _, softmax_probs, candidate_preferences, _ = response_function(data, poll_key)
#     expected_probs = jnp.mean(softmax_probs, axis=0)

#     # Boost viable candidates, penalize hopeless ones.
#     # softmax(prefs + alpha * log(p)) ∝ p^alpha * exp(prefs).
#     eps = 1e-8
#     adjusted_preferences = candidate_preferences

#     # Build a strategic response function that samples from adjusted preferences
#     def strategic_response_function(data, key, mask=None, *args, **kwargs):
#         prefs = adjusted_preferences
#         if mask is not None:
#             prefs = jnp.where(mask, prefs, -jnp.inf)
#         sample_key, next_key = jax.random.split(key)
#         vote, softmax_probs = _sample_choice(sample_key, prefs)
#         return vote, softmax_probs, prefs, next_key

#     # Run the plurality vote with the strategic response function
#     strategic_results = _vote_plurality(
#         data, strategic_response_function, key, *args, **kwargs
#     )

#     return {
#         **strategic_results,
#         "alpha": jnp.float32(alpha),
#     }
