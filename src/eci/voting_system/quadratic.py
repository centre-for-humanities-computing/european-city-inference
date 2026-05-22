import jax
import jax.numpy as jnp


# TODO: Allow positive and negative
def _vote_quadratic(
    data, response_function, key, *args, budget: float = 99.0, **kwargs
) -> dict:
    """Perform quadratic voting.

    Parameters
    ----------
    data:
        Agent data dict (beliefs, preferences, candidates).
    response_function:
        Function (data, key) -> (vote, softmax_probs, candidate_preferences, key).
    key:
        A JAX PRNG key (rng) used for seeding random operations.
    budget:
        Token budget for quadratic voting.

    Returns
    -------
        vote data.
    """
    # Sample round 1 preferences.
    _, softmax_probs, candidate_utilities, key = response_function(data, key)

    # Allocation QV
    votes_matrix, credits_spent = _compute_sequential_qv_allocation(
        key, candidate_utilities, budget
    )

    return {
        "votes": jnp.sum(votes_matrix, axis=0),
        "softmax": softmax_probs,
        "winner": jnp.argmax(jnp.sum(votes_matrix, axis=0)),
        "credits_spent": credits_spent,
        "qv_votes_matrix": votes_matrix,
        "candidate_utilities": candidate_utilities,
    }


# TODO: Implement different allocation strategies.
def _compute_sequential_qv_allocation(
    key,
    candidate_utilities,
    budget,
    num_votes: int = 5,
    noise_scale: float = 0.05,
):
    """Allocate QV credits to candidates per agent.

    Each agent picks candidates without replacement with
    probability proportional to softmax(candidate_utilities).

    Parameters
    ----------
    key:
        A JAX PRNG key for seeding random operations.
    candidate_utilities:
        Per-agent candidate utilities that drive the QV allocation.
    budget:
        Token budget for quadratic voting.
    num_votes:
        Number of distinct candidates each agent can vote for (without replacement).
    noise_scale:
        Scale of jitter added to credit weights to avoid deterministic ties.

    Returns
    -------
    votes_matrix:
        Shape (n_agents, n_candidates). Votes allocated per agent per candidate.
    credits_spent:
        Shape (n_agents, n_candidates). Total credits spent per agent per candidate.
    """
    _, num_candidates = candidate_utilities.shape
    num_votes = min(num_votes, num_candidates)
    per_pick_credit = budget / num_votes

    weights = jax.nn.softmax(candidate_utilities, axis=1) * per_pick_credit

    # Gumbel-top-k: top-k of (logits + Gumbel) ≡ sampling k distinct items
    # with prob ∝ softmax(logits).
    gumbel_key, noise_key = jax.random.split(key)
    gumbel = -jnp.log(
        -jnp.log(jax.random.uniform(gumbel_key, candidate_utilities.shape))
    )
    _, top_idx = jax.lax.top_k(candidate_utilities + gumbel, num_votes)
    picks = jnp.sum(jax.nn.one_hot(top_idx, num_candidates), axis=1)

    # Jitter avoids fully-deterministic credits when num_candidates <= num_votes
    # (every candidate is picked once). Clip ≥0 so sqrt stays real.
    noise = jax.random.normal(noise_key, weights.shape) * (
        noise_scale * per_pick_credit
    )
    credits_spent = picks * jnp.maximum(weights + noise, 0.0)
    votes_matrix = jnp.floor(jnp.sqrt(credits_spent)).astype(jnp.int32)
    return votes_matrix, credits_spent


# def strategic_quadratic_vote(
#     data,
#     response_function,
#     key,
#     *args,
#     alpha: float = 1.0,
#     budget: float = 99.0,
#     **kwargs,
# ) -> dict:
#     """Perform quadratic strategic voting.

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
#     budget:
#         Token budget for quadratic voting.

#     Returns
#     -------
#         vote data.
#     """
#     # Poll via response_function to get preferences and expected vote shares.
#     key, poll_key = jax.random.split(key)
#     _, softmax_probs, candidate_preferences, _ = response_function(data, poll_key)
#     expected_probs = jnp.mean(softmax_probs, axis=0)

#     # Boost viable candidates, penalize hopeless ones.
#     # softmax(prefs + alpha * log(p)) ∝ p^alpha * exp(prefs).
#     eps = 1e-8
#     adjusted_preferences = candidate_preferences + jnp.log(expected_probs + eps)

#     # Build a strategic response function that samples from adjusted preferences
#     def strategic_response_function(data, key, mask=None, *args, **kwargs):
#         prefs = adjusted_preferences
#         if mask is not None:
#             prefs = jnp.where(mask, prefs, -jnp.inf)
#         sample_key, next_key = jax.random.split(key)
#         vote, softmax_probs = _sample_choice(sample_key, prefs)
#         return vote, softmax_probs, prefs, next_key

#     # Run the QV vote with the strategic response function
#     strategic_results = _vote_quadratic(
#         data, strategic_response_function, key, *args, budget=budget, **kwargs
#     )

#     return {
#         **strategic_results,
#         "alpha": jnp.float32(alpha),
#     }
