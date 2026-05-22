import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.typing import ArrayLike

from eci.utils import kl_divergence


# TODO: allow the function to use something else than softmax
def _sample_choice(
    key: ArrayLike, preferences: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample a choice using a softmax probabilities over preferences.

    Parameters
    ----------
    key:
        Pseudo-Random Number Generator key for sampling.
    preferences:
        The preferences for each option.

    Returns
    -------
        vote: An array containing the chosen option.
        softmax_probs: The resulting softmax probability distribution.
    """
    # Calculate softmax per preference
    softmax_probs = jax.nn.softmax(preferences, axis=1)

    # Sample a choice
    vote = jax.random.categorical(key, preferences, axis=1)

    return vote, softmax_probs


# TODO: allow to compute the preference in a different way
def _compute_candidate_utilities(
    data: dict,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Evaluate the per-agent utility score for each candidate.

    Parameters
    ----------
    data:
        Agent data dict with `beliefs`, `preferences`, and `candidates`,
        each holding `mean` and `precision` arrays.

    Returns
    -------
    preference_score_per_agent : ArrayLike
        Shape (n_agents, n_candidates). Per-agent score for each candidate.
    pref_candidate_gap : ArrayLike
        Shape (n_agents, n_candidates). KL(candidate || preferences).
    belief_preference_gap : ArrayLike
        Shape (n_agents,). KL(beliefs || preferences).
    """
    # Compute the gap between belief and preference, and candidate and preference
    belief_preference_gap = _get_belief_preference_gap(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
    )

    # Compute the gap between candidate and preference
    pref_candidate_gap = _get_pref_candidate_gap(
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
    )

    # Compute the preference score for each candidate.
    #   (KL(BELIEF || PREF) - KL(POLICY || PREF)) / KL(BELIEF || PREF)
    # Epsilon guards against division by zero when beliefs == preferences.
    eps = 1e-8
    belief_preference_gap_safe = belief_preference_gap[:, jnp.newaxis] + eps
    preference_score_per_agent = (
        belief_preference_gap_safe - pref_candidate_gap
    ) / belief_preference_gap_safe

    # Alternative formulas (kept for reference):
    #   KL(BELIEF || PREF) - KL(POLICY || PREF)
    #   preference_score_per_agent = (
    #       belief_preference_gap[:, jnp.newaxis] - pref_candidate_gap
    #   )
    #
    #   KL(POLICY || PREF) - KL(BELIEF || PREF)
    #   preference_score_per_agent = (
    #       pref_candidate_gap - belief_preference_gap[:, jnp.newaxis]
    #   )
    return preference_score_per_agent, pref_candidate_gap, belief_preference_gap


# TODO: allow to compute the gap in a different way.
def _get_belief_preference_gap(
    beliefs_mean: ArrayLike,
    beliefs_precision: ArrayLike,
    pref_mean: ArrayLike,
    pref_precision: ArrayLike,
) -> jnp.ndarray:
    """Compute the KL divergence between agent beliefs and preferences.

    Parameters
    ----------
    beliefs_mean : ArrayLike
        Mean of agent beliefs.
    beliefs_precision : ArrayLike
        Precision of agent beliefs.
    pref_mean : ArrayLike
        Mean of agent preferences.
    pref_precision : ArrayLike
        Precision of agent preferences.

    Returns
    -------
    belief_preference_gap : jnp.ndarray
        The computed KL divergence summed per agent.
    """
    # Compute KL(Beliefs || Preferences)
    gap_per_preference = kl_divergence(
        beliefs_mean,
        beliefs_precision,
        pref_mean,
        pref_precision,
    )
    return jnp.sum(gap_per_preference, axis=-1)


# TODO: allow to compute the gap in a different way.
def _get_pref_candidate_gap(
    pref_mean: ArrayLike,
    pref_precision: ArrayLike,
    cand_mean: ArrayLike,
    cand_precision: ArrayLike,
) -> jnp.ndarray:
    """Compute the KL divergence between agent preferences and candidate policies.

    Parameters
    ----------
    pref_mean : ArrayLike
        Mean of agent preferences.
    pref_precision : ArrayLike
        Precision of agent preferences.
    cand_mean : ArrayLike
        Mean of candidate policies.
    cand_precision : ArrayLike
        Precision of candidate policies.

    Returns
    -------
    pref_cand_gap : jnp.ndarray
        The computed KL divergence summed per agent.
    """
    # Compute KL(Policy || Preferences)
    gap_per_dim = kl_divergence(
        pref_mean[:, None, :],
        pref_precision[:, None, :],
        cand_mean[None, :, :],
        cand_precision[None, :, :],
    )
    return jnp.sum(gap_per_dim, axis=-1)


def response_function(data, key, mask=None, *args, **kwargs):
    """Given simulated agent data, sample one vote per agent.

    Parameters
    ----------
    data:
        Agent data dict (beliefs, preferences, candidates).
    key:
        JAX PRNG key for sampling.
    mask:
        Optional boolean array of shape (n_candidates,). `True` keeps the
        candidate; `False` excludes it. Used e.g. for round 2 of plurality
        where only the top-2 candidates remain eligible.

    Returns
    -------
    vote : ArrayLike
        Shape (n_agents,). The sampled candidate index per agent.
    softmax_probs : ArrayLike
        Shape (n_agents, n_candidates). Softmax over the (masked) scores.
    candidate_preferences : ArrayLike
        Shape (n_agents, n_candidates). The (masked) score logits.
    next_key:
        A new PRNG key derived from the input key via `jax.random.split`.
    """
    candidate_utilities, _, _ = _compute_candidate_utilities(data)
    if mask is not None:
        candidate_utilities = jnp.where(mask, candidate_utilities, -jnp.inf)
    sample_key, next_key = jax.random.split(key)
    vote, softmax_probs = _sample_choice(sample_key, candidate_utilities)
    return vote, softmax_probs, candidate_utilities, next_key


def response_function_logpdf(data, key, mask=None, *args, **kwargs):
    """Sample one vote per agent using log-likelihood under preferences.

    For each agent i and candidate c, the score is

        sum_d norm.logpdf(cand_mean[c, d], loc=pref_mean[i, d],
                          scale=1/sqrt(pref_precision[i, d]))

    i.e. the log-probability of the candidate's position under the agent's
    Gaussian preference distribution. Higher = candidate is closer to the
    agent's preferences (in precision-weighted terms).

    Parameters
    ----------
    data:
        Agent data dict. Uses `preferences.mean`, `preferences.precision`
        and `candidates.mean`.
    key:
        JAX PRNG key for sampling.
    mask:
        Optional boolean array of shape (n_candidates,). `True` keeps the
        candidate; `False` excludes it.

    Returns
    -------
    vote : ArrayLike
        Shape (n_agents,). Sampled candidate index per agent.
    softmax_probs : ArrayLike
        Shape (n_agents, n_candidates). Softmax over the (masked) scores.
    candidate_utilities : ArrayLike
        Shape (n_agents, n_candidates). The (masked) logpdf scores.
    next_key:
        A fresh PRNG key derived from the input key via `jax.random.split`.
    """
    pref_mean = data["preferences"]["mean"]  # (n_agents, n_dim)
    pref_precision = data["preferences"]["precision"]  # (n_agents, n_dim)
    cand_mean = data["candidates"]["mean"]  # (n_candidates, n_dim)

    pref_scale = 1.0 / jnp.sqrt(pref_precision)

    logpdf_per_dim = norm.logpdf(
        cand_mean[None, :, :],  # (1, n_candidates, n_dim)
        loc=pref_mean[:, None, :],  # (n_agents, 1, n_dim)
        scale=pref_scale[:, None, :],  # (n_agents, 1, n_dim)
    )
    candidate_utilities = jnp.sum(logpdf_per_dim, axis=-1)  # (n_agents, n_candidates)

    if mask is not None:
        candidate_utilities = jnp.where(mask, candidate_utilities, -jnp.inf)
    sample_key, next_key = jax.random.split(key)
    vote, softmax_probs = _sample_choice(sample_key, candidate_utilities)
    return vote, softmax_probs, candidate_utilities, next_key


def response_function_pref(data, key, mask=None, *args, **kwargs):
    """Sample one vote per agent using preference-based utilities.

    Parameters
    ----------
    data:
        Agent data dict (beliefs, preferences, candidates).
    key:
        JAX PRNG key for sampling.
    mask:
        Optional boolean array of shape (n_candidates,). `True` keeps the
        candidate; `False` excludes it. Used e.g. for round 2 of plurality
        where only the top-2 candidates remain eligible.

    Returns
    -------
    vote : ArrayLike
        Shape (n_agents,). The sampled candidate index per agent.
    softmax_probs : ArrayLike
        Shape (n_agents, n_candidates). Softmax over the (masked) scores.
    candidate_preferences : ArrayLike
        Shape (n_agents, n_candidates). The (masked) score logits.
    next_key:
        A fresh PRNG key derived from the input key via `jax.random.split`.
        Callers can reuse it directly without splitting again.
    """
    _, pref_candidate_gap, _ = _compute_candidate_utilities(data)
    # KL(pref || candidate) is non-negative and grows with distance; negate so
    # that softmax favors candidates closer to each agent's preferences.
    candidate_utilities = -pref_candidate_gap
    if mask is not None:
        candidate_utilities = jnp.where(mask, candidate_utilities, -jnp.inf)
    sample_key, next_key = jax.random.split(key)
    vote, softmax_probs = _sample_choice(sample_key, candidate_utilities)
    return vote, softmax_probs, candidate_utilities, next_key
