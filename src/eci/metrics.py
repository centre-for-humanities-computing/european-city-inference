import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _winner_satisfaction(candidate_preferences: jax.Array, winner: int) -> jax.Array:
    """Compute the winner satisfaction metric.

    Parameters
    ----------
    candidate_preferences : dict
        array of preference for each candidate.
    winner : int
        winner of election.

    Returns
    -------
    satisafcation : jnp.ndarray
        sum of preferences of all agents for the winning candidate.
    """
    return jnp.sum(candidate_preferences[:, winner])


def _vote_efficiency(
    candidate_preferences: jax.Array, votes_matrix: jax.Array
) -> jax.Array:
    """Compute the vote efficiency metric.

    Parameters
    ----------
    candidate_preferences : dict
        array of preference for each candidate.
    votes_matrix : int
        matrix of token allocate per candidate.

    Returns
    -------
    vote_efficiency : jnp.ndarray
        Summed over agents of each agent's vote-weighted mean preference
        gap (per agent: total vote-weighted gap divided by that agent's
        total votes; agents who cast no votes contribute 0).

    """
    # Weighted sum of preferences
    weighted_gaps = votes_matrix * candidate_preferences
    sum_weighted_gaps = jnp.sum(weighted_gaps, axis=1)
    total_tokens = jnp.sum(votes_matrix, axis=1)

    # Avoid division by zero
    safe_tokens = jnp.where(total_tokens == 0, 1.0, total_tokens)
    vote_efficiency = sum_weighted_gaps / safe_tokens

    return jnp.sum(vote_efficiency)


def compute_metrics(
    candidate_preferences: jax.Array, votes_matrix: jax.Array, winner: int
) -> dict:
    """Compute the metric for a single simulation.

    Parameters
    ----------
    candidate_preferences : dict
        array of preference for each candidate.
    votes_matrix : int
        matrix of token allocate per candidate.
    winner : int
        winner of election.

    Returns
    -------
    metrics : jnp.ndarray

    """
    # Compute winner satisfaction
    winner_satisfaction = _winner_satisfaction(candidate_preferences, winner)

    # Compute vote efficiency
    vote_efficiency = _vote_efficiency(candidate_preferences, votes_matrix)

    return {
        "winner_satisfaction": jnp.sum(winner_satisfaction),
        "vote_efficiency": jnp.sum(vote_efficiency),
    }


def _extract_votes_matrix(sim_results, n_cand):
    """Pull a vote matrix from sim_results."""
    keys = list(sim_results.keys())
    first = sim_results[keys[0]]
    if "votes_matrix" in first:
        return jnp.stack([sim_results[k]["votes_matrix"] for k in keys])
    # ---- legacy fallback ------------------------------------------------
    if "qv_votes_matrix" in first:
        return jnp.stack([sim_results[k]["qv_votes_matrix"] for k in keys])
    votes_idx = jnp.stack([sim_results[k]["votes"] for k in keys])
    return jax.nn.one_hot(votes_idx, num_classes=n_cand)


def _extract_preference_gap(sim_results):
    """Pull preference gap, falling back to candidate_utilities."""
    keys = list(sim_results.keys())
    first = sim_results[keys[0]]
    pref_key = (
        "pref_candidate_gap" if "pref_candidate_gap" in first else "candidate_utilities"
    )
    return jnp.stack([sim_results[k][pref_key] for k in keys])


def batch_compute_metrics(sim_results):
    """Compute per-simulation metrics across a batch of simulations.

    Parameters
    ----------
    sim_results : dict
        Maps simulation id to a result dict (vote matrix, ``winner``,
        ``softmax`` and preference-gap keys) as returned by the voting rules.

    Returns
    -------
    pandas.DataFrame
        One row per simulation with ``winner_satisfaction``,
        ``vote_efficiency`` and a ``simulation_id`` column.
    """
    keys = list(sim_results.keys())
    n_cand = sim_results[keys[0]]["softmax"].shape[1]
    pref_gap = _extract_preference_gap(sim_results)
    votes_matrix = _extract_votes_matrix(sim_results, n_cand)
    winners = jnp.array([sim_results[k]["winner"] for k in keys], dtype=int)
    metrics = jax.vmap(compute_metrics, in_axes=(0, 0, 0))(
        pref_gap, votes_matrix, winners
    )
    df = pd.DataFrame(metrics)
    df["simulation_id"] = keys
    return df


def winner_frequencies(winners, n_candidates):
    """Empirical P(win) per candidate with the standard error over N sims.

    Parameters
    ----------
    winners : array-like, shape (n_sim,)
        Winning candidate index for each simulation.
    n_candidates : int

    Returns
    -------
    p : np.ndarray, shape (n_candidates,)
        Empirical win frequency per candidate.
    se : np.ndarray, shape (n_candidates,)
        Standard error of each frequency, ``sqrt(p(1 - p) / N)`` (the
        binomial-proportion SE over the N simulations — no bootstrap).
    """
    winners = np.asarray(winners)
    n = winners.shape[0]
    counts = np.bincount(winners.astype(int), minlength=n_candidates)
    p = counts / n
    se = np.sqrt(p * (1.0 - p) / n)
    return p, se


def uniform_baseline_test(winners, n_candidates):
    """Chi-square goodness-of-fit of the winner distribution against uniform."""
    from scipy.stats import chisquare

    counts = np.bincount(np.asarray(winners).astype(int), minlength=n_candidates)
    expected = np.full(n_candidates, counts.sum() / n_candidates)
    chi2, p_value = chisquare(counts, f_exp=expected)
    return float(chi2), float(p_value)


def winner_distribution_distance(winners_a, winners_b, n_candidates):
    """Total-variation distance between two systems' winner distributions."""
    pa = np.bincount(np.asarray(winners_a).astype(int), minlength=n_candidates) / len(
        winners_a
    )
    pb = np.bincount(np.asarray(winners_b).astype(int), minlength=n_candidates) / len(
        winners_b
    )
    return 0.5 * float(np.sum(np.abs(pa - pb)))


def winner_agreement(winners_a, winners_b):
    """Fraction of simulations where both systems elect the **same** winner."""
    a = np.asarray(winners_a).astype(int)
    b = np.asarray(winners_b).astype(int)
    if a.shape != b.shape:
        raise ValueError(
            "winner arrays must be aligned per-simulation "
            f"(got shapes {a.shape} and {b.shape})"
        )
    return float(np.mean(a == b))
