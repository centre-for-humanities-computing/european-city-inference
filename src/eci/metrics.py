import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _winner_satisfaction(candidate_scores: jax.Array, winner: int) -> jax.Array:
    """Compute the winner satisfaction metric.

    Parameters
    ----------
    candidate_scores : jax.Array
        Per-agent score for each candidate.
    winner : int
        Index of the winning candidate.

    Returns
    -------
    jax.Array
        Sum of all agents' scores for the winning candidate.
    """
    return jnp.sum(candidate_scores[:, winner])


def _vote_efficiency(candidate_scores: jax.Array, votes_matrix: jax.Array) -> jax.Array:
    """Compute the vote efficiency metric.

    Parameters
    ----------
    candidate_scores : jax.Array
        Per-agent score for each candidate.
    votes_matrix : jax.Array
        Per-agent votes allocated to each candidate.

    Returns
    -------
    jax.Array
        Summed over agents of each agent's vote-weighted mean candidate
        score (per agent: total weighted score divided by that agent's
        total votes; agents who cast no votes contribute 0).
    """
    weighted_scores = votes_matrix * candidate_scores
    sum_weighted_scores = jnp.sum(weighted_scores, axis=1)
    total_votes = jnp.sum(votes_matrix, axis=1)

    safe_vote_totals = jnp.where(total_votes == 0, 1.0, total_votes)
    efficiency_per_agent = sum_weighted_scores / safe_vote_totals

    return jnp.sum(efficiency_per_agent)


def compute_metrics(
    candidate_scores: jax.Array,
    votes_matrix: jax.Array,
    winner: int,
) -> dict:
    """Compute satisfaction and vote efficiency for one simulation.

    Parameters
    ----------
    candidate_scores : jax.Array
        Per-agent score for each candidate.
    votes_matrix : jax.Array
        Per-agent votes allocated to each candidate.
    winner : int
        Index of the winning candidate.

    Returns
    -------
    dict
        Scalar ``winner_satisfaction`` and ``vote_efficiency`` values.
    """
    winner_satisfaction = _winner_satisfaction(candidate_scores, winner)
    vote_efficiency = _vote_efficiency(candidate_scores, votes_matrix)

    return {
        "winner_satisfaction": winner_satisfaction,
        "vote_efficiency": vote_efficiency,
    }


def _extract_votes_matrix(simulation_results, candidate_count):
    """Extract one vote matrix per simulation, including legacy result shapes."""
    simulation_ids = list(simulation_results)
    first_result = simulation_results[simulation_ids[0]]
    if "votes_matrix" in first_result:
        return jnp.stack(
            [
                simulation_results[simulation_id]["votes_matrix"]
                for simulation_id in simulation_ids
            ]
        )
    # ---- legacy fallback ------------------------------------------------
    if "qv_votes_matrix" in first_result:
        return jnp.stack(
            [
                simulation_results[simulation_id]["qv_votes_matrix"]
                for simulation_id in simulation_ids
            ]
        )
    chosen_candidate_indices = jnp.stack(
        [simulation_results[simulation_id]["votes"] for simulation_id in simulation_ids]
    )
    return jax.nn.one_hot(chosen_candidate_indices, num_classes=candidate_count)


def _extract_candidate_scores(simulation_results):
    """Extract candidate scores, preferring the legacy preference-gap field."""
    simulation_ids = list(simulation_results)
    first_result = simulation_results[simulation_ids[0]]
    score_key = (
        "pref_candidate_gap"
        if "pref_candidate_gap" in first_result
        else "candidate_utilities"
    )
    return jnp.stack(
        [
            simulation_results[simulation_id][score_key]
            for simulation_id in simulation_ids
        ]
    )


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
    simulation_ids = list(sim_results)
    candidate_count = sim_results[simulation_ids[0]]["softmax"].shape[1]
    candidate_scores = _extract_candidate_scores(sim_results)
    votes_matrix = _extract_votes_matrix(sim_results, candidate_count)
    winners = jnp.array(
        [sim_results[simulation_id]["winner"] for simulation_id in simulation_ids],
        dtype=int,
    )
    metrics = jax.vmap(compute_metrics, in_axes=(0, 0, 0))(
        candidate_scores,
        votes_matrix,
        winners,
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame["simulation_id"] = simulation_ids
    return metrics_frame


def winner_frequencies(winners, n_candidates):
    """Empirical P(win) per candidate with the standard error over N sims.

    Parameters
    ----------
    winners : array-like, shape (n_simulations,)
        Winning candidate index for each simulation.
    n_candidates : int

    Returns
    -------
    frequencies : np.ndarray, shape (n_candidates,)
        Empirical win frequency per candidate.
    standard_errors : np.ndarray, shape (n_candidates,)
        Standard error of each frequency, ``sqrt(p(1 - p) / N)`` (the
        binomial-proportion SE over the N simulations — no bootstrap).
    """
    winners = np.asarray(winners)
    simulation_count = winners.shape[0]
    counts = np.bincount(winners.astype(int), minlength=n_candidates)
    frequencies = counts / simulation_count
    standard_errors = np.sqrt(frequencies * (1.0 - frequencies) / simulation_count)
    return frequencies, standard_errors


def uniform_baseline_test(winners, n_candidates):
    """Chi-square goodness-of-fit of the winner distribution against uniform."""
    from scipy.stats import chisquare

    counts = np.bincount(np.asarray(winners).astype(int), minlength=n_candidates)
    expected = np.full(n_candidates, counts.sum() / n_candidates)
    chi_square_statistic, p_value = chisquare(counts, f_exp=expected)
    return float(chi_square_statistic), float(p_value)


def winner_distribution_distance(winners_a, winners_b, n_candidates):
    """Total-variation distance between two systems' winner distributions."""
    frequencies_a = np.bincount(
        np.asarray(winners_a).astype(int), minlength=n_candidates
    ) / len(winners_a)
    frequencies_b = np.bincount(
        np.asarray(winners_b).astype(int), minlength=n_candidates
    ) / len(winners_b)
    return 0.5 * float(np.sum(np.abs(frequencies_a - frequencies_b)))


def winner_agreement(winners_a, winners_b):
    """Fraction of simulations where both systems elect the **same** winner."""
    winner_indices_a = np.asarray(winners_a).astype(int)
    winner_indices_b = np.asarray(winners_b).astype(int)
    if winner_indices_a.shape != winner_indices_b.shape:
        raise ValueError(
            "winner arrays must be aligned per-simulation "
            f"(got shapes {winner_indices_a.shape} and {winner_indices_b.shape})"
        )
    return float(np.mean(winner_indices_a == winner_indices_b))
