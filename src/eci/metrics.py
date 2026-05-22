from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.typing import ArrayLike


def _winner_satisfaction(candidate_preferences: ArrayLike, winner: int) -> float:
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
    candidate_preferences: np.ndarray, votes_matrix: np.ndarray
) -> float:
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
        average preference gap for the candidates chosen by voters,
        weighted by the number of votes they received.

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
    candidate_preferences: ArrayLike, votes_matrix: ArrayLike, winner: int
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


def batch_compute_metrics(sim_results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Compute metrics for all simulations.

    Parameters
    ----------
    sim_results : dict
        simulations results.

    Returns
    -------
    df : jnp.ndarray

    """
    # Data Extraction
    keys = list(sim_results.keys())
    first_res = sim_results[keys[0]]

    # Number of candidates
    n_cand = first_res["softmax"].shape[1]

    # Get preferences
    pref_key = "pref_candidate_gap"
    if pref_key not in first_res:
        pref_key = "candidate_utilities"
    pref_candidate_gap = jnp.stack([sim_results[k][pref_key] for k in keys])

    # Get winners
    winners = jnp.array([sim_results[k]["winner"] for k in keys], dtype=int)

    if "qv_votes_matrix" in first_res:
        # Quadratic Voting Case
        votes_matrix = jnp.stack([sim_results[k]["qv_votes_matrix"] for k in keys])

    else:
        # Plurality Voting Case
        votes_indices = jnp.stack([sim_results[k]["votes"] for k in keys])

        # Convert to one-hot encoding
        votes_matrix = jax.nn.one_hot(votes_indices, num_classes=n_cand)

    batch_fn = jax.vmap(compute_metrics, in_axes=(0, 0, 0))
    metrics_batch = batch_fn(pref_candidate_gap, votes_matrix, winners)

    # Create DataFrame
    df = pd.DataFrame(metrics_batch)
    df["simulation_id"] = keys

    return df
