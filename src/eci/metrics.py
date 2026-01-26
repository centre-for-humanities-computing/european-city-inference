from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.typing import ArrayLike


def compute_decision_metrics(
    agent_utilities: ArrayLike, decision_scores: ArrayLike, selected_option_idx: int
) -> dict:
    """Compute metrics for a simulation."""
    # Calculate sum of utilities for each option across all agents
    total_utility_per_option = jnp.sum(agent_utilities, axis=0)

    # Welfare realized by the selected winner
    realized_welfare = total_utility_per_option[selected_option_idx]

    # Maximum possible welfare
    optimal_welfare = jnp.max(total_utility_per_option)

    # Optimality Gap
    optimality_gap = optimal_welfare - realized_welfare

    # Extract satisfaction of each agent regarding the winner
    agent_satisfaction = agent_utilities[:, selected_option_idx]

    # Compute Gini Coefficient
    sorted_sat = jnp.sort(agent_satisfaction)
    n = agent_satisfaction.shape[0]
    idx = jnp.arange(1, n + 1)

    # Protect against division by zero if total satisfaction is 0
    sum_sat = jnp.sum(sorted_sat)
    safe_sum = jnp.where(sum_sat == 0, 1.0, sum_sat)

    gini = (jnp.sum((2 * idx - n - 1) * sorted_sat)) / (n * safe_sum)

    total_votes = jnp.sum(decision_scores)
    safe_total = jnp.where(total_votes == 0, 1.0, total_votes)

    # Calculate vote shares
    shares = decision_scores / safe_total

    # Compute Shannon Entropy
    shares_clipped = jnp.clip(shares, 1e-10, 1.0)
    entropy = -jnp.sum(shares_clipped * jnp.log(shares_clipped))

    return {
        "social_welfare": realized_welfare,
        "optimality_gap": optimality_gap,
        "is_optimal": optimality_gap < 1e-4,  # True (1.0) if gap is negligible
        "gini_index": gini,
        "mean_satisfaction": jnp.mean(agent_satisfaction),
        "winner_vote_share": shares[selected_option_idx],
        "decision_entropy": entropy,
    }


def batch_compute_metrics(sim_results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Compute metrics for all simulations."""
    # Extract Winners
    winners = jnp.array(
        [res["final_winner"] for res in sim_results.values()], dtype=int
    )

    # Compute Decision Scores
    first_res = next(iter(sim_results.values()))

    if "total_votes_per_candidate" in first_res:
        # Case: Quadratic Voting (Scores are pre-calculated)
        decision_scores = jnp.stack(
            [res["total_votes_per_candidate"] for res in sim_results.values()]
        )
    else:
        # Case: Plurality Voting (Recalculate scores from raw votes)
        votes_raw = jnp.stack(
            [res["vote_final_round_2"] for res in sim_results.values()]
        )
        num_candidates = first_res["softmax_probs_round_1"].shape[1]

        # Vectorized bincount to get (n_sims, n_candidates)
        decision_scores = jax.vmap(lambda x: jnp.bincount(x, length=num_candidates))(
            votes_raw
        )

    # Extract Agent Utilities (Proxy: Softmax Probs from Round 1)
    agent_utilities = jnp.stack(
        [res["softmax_probs_round_1"] for res in sim_results.values()]
    )

    # Vectorized Computation
    batch_fn = jax.vmap(compute_decision_metrics, in_axes=(0, 0, 0))
    metrics_batch = batch_fn(agent_utilities, decision_scores, winners)

    # Pandas Conversion
    df = pd.DataFrame({k: np.array(v).flatten() for k, v in metrics_batch.items()})

    df["simulation_id"] = list(sim_results.keys())

    return df


def compute_vote_shares(
    sim_results: Dict[int, Dict[str, Any]], num_candidates: int
) -> pd.DataFrame:
    """Generate a detailed DataFrame to visualize vote distributions."""
    # Data Extraction
    votes_r1 = np.array([res["vote_round_1"] for res in sim_results.values()])
    votes_r2 = np.array([res["vote_final_round_2"] for res in sim_results.values()])

    n_sims, n_voters = votes_r1.shape
    candidate_names = [f"C{i}" for i in range(num_candidates)]

    # Local Helper for Percentage Calculation
    def get_shares(vote_array):
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=num_candidates), 1, vote_array
        )
        return counts / n_voters

    shares_r1 = get_shares(votes_r1)
    shares_r2 = get_shares(votes_r2)

    # Construct Round 1 DataFrame
    df_r1 = pd.DataFrame(shares_r1, columns=candidate_names)
    df_r1["simulation"] = df_r1.index
    df_r1 = df_r1.melt(id_vars=["simulation"], var_name="candidate", value_name="share")
    df_r1["round"] = "Round 1"

    # Construct Round 2 DataFrame
    df_r2 = pd.DataFrame(shares_r2, columns=candidate_names)
    df_r2["simulation"] = df_r2.index
    df_r2 = df_r2.melt(id_vars=["simulation"], var_name="candidate", value_name="share")
    df_r2["round"] = "Round 2"

    # Merge
    return pd.concat([df_r1, df_r2], ignore_index=True)
