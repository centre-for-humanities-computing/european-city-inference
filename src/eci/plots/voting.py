"""Voting-system comparison plots."""

from typing import Any, Mapping, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eci.plots._context import _get_context


def compute_vote_shares(
    results_stacked: Mapping[str, Any], n_candidates: int
) -> np.ndarray:
    """Per-simulation vote share per candidate, shape (n_sim, n_candidates).

    Reads the uniform ``votes_matrix`` key produced by every voting rule
    since v0.1; falls back to the legacy keys (``qv_votes_matrix`` /
    per-voter ``votes`` indices) for older saved results.
    """
    # Preferred: uniform votes_matrix → just sum over agents and normalise.
    if "votes_matrix" in results_stacked:
        votes_per_candidate = (
            np.asarray(results_stacked["votes_matrix"]).sum(axis=1).astype(float)
        )
    # Legacy: QV gives per-candidate totals directly via "votes".
    elif "qv_votes_matrix" in results_stacked:
        votes_per_candidate = np.asarray(results_stacked["votes"], dtype=float)
    # Legacy: plurality stores per-voter indices in "votes".
    elif "votes" in results_stacked:
        chosen_candidate_indices = np.asarray(results_stacked["votes"])
        votes_per_candidate = np.stack(
            [
                (chosen_candidate_indices == candidate_index).sum(axis=1)
                for candidate_index in range(n_candidates)
            ],
            axis=1,
        ).astype(float)
    else:
        raise KeyError("results_stacked must contain a 'votes' or 'votes_matrix' key")
    vote_totals = votes_per_candidate.sum(axis=1, keepdims=True)
    return votes_per_candidate / np.maximum(vote_totals, 1e-12)


def plot_voting_system_comparison(
    shares_by_system: Mapping[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare mean vote share per candidate across voting systems."""
    system_names = list(shares_by_system)
    vote_shares = {
        system_name: np.asarray(system_shares)
        for system_name, system_shares in shares_by_system.items()
    }
    simulation_count, candidate_count = next(iter(vote_shares.values())).shape
    mean_shares = {
        system_name: system_shares.mean(axis=0)
        for system_name, system_shares in vote_shares.items()
    }
    share_standard_deviations = {
        system_name: system_shares.std(axis=0)
        for system_name, system_shares in vote_shares.items()
    }

    with _get_context():
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        else:
            figure = cast(plt.Figure, ax.figure)

        system_colors = sns.color_palette("viridis", n_colors=len(system_names))
        candidate_positions = np.arange(candidate_count)
        bar_width = 0.8 / len(system_names)

        for system_index, system_name in enumerate(system_names):
            offset = (system_index - (len(system_names) - 1) / 2) * bar_width
            ax.bar(
                candidate_positions + offset,
                mean_shares[system_name],
                width=bar_width * 0.95,
                yerr=share_standard_deviations[system_name],
                color=system_colors[system_index],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                ecolor="black",
                capsize=3,
                label=system_name,
            )

        ax.set_xticks(candidate_positions)
        ax.set_xticklabels(
            [f"C{candidate_index}" for candidate_index in range(candidate_count)]
        )
        ax.set_ylabel("Vote share")
        ax.set_ylim(0, 1)
        ax.set_title(
            "Voting system comparison "
            f"(mean ± 1 std over {simulation_count} simulations)",
            fontsize=11,
        )
        ax.legend(loc="upper right", frameon=True)
    return figure, ax


def plot_voting_metrics(
    combined_df: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot voting metrics (vote_efficiency, winner_satisfaction) per system."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plot_data = combined_df.rename(columns={"voting_system": "System"})
    figure, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    sns.stripplot(
        data=plot_data,
        x="System",
        y="vote_efficiency",
        hue="System",
        palette="viridis",
        alpha=0.6,
        jitter=0.25,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(
        "How well do votes reflect preferences?",
        fontsize=14,
        pad=15,
    )
    axes[0].set_ylabel("Total Weighted Utility")
    axes[0].set_xlabel("")

    sns.stripplot(
        data=plot_data,
        x="System",
        y="winner_satisfaction",
        hue="System",
        palette="viridis",
        alpha=0.6,
        jitter=0.25,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title(
        "Does the winner satisfy the group?",
        fontsize=14,
        pad=15,
    )
    axes[1].set_ylabel("Total Utility of Winner")
    axes[1].set_xlabel("")
    return figure, axes
