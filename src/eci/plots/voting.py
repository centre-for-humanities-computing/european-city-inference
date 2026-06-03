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
        v = np.asarray(results_stacked["votes_matrix"]).sum(axis=1).astype(float)
    # Legacy: QV gives per-candidate totals directly via "votes".
    elif "qv_votes_matrix" in results_stacked:
        v = np.asarray(results_stacked["votes"], dtype=float)
    # Legacy: plurality stores per-voter indices in "votes".
    elif "votes" in results_stacked:
        v_raw = np.asarray(results_stacked["votes"])
        v = np.stack(
            [(v_raw == c).sum(axis=1) for c in range(n_candidates)], axis=1
        ).astype(float)
    else:
        raise KeyError("results_stacked must contain a 'votes' or 'votes_matrix' key")
    return v / np.maximum(v.sum(axis=1, keepdims=True), 1e-12)


def plot_voting_system_comparison(
    shares_by_system: Mapping[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare mean vote share per candidate across voting systems."""
    systems = list(shares_by_system.keys())
    arrays = {s: np.asarray(v) for s, v in shares_by_system.items()}
    n_sim, n_candidates = next(iter(arrays.values())).shape
    means = {s: v.mean(axis=0) for s, v in arrays.items()}
    stds = {s: v.std(axis=0) for s, v in arrays.items()}

    with _get_context():
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        else:
            fig = cast(plt.Figure, ax.figure)

        colors = sns.color_palette("viridis", n_colors=len(systems))
        x = np.arange(n_candidates)
        width = 0.8 / len(systems)

        for i, s in enumerate(systems):
            offset = (i - (len(systems) - 1) / 2) * width
            ax.bar(
                x + offset,
                means[s],
                width=width * 0.95,
                yerr=stds[s],
                color=colors[i],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                ecolor="black",
                capsize=3,
                label=s,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(n_candidates)])
        ax.set_ylabel("Vote share")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Voting system comparison (mean ± 1 std over {n_sim} simulations)",
            fontsize=11,
        )
        ax.legend(loc="upper right", frameon=True)
    return fig, ax


def plot_voting_metrics(combined_df: pd.DataFrame):
    """Plot voting metrics (vote_efficiency, winner_satisfaction) per system."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plot_df = combined_df.rename(columns={"voting_system": "System"})
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    sns.stripplot(
        data=plot_df,
        x="System",
        y="vote_efficiency",
        hue="System",
        palette="viridis",
        alpha=0.6,
        jitter=0.25,
        legend=False,
        ax=ax[0],
    )
    ax[0].set_title("How well do votes reflect preferences?", fontsize=14, pad=15)
    ax[0].set_ylabel("Total Weighted Utility")
    ax[0].set_xlabel("")

    sns.stripplot(
        data=plot_df,
        x="System",
        y="winner_satisfaction",
        hue="System",
        palette="viridis",
        alpha=0.6,
        jitter=0.25,
        legend=False,
        ax=ax[1],
    )
    ax[1].set_title("Does the winner satisfy the group?", fontsize=14, pad=15)
    ax[1].set_ylabel("Total Utility of Winner")
    ax[1].set_xlabel("")
    return fig, ax
