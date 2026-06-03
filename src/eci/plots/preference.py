"""Preference / vote-share plots."""

from typing import Any, Optional, Tuple, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from eci.plots._context import _get_context


def plot_preference(
    env_data: Any,
    ax_array: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, list]:
    """Plot preference distributions for voters and candidates."""
    v_mu, v_pr = (
        np.array(env_data["preferences"]["mean"]),
        np.array(env_data["preferences"]["precision"]),
    )
    c_mu, c_pr = (
        np.array(env_data["candidates"]["mean"]),
        np.array(env_data["candidates"]["precision"]),
    )

    n_dims = v_mu.shape[1]
    v_sig, c_sig = 1.0 / np.sqrt(v_pr), 1.0 / np.sqrt(c_pr)
    lows = np.concatenate([(v_mu - 4 * v_sig).ravel(), (c_mu - 4 * c_sig).ravel()])
    highs = np.concatenate([(v_mu + 4 * v_sig).ravel(), (c_mu + 4 * c_sig).ravel()])
    x = np.linspace(lows.min(), highs.max(), 500)

    n_cands = c_mu.shape[0]
    cand_ids = [f"C{i}" for i in range(n_cands)]
    if ax_array is None:
        n_plots = min(2, n_dims)
        fig, axes = plt.subplots(
            n_plots,
            1,
            figsize=(6, 4 * n_plots),
            sharex=True,
            constrained_layout=True,
        )
        axes = [axes] if n_plots == 1 else list(axes)
    else:
        axes = list(np.atleast_1d(ax_array))
        n_plots = min(len(axes), n_dims)
        fig = axes[0].figure
    dims_to_plot = list(range(n_plots))
    colors = sns.color_palette("viridis", n_colors=n_cands)

    for ax_idx, dim_idx in enumerate(dims_to_plot):
        ax = axes[ax_idx]
        dim_v_mu, dim_v_sig = v_mu[:, dim_idx], 1.0 / np.sqrt(v_pr[:, dim_idx])
        dim_c_mu, dim_c_sig = c_mu[:, dim_idx], 1.0 / np.sqrt(c_pr[:, dim_idx])

        for mu, sigma in zip(dim_v_mu, dim_v_sig):
            ax.fill_between(
                x,
                norm.pdf(x, loc=mu, scale=sigma),
                color="black",
                alpha=0.3,
                linewidth=0,
            )

        for i, (mu, sigma) in enumerate(zip(dim_c_mu, dim_c_sig)):
            pdf = norm.pdf(x, loc=mu, scale=sigma)
            ax.fill_between(
                x,
                pdf,
                color=colors[i],
                alpha=0.5,
                label=cand_ids[i],
                linewidth=0,
            )

        ax.set_title(
            f"Dimension {dim_idx}", loc="center", fontsize=10, fontweight="normal"
        )
        ax.set_yticks([])
        voter_patch = mpatches.Patch(color="black", alpha=0.3, label="Voters")
        candidates_patch = mpatches.Patch(
            color=sns.color_palette("viridis", n_colors=10)[5],
            alpha=0.5,
            label="Candidates",
        )
        ax.legend(handles=[voter_patch, candidates_patch], loc="upper left")
    axes[-1].set_xlabel("Preference")
    return fig, axes


def plot_vote_shares(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot vote-share distributions per candidate × round."""
    try:
        ctx = _get_context()
    except NameError:
        import contextlib

        ctx = contextlib.nullcontext()

    with ctx:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = cast(plt.Figure, ax.figure)
        sns.stripplot(
            data=df,
            x="candidate",
            y="share",
            hue="round",
            dodge=True,
            alpha=0.6,
            jitter=True,
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Proportion of Votes per Candidate and Round")
        ax.set_ylabel("Vote Share")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
        return fig, ax
