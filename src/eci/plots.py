from typing import Any, ContextManager, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import norm

from eci.utils import _extract_env_data_vectorized

STYLE = "whitegrid"


def _get_context() -> ContextManager:
    """Enforce consistent styling across all plots."""
    return sns.axes_style(
        STYLE,
        rc={
            "axes.facecolor": "#f9f9f9",
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "font.family": "sans-serif",
        },
    )


def plot_preference(
    env: Any,
    ax_array: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot preference distributions."""
    # Get data
    env_data = _extract_env_data_vectorized(env)
    v_mu, v_pr = (
        np.array(env_data["preferences"]["mean"]),
        np.array(env_data["preferences"]["precision"]),
    )
    c_mu, c_pr = (
        np.array(env_data["candidates"]["mean"]),
        np.array(env_data["candidates"]["precision"]),
    )

    # Determine which dimensions to plot
    n_dims = v_mu.shape[1]
    dims_to_plot = [0, 1] if n_dims >= 2 else list(range(n_dims))

    # Calculate the x-range
    v_sig, c_sig = 1.0 / np.sqrt(v_pr), 1.0 / np.sqrt(c_pr)
    lows = np.concatenate([(v_mu - 4 * v_sig).ravel(), (c_mu - 4 * c_sig).ravel()])
    highs = np.concatenate([(v_mu + 4 * v_sig).ravel(), (c_mu + 4 * c_sig).ravel()])
    x = np.linspace(lows.min(), highs.max(), 500)

    # Setup the figure and axes for plotting
    n_cands, n_plots = c_mu.shape[0], len(dims_to_plot)
    cand_ids = [f"C{i}" for i in range(n_cands)]
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(6, 4), sharex=True, constrained_layout=True
    )
    axes = [axes] if n_plots == 1 and ax_array is None else axes
    colors = sns.color_palette("viridis", n_colors=n_cands)

    # Plot each dimension separately
    for ax_idx, dim_idx in enumerate(dims_to_plot):
        ax = axes[ax_idx]
        dim_v_mu, dim_v_sig = v_mu[:, dim_idx], 1.0 / np.sqrt(v_pr[:, dim_idx])
        dim_c_mu, dim_c_sig = c_mu[:, dim_idx], 1.0 / np.sqrt(c_pr[:, dim_idx])

        # Plot voters' preference distributions
        for mu, sigma in zip(dim_v_mu, dim_v_sig):
            ax.fill_between(
                x,
                norm.pdf(x, loc=mu, scale=sigma),
                color="black",
                alpha=0.3,
                linewidth=0,
            )

        # Plot candidates' preference distributions
        for i, (mu, sigma) in enumerate(zip(dim_c_mu, dim_c_sig)):
            pdf = norm.pdf(x, loc=mu, scale=sigma)
            ax.fill_between(
                x, pdf, color=colors[i], alpha=0.5, label=cand_ids[i], linewidth=0
            )

        # Format the axis with title and grid
        ax.set_title(
            f"Dimension {dim_idx}", loc="center", fontsize=10, fontweight="normal"
        )
        ax.set_yticks([])
        # Ajouter une légende personnalisée
        voter_patch = mpatches.Patch(color="black", alpha=0.3, label="Voters")
        candidates_patch = mpatches.Patch(
            color=sns.color_palette("viridis", n_colors=10)[5],
            alpha=0.5,
            label="Candidates",
        )
        ax.legend(handles=[voter_patch] + [candidates_patch], loc="upper left")
    # Set the x-axis label for the last plot
    axes[-1].set_xlabel("Preference")
    return fig, axes


def plot_vote_shares(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the distribution of vote shares per candidate and round.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns 'candidate', 'share', and 'round'.
    ax : plt.Axes, optional
        Pre-existing axes for the plot.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    """
    try:
        ctx = _get_context()
    except NameError:
        import contextlib

        ctx = contextlib.nullcontext()

    with ctx:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        sns.stripplot(
            data=df,
            x="candidate",
            y="share",
            hue="round",
            dodge=True,  # Separates Round 1 and Round 2 points
            alpha=0.6,  # Transparency
            jitter=True,  # Spreads points to show density
            palette="viridis",
            ax=ax,
        )

        ax.set_title("Proportion of Votes per Candidate and Round")
        ax.set_ylabel("Vote Share")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

        return fig, ax


def plot_belief_trajectory(
    expected_mean: np.ndarray,
    precisions: np.ndarray,
    observations: np.ndarray,
    preference_params: Tuple[float, float],
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plot the belief trajectory along with observation points and side density.

    Parameters
    ----------
    expected_mean : np.ndarray
        Array of mean beliefs over time.
    precisions : np.ndarray
        Array of precisions (inverse variance) over time.
    observations : np.ndarray
        Array of observed values.
    preference_params : Tuple[float, float]
        Tuple of (target_mean, target_precision).
    axes : Tuple[plt.Axes, plt.Axes], optional
        Tuple of (ax_main, ax_density). If None, a new figure is created.
    title_suffix : str, default ""
        Suffix to append to the plot title.
    ylim : Tuple[float, float], optional
        Y-axis limits (min, max).

    Returns
    -------
    Tuple[plt.Figure, plt.Axes, plt.Axes]
        The figure, main axis, and density axis.
    """
    # Prepare data
    time_steps = np.arange(len(observations))
    ci_bound = 1.96 * (1 / np.sqrt(precisions))

    target_mean, target_std = (
        preference_params[0],
        1 / np.sqrt(preference_params[1]),
    )

    # Setup Axes
    if axes is None:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.02)
        ax_main = fig.add_subplot(gs[0, :-1])
        ax_density = fig.add_subplot(gs[0, -1], sharey=ax_main)
    else:
        ax_main, ax_density = axes
        fig = ax_main.figure

    # Plot Main Trajectory (Observations, Mean, Confidence Interval)
    ax_main.scatter(
        time_steps,
        observations,
        s=15,
        c="gray",
        alpha=0.4,
        label="Observations",
    )
    ax_main.plot(expected_mean, c="#D62728", lw=2.5, label="Belief (Mean)")
    ax_main.fill_between(
        time_steps,
        expected_mean - ci_bound,
        expected_mean + ci_bound,
        color="#D62728",
        alpha=0.1,
        label="95% CI",
    )

    # Configure Main Axis
    if ylim:
        ax_main.set_ylim(ylim)
    ax_main.set(title=f"Belief Trajectory {title_suffix}", xlabel="Time Step")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, ls=":", alpha=0.6)

    # Plot Side Density (Target Distribution)
    y_min, y_max = ax_main.get_ylim()
    y_vals = np.linspace(
        min(y_min, target_mean - 4 * target_std),
        max(y_max, target_mean + 4 * target_std),
        500,
    )

    pdf = norm.pdf(y_vals, loc=target_mean, scale=target_std)
    pdf = (pdf / pdf.max() * 0.9) if pdf.max() > 0 else pdf

    ax_density.plot(pdf, y_vals, c="#555555", lw=1, alpha=0.8)
    ax_density.fill_betweenx(y_vals, 0, pdf, color="gray", alpha=0.2)
    ax_density.axhline(target_mean, c="k", ls="--", lw=1, alpha=0.5)

    # Configure Density Axis (share Y with main)
    ax_density.set(xlim=(0, 1))
    ax_density.axis("off")

    return fig, ax_main, ax_density


def plot_voting_metrics(combined_df: pd.DataFrame):
    """Plot voting metrics for different systems."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    plot_df = combined_df.rename(
        columns={
            "vote_efficiency": "vote_efficiency",
            "winner_satisfaction": "winner_satisfaction",
            "voting_system": "System",
        }
    )

    # Wider figure to fit all 6 systems comfortably
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    # vote_efficiency
    sns.stripplot(
        data=plot_df,
        x="System",
        y="vote_efficiency",
        hue="System",
        palette="viridis",
        alpha=0.6,
        jitter=0.25,
        legend=False,  # legend redundant with x-axis
        ax=ax[0],
    )
    ax[0].set_title("How well do votes reflect preferences?", fontsize=14, pad=15)
    ax[0].set_ylabel("Total Weighted Utility")
    ax[0].set_xlabel("")

    # winner_satisfaction
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
