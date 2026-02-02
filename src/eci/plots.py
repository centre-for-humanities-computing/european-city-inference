from typing import Any, ContextManager, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import norm

from eci.utils import _extract_env_data_vectorized

# Global Style Configuration
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
    dims_to_plot: Optional[List[int]] = None,
    auto_scale: bool = True,
    x_range_manual: Tuple[float, float] = (-100, 100),
    ax_array: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot preference landscape with wider auto-scaling to see full distributions."""
    # extract data
    env_data = _extract_env_data_vectorized(env)  # Check here

    v_mu = np.array(env_data["preferences"]["mean"])  # (N, D)
    v_pr = np.array(env_data["preferences"]["precision"])  # (N, D)
    c_mu = np.array(env_data["candidates"]["mean"])  # (M, D)
    c_pr = np.array(env_data["candidates"]["precision"])  # (M, D)

    # pick dims
    n_dims = v_mu.shape[1]
    if dims_to_plot is None:
        dims_to_plot = [0, 1] if n_dims >= 2 else list(range(n_dims))

    # calc range (wider zoom)
    if auto_scale:
        # calc sigmas
        v_sig = 1.0 / np.sqrt(v_pr)
        c_sig = 1.0 / np.sqrt(c_pr)

        # get bounds (mean +/- 4 stds ensures we see tails)
        lows = np.concatenate([(v_mu - 4 * v_sig).ravel(), (c_mu - 4 * c_sig).ravel()])
        highs = np.concatenate([(v_mu + 4 * v_sig).ravel(), (c_mu + 4 * c_sig).ravel()])

        x = np.linspace(lows.min(), highs.max(), 500)
    else:
        x = np.linspace(x_range_manual[0], x_range_manual[1], 500)

    # setup fig
    n_cands = c_mu.shape[0]
    cand_ids = [f"C{i}" for i in range(n_cands)]
    n_plots = len(dims_to_plot)

    if ax_array is None:
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(10, 4 * n_plots), sharex=True, constrained_layout=True
        )
        if n_plots == 1:
            axes = [axes]
    else:
        axes = np.atleast_1d(ax_array)
        fig = axes[0].figure

    colors = sns.color_palette("viridis", n_colors=n_cands)

    # plot dims
    for ax_idx, dim_idx in enumerate(dims_to_plot):
        ax = axes[ax_idx]

        # plot voters (faint lines)
        dim_v_mu = v_mu[:, dim_idx]
        dim_v_sig = 1.0 / np.sqrt(v_pr[:, dim_idx])

        for mu, sigma in zip(dim_v_mu, dim_v_sig):
            pdf = norm.pdf(x, loc=mu, scale=sigma)
            ax.plot(x, pdf, color="black", alpha=0.02, linewidth=1)

        # plot candidates (filled)
        dim_c_mu = c_mu[:, dim_idx]
        dim_c_sig = 1.0 / np.sqrt(c_pr[:, dim_idx])

        for i, (mu, sigma) in enumerate(zip(dim_c_mu, dim_c_sig)):
            pdf = norm.pdf(x, loc=mu, scale=sigma)
            ax.fill_between(x, pdf, color=colors[i], alpha=0.5, label=cand_ids[i])
            ax.plot(x, pdf, color=colors[i], alpha=1.0, linewidth=1.5)

        # format
        ax.set_title(f"Dimension {dim_idx}", loc="left", fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

        if ax_idx == 0:
            # legend
            v_proxy = plt.Line2D([0], [0], color="black", alpha=0.3, label="Voters")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles + [v_proxy],
                labels=labels + ["Voters"],
                loc="upper right",
            )

    axes[-1].set_xlabel("Preference Spectrum (Belief Space)")
    return fig


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
    # Use context manager if available, otherwise standard plotting
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

        # Adjust legend location if needed
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

        return fig, ax


def plot_belief_trajectory(
    means: np.ndarray,
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
    means : np.ndarray
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
    # 1. Prepare data
    time_steps = np.arange(len(observations))
    ci_bound = 1.96 * (1 / np.sqrt(precisions))

    target_mean, target_std = (
        preference_params[0],
        1 / np.sqrt(preference_params[1]),
    )

    # 2. Setup Axes (Main plot 5/6 width, Density plot 1/6 width)
    if axes is None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.02)
        ax_main = fig.add_subplot(gs[0, :-1])
        ax_density = fig.add_subplot(gs[0, -1], sharey=ax_main)
    else:
        ax_main, ax_density = axes
        fig = ax_main.figure

    # 3. Plot Main Trajectory (Observations, Mean, Confidence Interval)
    ax_main.scatter(
        time_steps,
        observations,
        s=15,
        c="gray",
        alpha=0.4,
        label="Observations",
    )
    ax_main.plot(means, c="#D62728", lw=2.5, label="Belief (Mean)")
    ax_main.fill_between(
        time_steps,
        means - ci_bound,
        means + ci_bound,
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

    # 4. Plot Side Density (Target Distribution)
    # Determine Y-range covering both current view and target distribution
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
    # Set style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Prepare data
    plot_df = combined_df.rename(
        columns={
            "vote_efficiency": "vote_efficiency",
            "winner_satisfaction": "winner_satisfaction",
            "voting_system": "System",
        }
    )

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # vote_efficiency
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
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig, ax
