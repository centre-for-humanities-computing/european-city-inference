from typing import Any, ContextManager, Mapping, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import norm

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
    env_data: Any,
    ax_array: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot preference distributions."""
    # Get data
    v_mu, v_pr = (
        np.array(env_data["preferences"]["mean"]),
        np.array(env_data["preferences"]["precision"]),
    )
    c_mu, c_pr = (
        np.array(env_data["candidates"]["mean"]),
        np.array(env_data["candidates"]["precision"]),
    )

    n_dims = v_mu.shape[1]

    # Calculate the x-range
    v_sig, c_sig = 1.0 / np.sqrt(v_pr), 1.0 / np.sqrt(c_pr)
    lows = np.concatenate([(v_mu - 4 * v_sig).ravel(), (c_mu - 4 * c_sig).ravel()])
    highs = np.concatenate([(v_mu + 4 * v_sig).ravel(), (c_mu + 4 * c_sig).ravel()])
    x = np.linspace(lows.min(), highs.max(), 500)

    # Setup the figure and axes for plotting. Plot as many dims as axes were
    # provided (capped by n_dims), or default to min(2, n_dims) when no axes
    # are passed in.
    n_cands = c_mu.shape[0]
    cand_ids = [f"C{i}" for i in range(n_cands)]
    if ax_array is None:
        n_plots = min(2, n_dims)
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(6, 4 * n_plots), sharex=True, constrained_layout=True
        )
        axes = [axes] if n_plots == 1 else list(axes)
    else:
        axes = list(np.atleast_1d(ax_array))
        n_plots = min(len(axes), n_dims)
        fig = axes[0].figure
    dims_to_plot = list(range(n_plots))
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


def plurality_results_to_share_df(
    results_stacked: Mapping[str, Any],
    n_candidates: int,
) -> pd.DataFrame:
    """Convert vmapped plurality results into the long-format DataFrame.

    Parameters
    ----------
    results_stacked : Mapping
        Output of `jax.vmap(_vote_plurality)`. Expects keys
        `vote_round_1` and `vote_final_round_2`, each of shape (n_sim, n_voters).
    n_candidates : int
        Number of candidates (needed for the round-1 share denominator;
        round 2 always has 2 finalists).
    """
    r1 = np.asarray(results_stacked["vote_round_1"])
    r2 = np.asarray(results_stacked["vote_final_round_2"])
    n_sim, n_voters = r1.shape

    rows = []
    for c in range(n_candidates):
        rows.append(
            pd.DataFrame(
                {
                    "candidate": f"C{c}",
                    "share": (r1 == c).sum(axis=1) / n_voters,
                    "round": "Round 1",
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "candidate": f"C{c}",
                    "share": (r2 == c).sum(axis=1) / n_voters,
                    "round": "Round 2",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _bootstrap_proportion_ci(
    wins: np.ndarray,
    n_candidates: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap CI for win probability per candidate."""
    rng = np.random.default_rng(seed)
    n_sim = len(wins)
    idx = rng.integers(0, n_sim, size=(n_boot, n_sim))
    resampled = wins[idx]
    props = np.stack(
        [(resampled == c).mean(axis=1) for c in range(n_candidates)], axis=1
    )
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    point = np.array([(wins == c).mean() for c in range(n_candidates)])
    return point, np.quantile(props, lo, axis=0), np.quantile(props, hi, axis=0)


def plot_winner_distribution(
    winners: np.ndarray,
    n_candidates: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of empirical win probability per candidate with bootstrap CI.

    Parameters
    ----------
    winners : np.ndarray
        1D array of winner indices, shape (n_sim,).
    n_candidates : int, optional
        Number of candidates. Inferred from `winners.max()+1` if omitted.
    """
    winners = np.asarray(winners)
    if n_candidates is None:
        n_candidates = int(winners.max()) + 1

    point, lo, hi = _bootstrap_proportion_ci(winners, n_candidates, n_boot, ci)
    err = np.stack([point - lo, hi - point])

    with _get_context():
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.figure

        colors = sns.color_palette("viridis", n_colors=n_candidates)
        x = np.arange(n_candidates)
        ax.bar(x, point, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, point, yerr=err, fmt="none", ecolor="black", capsize=4, lw=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(n_candidates)])
        ax.set_ylabel("Win probability")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Empirical P(win) over {len(winners)} simulations "
            f"({int(ci * 100)}% bootstrap CI)",
            fontsize=10,
        )
    return fig, ax


def plot_winner_distribution_grouped(
    winners_by_group: Mapping[str, np.ndarray],
    n_candidates: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
    palette: str = "tab10",
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of empirical P(win) overlaying several datasets.

    Parameters
    ----------
    winners_by_group : Mapping[str, np.ndarray]
        Dict mapping group label (e.g. "pl_bl") -> 1D array of winner indices.
    """
    arrays = {g: np.asarray(w) for g, w in winners_by_group.items()}
    if n_candidates is None:
        n_candidates = int(max(a.max() for a in arrays.values())) + 1

    groups = list(arrays.keys())
    n_groups = len(groups)
    stats = {
        g: _bootstrap_proportion_ci(arrays[g], n_candidates, n_boot, ci) for g in groups
    }

    with _get_context():
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        else:
            fig = ax.figure

        colors = sns.color_palette(palette, n_colors=n_groups)
        x = np.arange(n_candidates)
        width = 0.8 / n_groups

        for i, g in enumerate(groups):
            point, lo, hi = stats[g]
            err = np.stack([point - lo, hi - point])
            offset = (i - (n_groups - 1) / 2) * width
            ax.bar(
                x + offset,
                point,
                width=width * 0.95,
                color=colors[i],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=g,
            )
            ax.errorbar(
                x + offset,
                point,
                yerr=err,
                fmt="none",
                ecolor="black",
                capsize=2,
                lw=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(n_candidates)])
        ax.set_ylabel("Win probability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, frameon=True, loc="upper right")
    return fig, ax


def compute_vote_shares(
    results_stacked: Mapping[str, Any], n_candidates: int
) -> np.ndarray:
    """Per-simulation vote share per candidate, shape (n_sim, n_candidates).

    Handles both output shapes:
    - Plurality: `votes` holds per-voter candidate indices,
      shape (n_sim, n_voters); aggregated here into counts per candidate.
    - Quadratic: `votes` is already aggregated per candidate,
      shape (n_sim, n_candidates); marked by the `qv_votes_matrix` key.

    Each returned row sums to 1.
    """
    if "qv_votes_matrix" in results_stacked:
        v = np.asarray(results_stacked["votes"], dtype=float)
    elif "votes" in results_stacked:
        v_raw = np.asarray(results_stacked["votes"])
        v = np.stack(
            [(v_raw == c).sum(axis=1) for c in range(n_candidates)], axis=1
        ).astype(float)
    else:
        raise KeyError("results_stacked must contain a 'votes' key")
    return v / np.maximum(v.sum(axis=1, keepdims=True), 1e-12)


def plot_voting_system_comparison(
    shares_by_system: Mapping[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare mean vote share per candidate across systems.

    Parameters
    ----------
    shares_by_system : Mapping[str, np.ndarray]
        Dict mapping system name -> array of per-simulation vote shares,
        shape (n_sim, n_candidates). Each row should sum to 1.
        Use `compute_vote_shares` to build these from raw vmapped results.
    """
    systems = list(shares_by_system.keys())
    arrays = {s: np.asarray(v) for s, v in shares_by_system.items()}
    n_sim, n_candidates = next(iter(arrays.values())).shape

    means = {s: v.mean(axis=0) for s, v in arrays.items()}
    stds = {s: v.std(axis=0) for s, v in arrays.items()}

    with _get_context():
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        else:
            fig = ax.figure

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
