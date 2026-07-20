"""Belief trajectory plots (single voter)."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from numpy.typing import ArrayLike
from scipy.stats import norm


def plot_belief_trajectory(
    expected_mean: np.ndarray,
    expected_precision: np.ndarray,
    observations: np.ndarray,
    preference_params: Tuple[float, float],
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot a single voter's belief trajectory + side density panel."""
    time_steps = np.arange(len(observations))
    ci_bound = 1.96 * (1 / np.sqrt(expected_precision))
    target_mean, target_std = (
        preference_params[0],
        1 / np.sqrt(preference_params[1]),
    )

    if axes is None:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.02)
        ax_main = fig.add_subplot(gs[0, :-1])
        ax_density = fig.add_subplot(gs[0, -1], sharey=ax_main)
    else:
        ax_main, ax_density = axes
        fig = cast(plt.Figure, ax_main.figure)

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

    if ylim:
        ax_main.set_ylim(ylim)
    ax_main.set(title=f"Belief Trajectory {title_suffix}".strip(), xlabel="Time Step")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, ls=":", alpha=0.6)

    y_min, y_max = ax_main.get_ylim()
    y_vals = np.linspace(
        min(y_min, target_mean - 4 * target_std),
        max(y_max, target_mean + 4 * target_std),
        500,
    )

    pref_pdf = norm.pdf(y_vals, loc=target_mean, scale=target_std)
    belief_std_final = 1.0 / np.sqrt(expected_precision[-1])
    belief_pdf = norm.pdf(y_vals, loc=expected_mean[-1], scale=belief_std_final)

    peak = max(pref_pdf.max(), belief_pdf.max())
    if peak > 0:
        pref_pdf = pref_pdf / peak * 0.9
        belief_pdf = belief_pdf / peak * 0.9

    ax_density.fill_betweenx(y_vals, 0, pref_pdf, color="gray", alpha=0.2)
    ax_density.plot(pref_pdf, y_vals, c="#555555", lw=1, alpha=0.8, label="Preference")
    ax_density.axhline(target_mean, c="k", ls="--", lw=1, alpha=0.5)
    ax_density.fill_betweenx(y_vals, 0, belief_pdf, color="#D62728", alpha=0.15)
    ax_density.plot(belief_pdf, y_vals, c="#D62728", lw=1.5, alpha=0.9, label="Belief")
    ax_density.set(xlim=(0, 1))
    ax_density.legend(loc="upper right", fontsize=8, frameon=False)
    ax_density.axis("off")

    return fig, ax_main, ax_density


@dataclass(frozen=True)
class _BeliefVoteAxes:
    """Axes used by the combined belief and vote-evolution figure."""

    belief: plt.Axes
    density: plt.Axes
    plurality: plt.Axes
    plurality_colorbar: plt.Axes
    quadratic: plt.Axes
    quadratic_colorbar: plt.Axes


@dataclass(frozen=True)
class _VoteHeatmap:
    """Data and labels for one vote-evolution heatmap."""

    matrix: np.ndarray
    axis_label: str
    colorbar_label: str


@dataclass(frozen=True)
class _VoteHeatmapStyle:
    """Shared presentation settings for vote-evolution heatmaps."""

    candidate_labels: Sequence[str]
    shock_timestep: Optional[int]
    color_map: str
    maximum_value: float


def _create_belief_vote_axes(
    figure_size: Tuple[float, float],
) -> tuple[plt.Figure, _BeliefVoteAxes]:
    """Create the axes for a belief trajectory and two vote heatmaps."""
    figure = plt.figure(figsize=figure_size, constrained_layout=True)
    grid = gridspec.GridSpec(
        3,
        2,
        figure=figure,
        height_ratios=[2.4, 1, 1],
        width_ratios=[8, 1],
        wspace=0.03,
    )
    belief_axis = figure.add_subplot(grid[0, 0])
    return figure, _BeliefVoteAxes(
        belief=belief_axis,
        density=figure.add_subplot(grid[0, 1], sharey=belief_axis),
        plurality=figure.add_subplot(grid[1, 0], sharex=belief_axis),
        plurality_colorbar=figure.add_subplot(grid[1, 1]),
        quadratic=figure.add_subplot(grid[2, 0], sharex=belief_axis),
        quadratic_colorbar=figure.add_subplot(grid[2, 1]),
    )


def _plot_vote_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    colorbar_axis: plt.Axes,
    heatmap: _VoteHeatmap,
    style: _VoteHeatmapStyle,
) -> None:
    """Draw one candidate-by-timestep vote heatmap."""
    candidate_count, timestep_count = heatmap.matrix.shape
    image = axis.imshow(
        heatmap.matrix,
        aspect="auto",
        cmap=style.color_map,
        vmin=0,
        vmax=style.maximum_value,
        extent=[0, timestep_count, -0.5, candidate_count - 0.5],
        origin="lower",
        interpolation="nearest",
    )
    axis.set_yticks(range(candidate_count))
    axis.set_yticklabels(style.candidate_labels)
    axis.set_ylabel(heatmap.axis_label)
    if style.shock_timestep is not None:
        axis.axvline(
            style.shock_timestep,
            color="cyan",
            ls="--",
            lw=0.8,
            alpha=0.6,
        )
    figure.colorbar(image, cax=colorbar_axis, label=heatmap.colorbar_label)


def plot_belief_vote_evolution(
    expected_mean: ArrayLike,
    expected_precision: ArrayLike,
    observations: ArrayLike,
    preference_params: Tuple[float, float],
    plurality_matrix: ArrayLike,
    quadratic_matrix: ArrayLike,
    candidate_labels: Optional[Sequence[str]] = None,
    shock_t: Optional[int] = None,
    title: str = "Belief trajectory and vote evolution",
    plurality_label: str = "Plurality\nP(vote)",
    quadratic_label: str = "Quadratic\nvote share",
    plurality_cbar: str = "softmax prob",
    quadratic_cbar: str = "avg vote share",
    vmax: float = 1.0,
    cmap: str = "magma",
    figsize: Tuple[float, float] = (13, 7.5),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Stacked figure: a belief trajectory above two vote-distribution heatmaps.

    The top panel reuses :func:`plot_belief_trajectory` (belief mean ± 95% CI
    over observations, with a side density comparing belief and preference).
    The two lower panels are per-candidate, per-timestep vote-distribution
    heatmaps (e.g. plurality softmax probability and quadratic vote share),
    sharing the time axis with the belief panel.

    Works for a single voter (tutorial 5: per-timestep softmax / QV share) or a
    population (tutorial 3: mean vote share over the electorate) — it only needs
    two ``(n_candidates, n_steps)`` matrices and one representative belief
    trajectory for the top panel.

    Parameters
    ----------
    expected_mean, expected_precision, observations, preference_params:
        Top-panel belief-trajectory inputs (see :func:`plot_belief_trajectory`).
    plurality_matrix, quadratic_matrix:
        ``(n_candidates, n_steps)`` vote-distribution matrices for the two rules.
        Rows are candidates (already in the order you want displayed), columns
        are timesteps.
    candidate_labels:
        Row labels (default ``["C0", "C1", ...]``).
    shock_t:
        If given, draws a dashed marker at this timestep on every panel.
    title:
        Title of the top panel (left-aligned, bold).
    plurality_label, quadratic_label, plurality_cbar, quadratic_cbar:
        Axis / colorbar labels for the two heatmaps.
    vmax:
        Upper colour limit shared by both heatmaps (default 1.0). Pass
        ``max(plurality_matrix.max(), quadratic_matrix.max())`` to use the full
        dynamic range when shares are well below 1.
    cmap, figsize:
        Heatmap colormap and figure size.

    Returns
    -------
    fig, (ax_belief, ax_density, ax_plurality, ax_quadratic)
    """
    belief_means = np.asarray(expected_mean).squeeze()
    plurality_votes = np.asarray(plurality_matrix)
    quadratic_votes = np.asarray(quadratic_matrix)
    candidate_count = plurality_votes.shape[0]
    if candidate_labels is None:
        candidate_labels = [
            f"C{candidate_index}" for candidate_index in range(candidate_count)
        ]

    figure, axes = _create_belief_vote_axes(figsize)
    plot_belief_trajectory(
        expected_mean=belief_means,
        expected_precision=np.asarray(expected_precision).squeeze(),
        observations=np.asarray(observations).squeeze(),
        preference_params=preference_params,
        axes=(axes.belief, axes.density),
    )
    if shock_t is not None:
        axes.belief.axvline(shock_t, color="#444", ls="--", lw=1, alpha=0.45)
    axes.belief.set_title("")
    axes.belief.set_title(title, loc="left", fontweight="bold", pad=10)
    axes.belief.set_xlabel("")

    heatmap_style = _VoteHeatmapStyle(candidate_labels, shock_t, cmap, vmax)
    _plot_vote_heatmap(
        figure,
        axes.plurality,
        axes.plurality_colorbar,
        _VoteHeatmap(plurality_votes, plurality_label, plurality_cbar),
        heatmap_style,
    )
    _plot_vote_heatmap(
        figure,
        axes.quadratic,
        axes.quadratic_colorbar,
        _VoteHeatmap(quadratic_votes, quadratic_label, quadratic_cbar),
        heatmap_style,
    )
    axes.quadratic.set_xlabel("Time step")

    return figure, (axes.belief, axes.density, axes.plurality, axes.quadratic)


def animate_belief_trajectory(
    expected_mean: np.ndarray,
    expected_precision: np.ndarray,
    observations: np.ndarray,
    preference_params: Tuple[float, float],
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    interval: int = 100,
    figsize: Tuple[float, float] = (8, 4),
) -> FuncAnimation:
    """Animate the belief trajectory over time."""
    expected_mean = np.asarray(expected_mean)
    expected_precision = np.asarray(expected_precision)
    observations = np.asarray(observations)

    n_steps = len(observations)
    time_steps = np.arange(n_steps)
    ci_bound = 1.96 * (1.0 / np.sqrt(expected_precision))
    target_mean, target_std = (
        preference_params[0],
        1.0 / np.sqrt(preference_params[1]),
    )
    belief_std = 1.0 / np.sqrt(expected_precision)

    if ylim is None:
        lo = min(
            float(observations.min()),
            float((expected_mean - ci_bound).min()),
            target_mean - 4 * target_std,
        )
        hi = max(
            float(observations.max()),
            float((expected_mean + ci_bound).max()),
            target_mean + 4 * target_std,
        )
        pad = 0.05 * (hi - lo)
        ylim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.02)
    ax_main = fig.add_subplot(gs[0, :-1])
    ax_density = fig.add_subplot(gs[0, -1], sharey=ax_main)
    ax_main.set_xlim(0, max(n_steps - 1, 1))
    ax_main.set_ylim(*ylim)
    ax_main.set(xlabel="Time Step")
    ax_main.grid(True, ls=":", alpha=0.6)
    ax_density.set(xlim=(0, 1))
    ax_density.axis("off")

    y_vals = np.linspace(ylim[0], ylim[1], 500)
    pref_pdf_raw = norm.pdf(y_vals, loc=target_mean, scale=target_std)
    final_belief_pdf_raw = norm.pdf(y_vals, loc=expected_mean[-1], scale=belief_std[-1])
    peak = max(pref_pdf_raw.max(), final_belief_pdf_raw.max())
    pdf_scale = (0.9 / peak) if peak > 0 else 1.0
    pref_pdf = pref_pdf_raw * pdf_scale
    ax_density.fill_betweenx(y_vals, 0, pref_pdf, color="gray", alpha=0.2)
    ax_density.plot(pref_pdf, y_vals, c="#555555", lw=1, alpha=0.8, label="Preference")
    ax_density.axhline(target_mean, c="k", ls="--", lw=1, alpha=0.5)

    obs_scatter = ax_main.scatter(
        [], [], s=15, c="gray", alpha=0.4, label="Observations"
    )
    (mean_line,) = ax_main.plot([], [], c="#D62728", lw=2.5, label="Belief (Mean)")
    ci_fill: dict[str, Any] = {"poly": None}
    belief_fill: dict[str, Any] = {"poly": None}
    (belief_line,) = ax_density.plot(
        [], [], c="#D62728", lw=1.5, alpha=0.9, label="Belief"
    )
    title = ax_main.set_title(f"Belief Trajectory {title_suffix}".strip())
    ax_main.legend(loc="upper left")
    ax_density.legend(loc="upper right", fontsize=8, frameon=False)

    def update(frame: int):
        """Render animation frame ``frame`` (matplotlib ``FuncAnimation`` callback)."""
        k = frame + 1
        obs_scatter.set_offsets(np.c_[time_steps[:k], observations[:k]])
        mean_line.set_data(time_steps[:k], expected_mean[:k])
        if ci_fill["poly"] is not None:
            ci_fill["poly"].remove()
        ci_fill["poly"] = ax_main.fill_between(
            time_steps[:k],
            expected_mean[:k] - ci_bound[:k],
            expected_mean[:k] + ci_bound[:k],
            color="#D62728",
            alpha=0.1,
        )
        cur_pdf = norm.pdf(y_vals, loc=expected_mean[frame], scale=belief_std[frame])
        cur_pdf = cur_pdf / cur_pdf.max() * 0.9 if cur_pdf.max() > 0 else cur_pdf
        belief_line.set_data(cur_pdf, y_vals)
        if belief_fill["poly"] is not None:
            belief_fill["poly"].remove()
        belief_fill["poly"] = ax_density.fill_betweenx(
            y_vals, 0, cur_pdf, color="#D62728", alpha=0.15
        )
        title.set_text(f"Belief Trajectory {title_suffix} — step {frame}".strip())
        return obs_scatter, mean_line, belief_line

    anim = FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=False, repeat=False
    )
    return anim
