"""Belief trajectory plots (single voter)."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import ArrayLike
from scipy.stats import norm


@dataclass(frozen=True)
class _BeliefDensityData:
    """Preference and final-belief densities for the side panel."""

    axis_values: np.ndarray
    preference: np.ndarray
    final_belief: np.ndarray
    preference_mean: float


def _calculate_belief_density_data(
    main_axis: plt.Axes,
    expected_mean: np.ndarray,
    expected_precision: np.ndarray,
    preference_params: Tuple[float, float],
) -> _BeliefDensityData:
    """Calculate normalized preference and final-belief density curves."""
    preference_mean = preference_params[0]
    preference_standard_deviation = 1 / np.sqrt(preference_params[1])
    visible_minimum, visible_maximum = main_axis.get_ylim()
    density_axis_values = np.linspace(
        min(
            visible_minimum,
            preference_mean - 4 * preference_standard_deviation,
        ),
        max(
            visible_maximum,
            preference_mean + 4 * preference_standard_deviation,
        ),
        500,
    )
    preference_density = norm.pdf(
        density_axis_values,
        loc=preference_mean,
        scale=preference_standard_deviation,
    )
    final_belief_density = norm.pdf(
        density_axis_values,
        loc=expected_mean[-1],
        scale=1.0 / np.sqrt(expected_precision[-1]),
    )
    density_peak = max(
        preference_density.max(),
        final_belief_density.max(),
    )
    if density_peak > 0:
        preference_density = preference_density / density_peak * 0.9
        final_belief_density = final_belief_density / density_peak * 0.9

    return _BeliefDensityData(
        axis_values=density_axis_values,
        preference=preference_density,
        final_belief=final_belief_density,
        preference_mean=preference_mean,
    )


def _plot_belief_density(
    density_axis: plt.Axes,
    density_data: _BeliefDensityData,
) -> None:
    """Draw preference and final-belief densities on the side panel."""
    density_axis.fill_betweenx(
        density_data.axis_values,
        0,
        density_data.preference,
        color="gray",
        alpha=0.2,
    )
    density_axis.plot(
        density_data.preference,
        density_data.axis_values,
        c="#555555",
        lw=1,
        alpha=0.8,
        label="Preference",
    )
    density_axis.axhline(
        density_data.preference_mean,
        c="k",
        ls="--",
        lw=1,
        alpha=0.5,
    )
    density_axis.fill_betweenx(
        density_data.axis_values,
        0,
        density_data.final_belief,
        color="#D62728",
        alpha=0.15,
    )
    density_axis.plot(
        density_data.final_belief,
        density_data.axis_values,
        c="#D62728",
        lw=1.5,
        alpha=0.9,
        label="Belief",
    )
    density_axis.set(xlim=(0, 1))
    density_axis.legend(loc="upper right", fontsize=8, frameon=False)
    density_axis.axis("off")


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
    confidence_bounds = 1.96 * (1 / np.sqrt(expected_precision))

    if axes is None:
        figure = plt.figure(figsize=(8, 4))
        grid = gridspec.GridSpec(1, 6, figure=figure, wspace=0.02)
        main_axis = figure.add_subplot(grid[0, :-1])
        density_axis = figure.add_subplot(grid[0, -1], sharey=main_axis)
    else:
        main_axis, density_axis = axes
        figure = cast(plt.Figure, main_axis.figure)

    main_axis.scatter(
        time_steps,
        observations,
        s=15,
        c="gray",
        alpha=0.4,
        label="Observations",
    )
    main_axis.plot(expected_mean, c="#D62728", lw=2.5, label="Belief (Mean)")
    main_axis.fill_between(
        time_steps,
        expected_mean - confidence_bounds,
        expected_mean + confidence_bounds,
        color="#D62728",
        alpha=0.1,
        label="95% CI",
    )

    if ylim:
        main_axis.set_ylim(ylim)
    main_axis.set(
        title=f"Belief Trajectory {title_suffix}".strip(),
        xlabel="Time Step",
    )
    main_axis.legend(loc="upper left")
    main_axis.grid(True, ls=":", alpha=0.6)

    density_data = _calculate_belief_density_data(
        main_axis,
        expected_mean,
        expected_precision,
        preference_params,
    )
    _plot_belief_density(density_axis, density_data)

    return figure, main_axis, density_axis


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
        extent=(0, timestep_count, -0.5, candidate_count - 0.5),
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


@dataclass(frozen=True)
class _BeliefAnimationData:
    """Numerical series required to render each animation frame."""

    time_steps: np.ndarray
    observations: np.ndarray
    expected_means: np.ndarray
    confidence_bounds: np.ndarray
    belief_standard_deviations: np.ndarray
    density_axis_values: np.ndarray


@dataclass(frozen=True)
class _BeliefAnimationAxes:
    """Main and density axes used by a belief animation."""

    main: plt.Axes
    density: plt.Axes


@dataclass
class _BeliefAnimationArtists:
    """Mutable artists updated as the animation advances."""

    observations: PathCollection
    belief_mean: Line2D
    belief_density: Line2D
    title: Text
    confidence_band: Optional[PolyCollection] = None
    belief_density_fill: Optional[PolyCollection] = None


@dataclass
class _BeliefFrameRenderer:
    """Render one frame of a belief trajectory animation."""

    data: _BeliefAnimationData
    axes: _BeliefAnimationAxes
    artists: _BeliefAnimationArtists
    title_suffix: str

    def __call__(self, frame: int) -> tuple[PathCollection, Line2D, Line2D]:
        visible_steps = frame + 1
        self.artists.observations.set_offsets(
            np.c_[
                self.data.time_steps[:visible_steps],
                self.data.observations[:visible_steps],
            ]
        )
        self.artists.belief_mean.set_data(
            self.data.time_steps[:visible_steps],
            self.data.expected_means[:visible_steps],
        )

        if self.artists.confidence_band is not None:
            self.artists.confidence_band.remove()
        self.artists.confidence_band = self.axes.main.fill_between(
            self.data.time_steps[:visible_steps],
            self.data.expected_means[:visible_steps]
            - self.data.confidence_bounds[:visible_steps],
            self.data.expected_means[:visible_steps]
            + self.data.confidence_bounds[:visible_steps],
            color="#D62728",
            alpha=0.1,
        )

        current_density = norm.pdf(
            self.data.density_axis_values,
            loc=self.data.expected_means[frame],
            scale=self.data.belief_standard_deviations[frame],
        )
        maximum_density = current_density.max()
        if maximum_density > 0:
            current_density = current_density / maximum_density * 0.9
        self.artists.belief_density.set_data(
            current_density,
            self.data.density_axis_values,
        )

        if self.artists.belief_density_fill is not None:
            self.artists.belief_density_fill.remove()
        self.artists.belief_density_fill = self.axes.density.fill_betweenx(
            self.data.density_axis_values,
            0,
            current_density,
            color="#D62728",
            alpha=0.15,
        )
        self.artists.title.set_text(
            f"Belief Trajectory {self.title_suffix} — step {frame}".strip()
        )
        return (
            self.artists.observations,
            self.artists.belief_mean,
            self.artists.belief_density,
        )


def _calculate_animation_limits(
    observations: np.ndarray,
    expected_means: np.ndarray,
    confidence_bounds: np.ndarray,
    preference_params: Tuple[float, float],
) -> Tuple[float, float]:
    """Calculate y-axis limits that contain observations and both distributions."""
    preference_mean = preference_params[0]
    preference_standard_deviation = 1.0 / np.sqrt(preference_params[1])
    lower_bound = min(
        float(observations.min()),
        float((expected_means - confidence_bounds).min()),
        preference_mean - 4 * preference_standard_deviation,
    )
    upper_bound = max(
        float(observations.max()),
        float((expected_means + confidence_bounds).max()),
        preference_mean + 4 * preference_standard_deviation,
    )
    padding = 0.05 * (upper_bound - lower_bound)
    return lower_bound - padding, upper_bound + padding


def _create_animation_axes(
    n_steps: int,
    y_limits: Tuple[float, float],
    figure_size: Tuple[float, float],
) -> tuple[plt.Figure, _BeliefAnimationAxes]:
    """Create and configure the axes for a belief animation."""
    figure = plt.figure(figsize=figure_size)
    grid = gridspec.GridSpec(1, 6, figure=figure, wspace=0.02)
    main_axis = figure.add_subplot(grid[0, :-1])
    density_axis = figure.add_subplot(grid[0, -1], sharey=main_axis)
    main_axis.set_xlim(0, max(n_steps - 1, 1))
    main_axis.set_ylim(*y_limits)
    main_axis.set(xlabel="Time Step")
    main_axis.grid(True, ls=":", alpha=0.6)
    density_axis.set(xlim=(0, 1))
    density_axis.axis("off")
    return figure, _BeliefAnimationAxes(main_axis, density_axis)


def _plot_preference_density(
    density_axis: plt.Axes,
    density_axis_values: np.ndarray,
    preference_params: Tuple[float, float],
    final_belief_mean: float,
    final_belief_standard_deviation: float,
) -> None:
    """Plot the fixed preference density beside the animated belief density."""
    preference_mean = preference_params[0]
    preference_standard_deviation = 1.0 / np.sqrt(preference_params[1])
    preference_density = norm.pdf(
        density_axis_values,
        loc=preference_mean,
        scale=preference_standard_deviation,
    )
    final_belief_density = norm.pdf(
        density_axis_values,
        loc=final_belief_mean,
        scale=final_belief_standard_deviation,
    )
    density_peak = max(preference_density.max(), final_belief_density.max())
    density_scale = (0.9 / density_peak) if density_peak > 0 else 1.0
    scaled_preference_density = preference_density * density_scale
    density_axis.fill_betweenx(
        density_axis_values,
        0,
        scaled_preference_density,
        color="gray",
        alpha=0.2,
    )
    density_axis.plot(
        scaled_preference_density,
        density_axis_values,
        c="#555555",
        lw=1,
        alpha=0.8,
        label="Preference",
    )
    density_axis.axhline(
        preference_mean,
        c="k",
        ls="--",
        lw=1,
        alpha=0.5,
    )


def _initialize_animation_artists(
    axes: _BeliefAnimationAxes,
    title_suffix: str,
) -> _BeliefAnimationArtists:
    """Create the artists that will be updated for every animation frame."""
    observation_points = axes.main.scatter(
        [], [], s=15, c="gray", alpha=0.4, label="Observations"
    )
    (mean_line,) = axes.main.plot([], [], c="#D62728", lw=2.5, label="Belief (Mean)")
    (belief_density_line,) = axes.density.plot(
        [], [], c="#D62728", lw=1.5, alpha=0.9, label="Belief"
    )
    title = axes.main.set_title(f"Belief Trajectory {title_suffix}".strip())
    axes.main.legend(loc="upper left")
    axes.density.legend(loc="upper right", fontsize=8, frameon=False)
    return _BeliefAnimationArtists(
        observations=observation_points,
        belief_mean=mean_line,
        belief_density=belief_density_line,
        title=title,
    )


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

    timestep_count = len(observations)
    time_steps = np.arange(timestep_count)
    confidence_bounds = 1.96 * (1.0 / np.sqrt(expected_precision))
    belief_standard_deviations = 1.0 / np.sqrt(expected_precision)

    if ylim is None:
        ylim = _calculate_animation_limits(
            observations,
            expected_mean,
            confidence_bounds,
            preference_params,
        )

    figure, axes = _create_animation_axes(timestep_count, ylim, figsize)
    density_axis_values = np.linspace(ylim[0], ylim[1], 500)
    _plot_preference_density(
        axes.density,
        density_axis_values,
        preference_params,
        float(expected_mean[-1]),
        float(belief_standard_deviations[-1]),
    )

    animation_data = _BeliefAnimationData(
        time_steps=time_steps,
        observations=observations,
        expected_means=expected_mean,
        confidence_bounds=confidence_bounds,
        belief_standard_deviations=belief_standard_deviations,
        density_axis_values=density_axis_values,
    )
    frame_renderer = _BeliefFrameRenderer(
        data=animation_data,
        axes=axes,
        artists=_initialize_animation_artists(axes, title_suffix),
        title_suffix=title_suffix,
    )
    animation = FuncAnimation(
        figure,
        frame_renderer,
        frames=timestep_count,
        interval=interval,
        blit=False,
        repeat=False,
    )
    return animation
