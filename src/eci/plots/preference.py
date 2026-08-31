"""Preference / vote-share plots."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from eci.plots._context import _get_context


@dataclass(frozen=True)
class _PreferencePlotData:
    """Arrays needed to draw voter and candidate preference densities."""

    voter_means: np.ndarray
    voter_standard_deviations: np.ndarray
    candidate_means: np.ndarray
    candidate_standard_deviations: np.ndarray
    density_axis_values: np.ndarray

    @property
    def dimension_count(self) -> int:
        """Number of preference dimensions."""
        return self.voter_means.shape[1]

    @property
    def candidate_count(self) -> int:
        """Number of candidate distributions."""
        return self.candidate_means.shape[0]


def _prepare_preference_plot_data(env_data: Any) -> _PreferencePlotData:
    """Convert election distributions into density-plot arrays."""
    voter_means = np.array(env_data["preferences"]["mean"])
    voter_standard_deviations = 1.0 / np.sqrt(
        np.array(env_data["preferences"]["precision"])
    )
    candidate_means = np.array(env_data["candidates"]["mean"])
    candidate_standard_deviations = 1.0 / np.sqrt(
        np.array(env_data["candidates"]["precision"])
    )
    lower_bounds = np.concatenate(
        [
            (voter_means - 4 * voter_standard_deviations).ravel(),
            (candidate_means - 4 * candidate_standard_deviations).ravel(),
        ]
    )
    upper_bounds = np.concatenate(
        [
            (voter_means + 4 * voter_standard_deviations).ravel(),
            (candidate_means + 4 * candidate_standard_deviations).ravel(),
        ]
    )
    return _PreferencePlotData(
        voter_means=voter_means,
        voter_standard_deviations=voter_standard_deviations,
        candidate_means=candidate_means,
        candidate_standard_deviations=candidate_standard_deviations,
        density_axis_values=np.linspace(
            lower_bounds.min(),
            upper_bounds.max(),
            500,
        ),
    )


def _create_preference_axes(
    dimension_count: int,
    existing_axes: Optional[np.ndarray],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create up to two preference axes or reuse axes supplied by the caller."""
    if existing_axes is not None:
        axes = list(np.atleast_1d(existing_axes))
        return cast(plt.Figure, axes[0].figure), axes

    plot_count = min(2, dimension_count)
    figure, created_axes = plt.subplots(
        plot_count,
        1,
        figsize=(6, 4 * plot_count),
        sharex=True,
        constrained_layout=True,
    )
    axes = [created_axes] if plot_count == 1 else list(created_axes)
    return figure, axes


def _plot_preference_dimension(
    axis: plt.Axes,
    plot_data: _PreferencePlotData,
    dimension_index: int,
    candidate_colors: Sequence[Any],
) -> None:
    """Draw all voter and candidate densities for one preference dimension."""
    for distribution_mean, standard_deviation in zip(
        plot_data.voter_means[:, dimension_index],
        plot_data.voter_standard_deviations[:, dimension_index],
    ):
        axis.fill_between(
            plot_data.density_axis_values,
            norm.pdf(
                plot_data.density_axis_values,
                loc=distribution_mean,
                scale=standard_deviation,
            ),
            color="black",
            alpha=0.3,
            linewidth=0,
        )

    candidate_distributions = zip(
        plot_data.candidate_means[:, dimension_index],
        plot_data.candidate_standard_deviations[:, dimension_index],
    )
    for candidate_index, (distribution_mean, standard_deviation) in enumerate(
        candidate_distributions
    ):
        candidate_density = norm.pdf(
            plot_data.density_axis_values,
            loc=distribution_mean,
            scale=standard_deviation,
        )
        axis.fill_between(
            plot_data.density_axis_values,
            candidate_density,
            color=candidate_colors[candidate_index],
            alpha=0.5,
            label=f"C{candidate_index}",
            linewidth=0,
        )

    axis.set_title(
        f"Dimension {dimension_index}",
        loc="center",
        fontsize=10,
        fontweight="normal",
    )
    axis.set_yticks([])
    voter_patch = mpatches.Patch(color="black", alpha=0.3, label="Voters")
    candidates_patch = mpatches.Patch(
        color=sns.color_palette("viridis", n_colors=10)[5],
        alpha=0.5,
        label="Candidates",
    )
    axis.legend(handles=[voter_patch, candidates_patch], loc="upper left")


def plot_preference(
    env_data: Any,
    ax_array: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, list[plt.Axes]]:
    """Plot preference distributions for voters and candidates."""
    plot_data = _prepare_preference_plot_data(env_data)
    figure, axes = _create_preference_axes(plot_data.dimension_count, ax_array)
    candidate_colors = sns.color_palette(
        "viridis",
        n_colors=plot_data.candidate_count,
    )
    axes_to_plot = axes[: plot_data.dimension_count]
    for dimension_index, axis in enumerate(axes_to_plot):
        _plot_preference_dimension(
            axis,
            plot_data,
            dimension_index,
            candidate_colors,
        )
    axes[-1].set_xlabel("Preference")
    return figure, axes


def plot_vote_shares(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot vote-share distributions per candidate × round."""
    try:
        plot_context = _get_context()
    except NameError:
        import contextlib

        plot_context = contextlib.nullcontext()

    with plot_context:
        if ax is None:
            figure, ax = plt.subplots(figsize=(12, 6))
        else:
            figure = cast(plt.Figure, ax.figure)
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
        return figure, ax
