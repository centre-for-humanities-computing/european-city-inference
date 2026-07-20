"""Winner distribution plots (single- and multi-dataset)."""

from typing import Any, Mapping, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eci.plots._context import _get_context


def plurality_results_to_share_df(
    results_stacked: Mapping[str, Any],
    n_candidates: int,
) -> pd.DataFrame:
    """Convert vmapped plurality results into long-format vote shares.

    Expects keys ``vote_round_1`` and ``vote_final_round_2`` (per-voter
    indices). Returns one row per ``(candidate, round)`` for each
    simulation.
    """
    first_round_votes = np.asarray(results_stacked["vote_round_1"])
    final_round_votes = np.asarray(results_stacked["vote_final_round_2"])
    voter_count = first_round_votes.shape[1]

    candidate_round_frames = []
    for candidate_index in range(n_candidates):
        candidate_round_frames.append(
            pd.DataFrame(
                {
                    "candidate": f"C{candidate_index}",
                    "share": (first_round_votes == candidate_index).sum(axis=1)
                    / voter_count,
                    "round": "Round 1",
                }
            )
        )
        candidate_round_frames.append(
            pd.DataFrame(
                {
                    "candidate": f"C{candidate_index}",
                    "share": (final_round_votes == candidate_index).sum(axis=1)
                    / voter_count,
                    "round": "Round 2",
                }
            )
        )
    return pd.concat(candidate_round_frames, ignore_index=True)


def _bootstrap_proportion_ci(
    wins: np.ndarray,
    n_candidates: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap CI for win probability per candidate."""
    random_generator = np.random.default_rng(seed)
    simulation_count = len(wins)
    resampled_indices = random_generator.integers(
        0,
        simulation_count,
        size=(n_boot, simulation_count),
    )
    resampled_winners = wins[resampled_indices]
    bootstrap_proportions = np.stack(
        [
            (resampled_winners == candidate_index).mean(axis=1)
            for candidate_index in range(n_candidates)
        ],
        axis=1,
    )
    lower_quantile = (1 - ci) / 2
    upper_quantile = 1 - lower_quantile
    point_estimates = np.array(
        [(wins == candidate_index).mean() for candidate_index in range(n_candidates)]
    )
    lower_bounds = np.quantile(
        bootstrap_proportions,
        lower_quantile,
        axis=0,
    )
    upper_bounds = np.quantile(
        bootstrap_proportions,
        upper_quantile,
        axis=0,
    )
    return point_estimates, lower_bounds, upper_bounds


def plot_winner_distribution(
    winners: np.ndarray,
    n_candidates: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of empirical P(win) per candidate with bootstrap CI."""
    winners = np.asarray(winners)
    if n_candidates is None:
        n_candidates = int(winners.max()) + 1

    point_estimates, lower_bounds, upper_bounds = _bootstrap_proportion_ci(
        winners,
        n_candidates,
        n_boot,
        ci,
    )
    error_ranges = np.stack(
        [
            point_estimates - lower_bounds,
            upper_bounds - point_estimates,
        ]
    )

    with _get_context():
        if ax is None:
            figure, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        else:
            figure = cast(plt.Figure, ax.figure)

        candidate_colors = sns.color_palette(
            "viridis",
            n_colors=n_candidates,
        )
        candidate_positions = np.arange(n_candidates)
        ax.bar(
            candidate_positions,
            point_estimates,
            color=candidate_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            candidate_positions,
            point_estimates,
            yerr=error_ranges,
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.2,
        )
        ax.set_xticks(candidate_positions)
        ax.set_xticklabels(
            [f"C{candidate_index}" for candidate_index in range(n_candidates)]
        )
        ax.set_ylabel("Win probability")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Empirical P(win) over {len(winners)} simulations "
            f"({int(ci * 100)}% bootstrap CI)",
            fontsize=10,
        )
    return figure, ax


def plot_winner_distribution_grouped(
    winners_by_group: Mapping[str, np.ndarray],
    n_candidates: Optional[int] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
    palette: Union[str, Sequence[str]] = "tab10",
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of empirical P(win) overlaying several datasets."""
    winner_arrays = {
        group_name: np.asarray(group_winners)
        for group_name, group_winners in winners_by_group.items()
    }
    if n_candidates is None:
        highest_winner_index = max(
            winner_indices.max() for winner_indices in winner_arrays.values()
        )
        n_candidates = int(highest_winner_index) + 1

    group_names = list(winner_arrays)
    group_count = len(group_names)
    statistics_by_group = {
        group_name: _bootstrap_proportion_ci(
            winner_arrays[group_name],
            n_candidates,
            n_boot,
            ci,
        )
        for group_name in group_names
    }

    with _get_context():
        if ax is None:
            figure, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        else:
            figure = cast(plt.Figure, ax.figure)

        group_colors = sns.color_palette(palette, n_colors=group_count)
        candidate_positions = np.arange(n_candidates)
        bar_width = 0.8 / group_count

        for group_index, group_name in enumerate(group_names):
            point_estimates, lower_bounds, upper_bounds = statistics_by_group[
                group_name
            ]
            error_ranges = np.stack(
                [
                    point_estimates - lower_bounds,
                    upper_bounds - point_estimates,
                ]
            )
            offset = (group_index - (group_count - 1) / 2) * bar_width
            ax.bar(
                candidate_positions + offset,
                point_estimates,
                width=bar_width * 0.95,
                color=group_colors[group_index],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=group_name,
            )
            ax.errorbar(
                candidate_positions + offset,
                point_estimates,
                yerr=error_ranges,
                fmt="none",
                ecolor="black",
                capsize=2,
                lw=0.9,
            )

        ax.set_xticks(candidate_positions)
        ax.set_xticklabels(
            [f"C{candidate_index}" for candidate_index in range(n_candidates)]
        )
        ax.set_ylabel("Win probability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, frameon=True, loc="upper right")
    return figure, ax
