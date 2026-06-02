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
    """Bar chart of empirical P(win) per candidate with bootstrap CI."""
    winners = np.asarray(winners)
    if n_candidates is None:
        n_candidates = int(winners.max()) + 1

    point, lo, hi = _bootstrap_proportion_ci(winners, n_candidates, n_boot, ci)
    err = np.stack([point - lo, hi - point])

    with _get_context():
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        else:
            fig = cast(plt.Figure, ax.figure)

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
    palette: Union[str, Sequence[str]] = "tab10",
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of empirical P(win) overlaying several datasets."""
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
            fig = cast(plt.Figure, ax.figure)

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
