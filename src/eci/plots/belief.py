"""Belief trajectory plots (single voter)."""

from typing import Any, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
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


def plot_belief_vote_evolution(
    expected_mean,
    expected_precision,
    observations,
    preference_params,
    plurality_matrix,
    quadratic_matrix,
    candidate_labels=None,
    shock_t=None,
    title="Belief trajectory and vote evolution",
    plurality_label="Plurality\nP(vote)",
    quadratic_label="Quadratic\nvote share",
    plurality_cbar="softmax prob",
    quadratic_cbar="avg vote share",
    vmax=1.0,
    cmap="magma",
    figsize=(13, 7.5),
):
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
    expected_mean = np.asarray(expected_mean).squeeze()
    plurality_matrix = np.asarray(plurality_matrix)
    quadratic_matrix = np.asarray(quadratic_matrix)
    n_cand, n_steps = plurality_matrix.shape
    if candidate_labels is None:
        candidate_labels = [f"C{i}" for i in range(n_cand)]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[2.4, 1, 1],
        width_ratios=[8, 1],
        wspace=0.03,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_den = fig.add_subplot(gs[0, 1], sharey=ax_b)
    ax_pl = fig.add_subplot(gs[1, 0], sharex=ax_b)
    cax_pl = fig.add_subplot(gs[1, 1])
    ax_qv = fig.add_subplot(gs[2, 0], sharex=ax_b)
    cax_qv = fig.add_subplot(gs[2, 1])

    # --- Top: belief trajectory + side density ---------------------------
    plot_belief_trajectory(
        expected_mean=expected_mean,
        expected_precision=np.asarray(expected_precision).squeeze(),
        observations=np.asarray(observations).squeeze(),
        preference_params=preference_params,
        axes=(ax_b, ax_den),
    )
    if shock_t is not None:
        ax_b.axvline(shock_t, color="#444", ls="--", lw=1, alpha=0.45)
    # Replace the helper's centred "Belief Trajectory" title with our own.
    ax_b.set_title("")
    ax_b.set_title(title, loc="left", fontweight="bold", pad=10)
    ax_b.set_xlabel("")

    # --- Two vote-distribution heatmaps (shared time axis & colour scale) -
    extent = [0, n_steps, -0.5, n_cand - 0.5]
    for ax, cax, mat, ylab, cbar in [
        (ax_pl, cax_pl, plurality_matrix, plurality_label, plurality_cbar),
        (ax_qv, cax_qv, quadratic_matrix, quadratic_label, quadratic_cbar),
    ]:
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_yticks(range(n_cand))
        ax.set_yticklabels(candidate_labels)
        ax.set_ylabel(ylab)
        if shock_t is not None:
            ax.axvline(shock_t, color="cyan", ls="--", lw=0.8, alpha=0.6)
        fig.colorbar(im, cax=cax, label=cbar)
    ax_qv.set_xlabel("Time step")

    return fig, (ax_b, ax_den, ax_pl, ax_qv)


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
