from typing import Any, ContextManager, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


class SimulationVisualizer:
    """A class dedicated to graphical rendering."""

    def __init__(self, style: str = "whitegrid") -> None:
        self.style = style

    def _get_context(self) -> ContextManager:
        """Get the seaborn context."""
        return sns.axes_style(
            self.style,
            rc={
                "axes.facecolor": "#f9f9f9",
                "grid.color": "#e0e0e0",
                "grid.linestyle": "--",
            },
        )

    # TODO: ADD BELIEF OF AGENT IN THE PLOT
    def plot_preference_distributions(
        self, data: pd.DataFrame, axes: Optional[np.ndarray] = None
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """
        Plot the preference distributions.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing preference data.
        axes : np.ndarray, optional
            Array of axes to plot on. If None, a new figure is created.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray] or None
            The figure and axes objects, or None if data is empty.
        """
        if data.empty:
            return None

        with self._get_context():
            prefs = sorted(data["preference"].unique())
            cand_ids = sorted(c for c in data["id"].unique() if str(c).startswith("C"))
            palette = sns.color_palette("viridis", n_colors=len(cand_ids))

            if axes is None:
                fig, axes = plt.subplots(
                    len(prefs), 1, figsize=(12, 5 * len(prefs)), sharex=True
                )
            else:
                axes = np.atleast_1d(axes)
                fig = axes[0].figure

            for ax, pref in zip(axes, prefs):
                sub_df = data[data["preference"] == pref]

                # Plot Candidates
                for i, cid in enumerate(cand_ids):
                    cdf = sub_df[sub_df["id"] == cid]
                    ax.fill_between(
                        cdf["x"], cdf["pdf"], color=palette[i], alpha=0.8, label=cid
                    )

                # Plot Voters
                if not (v_df := sub_df[sub_df["group"] == "Voter"]).empty:
                    for i, (_, v_data) in enumerate(v_df.groupby("id")):
                        ax.fill_between(
                            v_data["x"],
                            v_data["pdf"],
                            color="black",
                            alpha=0.03,
                            label="Voters" if i == 0 else "",
                        )
                ax.set(title=pref)
                ax.legend(loc="upper right")

            fig.tight_layout()
            return fig, axes

    def plot_belief_trajectory(
        self,
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

        with self._get_context():
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

    def plot_vote_proportions(
        self,
        vote_counts: List[Dict[str, Any]],
        plot_kind: str = "histogram",
        axes: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """
        Plot the vote proportions for each candidate across rounds.

        Parameters
        ----------
        vote_counts : List[Dict[str, Any]]
            List of dictionaries containing vote data.
        plot_kind : str, default "histogram"
            Type of plot ("histogram" or "stripplot").
        axes : np.ndarray, optional
            Array of axes to plot on.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray] or None
            The figure and axes objects, or None if no data.
        """
        if not vote_counts:
            return None

        # 1. Data Preparation
        df = pd.DataFrame(vote_counts).sort_values("candidate_id")
        df["candidate_id"] = df["candidate_id"].astype(str)
        rounds = sorted(df["round"].unique())
        n_rounds = len(rounds)

        with self._get_context():
            # 2. Figure Creation
            if axes is None:
                fig, axes = plt.subplots(
                    1, n_rounds, figsize=(6 * n_rounds, 6), sharey=True, sharex=True
                )
            else:
                axes = np.atleast_1d(axes)
                fig = axes[0].figure

            palette = dict(
                zip(
                    (cands := df["candidate_id"].unique()),
                    sns.color_palette("tab10", len(cands)),
                )
            )

            # 3. Plot Loop
            for i, (ax, r_num) in enumerate(zip(axes, rounds)):
                r_data, is_last = df[df["round"] == r_num], i == n_rounds - 1

                if plot_kind == "stripplot":
                    sns.stripplot(
                        data=r_data,
                        x="proportion",
                        y="candidate_id",
                        hue="candidate_id",
                        palette=palette,
                        orient="h",
                        size=6,
                        alpha=0.8,
                        ax=ax,
                        legend=False,
                    )
                    ax.grid(axis="x", ls="--", alpha=0.5)
                else:
                    sns.histplot(
                        data=r_data,
                        x="proportion",
                        hue="candidate_id",
                        palette=palette,
                        bins=20,
                        stat="density",
                        element="step",
                        fill=True,
                        alpha=0.3,
                        common_norm=False,
                        ax=ax,
                        legend=is_last,
                    )

                ax.set(
                    title=f"Round {r_num}",
                    xlim=(0, 1.05),
                    xlabel="Vote Share",
                    ylabel="Density" if i == 0 and plot_kind != "stripplot" else "",
                )
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

                if plot_kind != "stripplot" and is_last:
                    sns.move_legend(ax, "upper right", title="Candidates")

            fig.suptitle(
                "Vote Distribution Analysis per Round", fontsize=16, fontweight="bold"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            return fig, axes
