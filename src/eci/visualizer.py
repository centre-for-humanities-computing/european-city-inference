from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


class SimulationVisualizer:
    """A class dedicated purely to graphical rendering."""

    def __init__(self, style: str = "whitegrid"):
        self.style = style

    def _get_context(self):
        """Get the seaborn context with custom rc parameters."""
        return sns.axes_style(
            self.style,
            rc={
                "axes.facecolor": "#f9f9f9",
                "grid.color": "#e0e0e0",
                "grid.linestyle": "--",
            },
        )

    def plot_preference_distributions(
        self, data: pd.DataFrame, axes: Optional[np.ndarray] = None
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """Plot the preference distributions."""
        if data.empty:
            return None, None

        with self._get_context():
            preferences = sorted(data["preference"].unique())
            num_preferences = len(preferences)
            candidate_ids = sorted(
                [c for c in data["id"].unique() if str(c).startswith("C")]
            )
            palette = sns.color_palette("viridis", n_colors=len(candidate_ids))

            # Smart axes creation
            if axes is None:
                fig, axes = plt.subplots(
                    num_preferences, 1, figsize=(12, 5 * num_preferences), sharex=True
                )
            else:
                axes = np.atleast_1d(axes)
                fig = axes[0].get_figure()

            # Plotting
            for ax, pref in zip(axes, preferences):
                sub_df = data[data["preference"] == pref]

                # Plot Candidates
                for i, cand_id in enumerate(candidate_ids):
                    cand_df = sub_df[sub_df["id"] == cand_id]
                    ax.fill_between(
                        cand_df["x"],
                        cand_df["pdf"],
                        color=palette[i],
                        alpha=0.8,
                        label=cand_id,
                    )

                # Plot Voters (Grouped)
                voter_df = sub_df[sub_df["group"] == "Voter"]
                if not voter_df.empty:
                    for i, (v_id, v_data) in enumerate(voter_df.groupby("id")):
                        label = "Voters" if i == 0 else ""
                        ax.fill_between(
                            v_data["x"],
                            v_data["pdf"],
                            color="black",
                            alpha=0.03,
                            label=label,
                        )

                ax.set_title(pref)
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
        """Plot the belief trajectory along with observation points and side density."""
        std_devs = 1 / np.sqrt(precisions)

        with self._get_context():
            if axes is None:
                fig = plt.figure(figsize=(12, 6))
                # Ajustement ratios: 5 parts Traj, 1 part Densité
                gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.02)
                ax_main = fig.add_subplot(gs[0, 0:5])
                ax_density = fig.add_subplot(gs[0, 5], sharey=ax_main)
            else:
                ax_main, ax_density = axes
                fig = ax_main.get_figure()

            # --- 1. Main Trajectory ---
            time_steps = range(len(observations))
            ax_main.scatter(
                time_steps,
                observations,
                s=15,
                color="gray",
                alpha=0.4,
                label="Observations",
            )
            ax_main.plot(means, color="#D62728", linewidth=2.5, label="Belief (Mean)")

            # Intervalle de confiance
            ax_main.fill_between(
                time_steps,
                means - 1.96 * std_devs,
                means + 1.96 * std_devs,
                color="#D62728",
                alpha=0.1,
                label="95% CI",
            )

            # Gestion du zoom Y (essentiel pour que la densité soit belle)
            if ylim:
                ax_main.set_ylim(ylim)

            ax_main.set_title(f"Belief Trajectory {title_suffix}")
            ax_main.legend(loc="upper left", frameon=True)
            ax_main.set_xlabel("Time Step")
            ax_main.grid(True, linestyle=":", alpha=0.6)

            target_mean, target_prec = preference_params
            target_std = 1 / np.sqrt(target_prec)

            if ylim:
                y_min, y_max = ylim
            else:
                curr_ymin, curr_ymax = ax_main.get_ylim()
                y_min, y_max = (
                    min(curr_ymin, target_mean - 4 * target_std),
                    max(curr_ymax, target_mean + 4 * target_std),
                )

            y_vals = np.linspace(y_min, y_max, 500)

            pdf = norm.pdf(y_vals, loc=target_mean, scale=target_std)

            local_max = pdf.max()
            if local_max > 0:
                pdf_norm = (pdf / local_max) * 0.9
            else:
                pdf_norm = pdf

            ax_density.plot(pdf_norm, y_vals, color="#555555", lw=1, alpha=0.8)
            ax_density.fill_betweenx(y_vals, 0, pdf_norm, color="gray", alpha=0.2)

            ax_density.axhline(
                target_mean, color="black", linestyle="--", linewidth=1, alpha=0.5
            )

            ax_density.set_xlim(0, 1.0)
            ax_density.axis("off")
            return fig, ax_main, ax_density

    def plot_vote_proportions(
        self,
        vote_counts: List[Dict[str, Any]],
        plot_kind: str = "histogram",
        axes: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """Plot the vote proportions for each candidate across rounds."""
        if not vote_counts:
            return None

        df = pd.DataFrame(vote_counts)
        n_rounds = df["round"].nunique()

        with self._get_context():
            if axes is None:
                fig, axes = plt.subplots(
                    1, n_rounds, figsize=(9 * n_rounds, 7), sharey=True
                )
                axes = np.atleast_1d(axes)
            else:
                fig = axes[0].get_figure()

            all_cands = sorted(df["candidate_id"].unique())
            palette = dict(zip(all_cands, sns.color_palette("tab10", len(all_cands))))

            for ax, r_num in zip(axes, sorted(df["round"].unique())):
                round_data = df[df["round"] == r_num]

                if plot_kind == "stripplot":
                    sns.stripplot(
                        data=round_data,
                        x="proportion",
                        y="candidate_id",
                        hue="candidate_id",
                        palette=palette,
                        orient="h",
                        ax=ax,
                        legend=False,
                    )
                else:
                    for cid in round_data["candidate_id"].unique():
                        subset = round_data[round_data["candidate_id"] == cid]
                        ax.hist(
                            subset["proportion"],
                            bins=20,
                            density=True,
                            alpha=0.6,
                            color=palette.get(cid),
                            label=f"C{cid}",
                        )
                    ax.legend()

                ax.set_title(f"Round {r_num}")
                ax.set_xlabel("Vote Share")

            return fig, axes
