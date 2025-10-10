from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from eci.agents import Voter

if TYPE_CHECKING:
    from environment import Environment


class DataCollector:
    """Collects and stores data from the simulation."""

    def __init__(self):
        """Initialize the DataCollector with an empty list for records."""
        self.records = []

    def _process_round_results(
        self, results: Dict[int, float], round_num: int
    ) -> Dict[str, Any]:
        """Process scores and proportions for a single round.

        Parameters
        ----------
        results
            A dictionary mapping candidate IDs to their scores.
        round_num
            The round number to which the results apply.

        Returns
        -------
        Dict[str, Any]
            A dictionary with processed scores and proportions, ready to be
            added to the main step record.
        """
        if not results:
            return {}

        processed_data = {}
        processed_data.update(
            {
                f"candidate_{cid}_score_r{round_num}": score
                for cid, score in results.items()
            }
        )
        total_score = sum(results.values())
        if total_score > 0:
            processed_data.update(
                {
                    f"candidate_{cid}_prop_r{round_num}": score / total_score
                    for cid, score in results.items()
                }
            )
        return processed_data

    def collect(self, environment: "Environment") -> None:
        """Record the state of the model.

        Parameters
        ----------
        environment
            The simulation environment instance from which to collect data.
        """
        step_info = {
            "step": environment.scheduler.step_count,
            "winner_id": environment.winner_id,
            "voting_system": environment.voting_system.name,
        }

        if environment.last_round1_results is not None:
            step_info.update(
                self._process_round_results(environment.last_round1_results, 1)
            )
        if environment.last_round2_results is not None:
            step_info.update(
                self._process_round_results(environment.last_round2_results, 2)
            )

        # Create a dictionary to store each agent's vote at this step
        individual_votes: dict[str, str | int | None] = {
            f"voter_{v.id}_vote": (
                ",".join(map(str, v.last_vote))
                if isinstance(v.last_vote, list)
                else v.last_vote
            )
            for v in environment.voters
        }
        step_info.update(individual_votes)

        self.records.append(step_info)

    def get_dataframe(self) -> pd.DataFrame:
        """Export collected data to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row is a simulation step and columns
            represent different collected metrics.
        """
        return pd.DataFrame(self.records).fillna(0)

    def get_long_dataframe(self) -> pd.DataFrame:
        """Export data to a long-format DataFrame.

        Returns
        -------
        pd.DataFrame
            A long-format DataFrame with columns for step, winner, candidate,
            round, score, and proportion.
        """
        wide_df = self.get_dataframe()
        if wide_df.empty:
            return pd.DataFrame()

        id_vars = ["step", "winner_id", "voting_system"]
        # Exclude the new individual vote columns from the melt operation
        value_vars_all = [
            c
            for c in wide_df.columns
            if c not in id_vars and not c.startswith("voter_")
        ]

        score_cols = [col for col in value_vars_all if "_score_r" in col]
        prop_cols = [col for col in value_vars_all if "_prop_r" in col]

        if not score_cols:
            return wide_df  # Return the wide DF with votes if no scores are present

        # Melt scores
        scores_long = wide_df.melt(
            id_vars=id_vars, value_vars=score_cols, var_name="info", value_name="score"
        )
        score_extract = scores_long["info"].str.extract(r"candidate_(\d+)_score_r(\d+)")
        scores_long["candidate_id"] = pd.to_numeric(score_extract[0])
        scores_long["round"] = pd.to_numeric(score_extract[1])

        # Melt proportions
        props_long = wide_df.melt(
            id_vars=id_vars,
            value_vars=prop_cols,
            var_name="info",
            value_name="proportion",
        )
        prop_extract = props_long["info"].str.extract(r"candidate_(\d+)_prop_r(\d+)")
        props_long["candidate_id"] = pd.to_numeric(prop_extract[0])
        props_long["round"] = pd.to_numeric(prop_extract[1])

        # Merge
        merge_cols = id_vars + ["candidate_id", "round"]
        final_df = pd.merge(
            scores_long[merge_cols + ["score"]],
            props_long[merge_cols + ["proportion"]],
            on=merge_cols,
            how="left",
        )

        return final_df.dropna(subset=["candidate_id"]).astype(
            {"candidate_id": int, "round": int}
        )


class SimulationVisualizer:
    """A class dedicated to plotting and visualizing simulation results.

    This class provides a suite of methods to generate plots
    from the data produced by the simulation environment.

    Parameters
    ----------
    environment
        The main simulation environment instance.
    datacollector
        The data collector instance containing the simulation records.

    Attributes
    ----------
    env : Environment
        A reference to the simulation environment.
    collector : DataCollector
        A reference to the data collector.
    input_data : np.ndarray
        A reference to the input data used for the simulation.
    """

    def __init__(self, environment: "Environment", datacollector: "DataCollector"):
        """Initialize the visualizer and sets a custom plot theme."""
        self.env = environment
        self.collector = datacollector
        self.input_data = environment.input_data
        sns.set_theme(
            style="whitegrid",
            rc={
                "axes.facecolor": "#f9f9f9",
                "figure.facecolor": "#f9f9f9",
                "grid.color": "#e0e0e0",
                "grid.linestyle": "--",
                "font.family": "sans-serif",
                "font.sans-serif": "Helvetica",
            },
        )

    def _prepare_preference_df(self, num_voters_to_show: int) -> pd.DataFrame:
        """Prepare the DataFrame for preference distribution plotting.

        Parameters
        ----------
        num_voters_to_show
            The number of individual voter distributions to include in the data.

        Returns
        -------
        pd.DataFrame
            A DataFrame formatted for plotting preference distributions.
        """
        voters = self.env.voters
        candidates = self.env.candidates
        n_preferences = self.env.num_preferences
        x_vals = np.linspace(-3, 4, 400)
        rows = []

        for c in candidates:
            mus, precisions = c.policy["mean"], c.policy["precision"]
            for pref in range(n_preferences):
                pdf = norm.pdf(
                    x_vals, loc=mus[pref], scale=1 / np.sqrt(precisions[pref])
                )
                rows.extend(
                    [
                        {
                            "group": "Candidate",
                            "id": f"C{c.id}",
                            "preference": f"Topic {pref + 1}",
                            "x": x,
                            "pdf": y,
                        }
                        for x, y in zip(x_vals, pdf)
                    ]
                )

        for v in voters[:num_voters_to_show]:
            mus, precisions = v.preferences["mean"], v.preferences["precision"]
            for pref in range(n_preferences):
                pdf = norm.pdf(
                    x_vals, loc=mus[pref], scale=1 / np.sqrt(precisions[pref])
                )
                rows.extend(
                    [
                        {
                            "group": "Voter",
                            "id": f"V{v.id}",
                            "preference": f"Topic {pref + 1}",
                            "x": x,
                            "pdf": y,
                        }
                        for x, y in zip(x_vals, pdf)
                    ]
                )

        return pd.DataFrame(rows)

    def plot_preference_distributions(
        self, num_voters_to_show: int = 10, axes: Optional[np.ndarray] = None
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """Visualizes preference distributions and returns the plot objects.

        Parameters
        ----------
        num_voters_to_show
            The number of individual voter distributions to show (default is 10).
        axes
            An array of matplotlib axes to plot on. If None, new axes are created.

        Returns
        -------
        Optional[Tuple[plt.Figure, np.ndarray]]
            A tuple containing the matplotlib Figure and axes array, or None if
            plotting is not possible.
        """
        df = self._prepare_preference_df(num_voters_to_show)
        if df.empty:
            print("No data available to plot preference distributions.")
            return None, None

        preferences = sorted(df["preference"].unique())
        num_preferences = len(preferences)
        candidate_ids = sorted([c for c in df["id"].unique() if c.startswith("C")])
        palette = sns.color_palette("viridis", n_colors=len(candidate_ids))

        # --- Axes Creation Logic ---
        if axes is None:
            # If no axes are provided, create them based on the number of preferences.
            fig, axes = plt.subplots(
                num_preferences, 1, figsize=(12, 5 * num_preferences), sharex=True
            )
        else:
            # If axes are provided, get the figure and validate the shape.
            if len(axes) < num_preferences:
                raise ValueError(
                    f"Provided axes array is too small. "
                    f"Data requires {num_preferences} subplots, but got {len(axes)}."
                )
            # Assume the first axis's figure is the parent for all.
            fig = axes[0].get_figure()

        # Ensure axes is always an iterable array, even if there's only one subplot.
        axes = np.atleast_1d(axes)

        # --- Plotting Logic ---
        for ax, pref in zip(axes, preferences):
            sub_df = df[df["preference"] == pref]
            for i, cand_id in enumerate(candidate_ids):
                cand_df = sub_df[sub_df["id"] == cand_id]
                ax.fill_between(
                    cand_df["x"],
                    cand_df["pdf"],
                    color=palette[i],
                    alpha=0.8,
                    label=cand_id,
                )

            voter_df = sub_df[sub_df["group"] == "Voter"]
            for i, (voter_id, data) in enumerate(voter_df.groupby("id")):
                label = "Voters" if i == 0 else ""
                ax.fill_between(
                    data["x"], data["pdf"], color="black", alpha=0.03, label=label
                )

            ax.set_title(pref)
            ax.set_ylabel("Density")
            ax.legend(title="Candidate ID", frameon=True)
            ax.set_ylim(bottom=0)

        axes[-1].set_xlabel("Preference Value")
        fig.tight_layout()

        return fig, axes

    def plot_belief_trajectory(
        self,
        voter: "Voter",
        preference_index: int = 0,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    ) -> Optional[Tuple[plt.Figure, plt.Axes, plt.Axes]]:
        """Plot a voter's belief trajectory for a specific preference.

        Parameters
        ----------
        voter
            The Voter agent whose belief trajectory will be plotted.
        preference_index
            The index of the preference topic to plot (default is 0).
        axes
            A tuple of (ax_main, ax_density) to plot on. If None, a new
            figure and axes will be created.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes, plt.Axes]]
            A tuple containing the Figure and two axes (main and density), or
            None if plotting is not possible.
        """
        if not hasattr(voter, "traj") or not voter.traj:
            print(f"Voter {voter.id} has no trajectory data to plot.")
            return None, None, None

        try:
            traj = voter.traj[preference_index]
            mean, std_dev = traj["expected_mean"][voter.id], traj["precision"][voter.id]
            observations = self.input_data[:, preference_index]
        except (KeyError, IndexError, TypeError):
            print(f"Could not parse trajectory data for Voter {voter.id}")
            return None, None, None

        # --- Figure and axes creation logic ---
        if axes is None:
            # If no axes are provided, create the figure and GridSpec layout.
            fig = plt.figure(figsize=(15, 7))
            gs = gridspec.GridSpec(1, 5, figure=fig)
            ax_main = fig.add_subplot(gs[0, 0:4])
            ax_density = fig.add_subplot(gs[0, 4], sharey=ax_main)
        else:
            # If axes are provided, unpack them and get the parent figure.
            ax_main, ax_density = axes
            fig = ax_main.get_figure()

        ax_main.scatter(
            range(len(observations)),
            observations,
            s=10,
            alpha=0.6,
            label="Observations",
            color="gray",
            zorder=2,
        )
        ax_main.plot(
            mean,
            label="Voter's Belief (Mean)",
            color="crimson",
            linewidth=2.5,
            zorder=3,
        )

        upper_bound = mean + 1.96 * std_dev
        lower_bound = mean - 1.96 * std_dev

        ax_main.fill_between(
            range(len(mean)),
            lower_bound,
            upper_bound,
            color="crimson",
            alpha=0.2,
            label="95% Confidence Interval",
            zorder=1,
        )

        ax_main.set_title(f"Belief Trajectory for Voter {voter.id}")
        ax_main.set_xlabel("Time Step")
        ax_main.set_ylabel("Preference Value")
        ax_main.legend()
        ax_main.grid(True, linestyle="--", alpha=0.6)
        voter.preferences

        # --- Horizontal density plot ---
        mean_preference = voter.preferences["mean"][0]
        precision_preference = voter.preferences["precision"][0]

        # Calculate the PDF of the voter's preference
        y_vals = np.linspace(-3, 4, 400)
        y_min = y_vals.min()
        y_max = y_vals.max()
        y_vals_normalized = (y_vals - y_min) / (y_max - y_min)

        pdf = norm.pdf(
            y_vals,
            loc=mean_preference,
            scale=1 / np.sqrt(precision_preference),
        )
        # Normalize PDF for plotting aesthetics
        pdf = pdf / np.max(pdf)

        # Plot the horizontal density
        ax_density.fill_betweenx(
            y=y_vals_normalized,  # Use original y-values for correct positioning
            x1=0,  # Start from x=0
            x2=pdf,  # Density values on the x-axis
            color="lightgrey",
            alpha=0.8,
            edgecolor="grey",
        )

        # Configure axes
        ax_density.set_xlabel("Density")
        ax_density.tick_params(axis="y", labelleft=False)  # Hide y-axis labels
        plt.tight_layout()
        return fig, ax_main, ax_density

    def plot_simulation_results_distribution(
        self, plot_kind: str = "histogram", axes: Optional[np.ndarray] = None
    ) -> Optional[Tuple[plt.Figure, np.ndarray]]:
        """Visualize the distribution of vote proportions for each round.

        Parameters
        ----------
        plot_kind
            The kind of plot to generate, either 'histogram' or 'stripplot'
            (default is 'histogram').
        axes
            A numpy array of matplotlib axes to plot on. If None, new figure
            and axes will be created.

        Returns
        -------
        Optional[Tuple[plt.Figure, np.ndarray]]
            A tuple containing the matplotlib Figure and axes array, or None if
            plotting is not possible.
        """
        results_df = self.collector.get_long_dataframe()

        if results_df.empty or "round" not in results_df.columns:
            print("DataFrame is empty or does not contain round data.")
            return None

        n_sims = results_df["step"].nunique()

        # --- Figure creation logic ---
        if axes is None:
            # If no axes are provided, create a new figure and axes.
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            # Only add the main title if we are creating the figure here.
            fig.suptitle(
                f"Distribution of Vote Proportions Across {n_sims} Steps", fontsize=16
            )
        else:
            # If axes are provided, simply get the parent figure.
            fig = axes[0].get_figure()

        all_candidate_ids = sorted(results_df["candidate_id"].unique())
        palette = sns.color_palette("viridis", n_colors=len(all_candidate_ids))
        color_map = {cid: color for cid, color in zip(all_candidate_ids, palette)}

        df_r1 = results_df[results_df["round"] == 1]
        df_r2 = results_df[results_df["round"] == 2]

        def _plot_single_round(ax, data, round_num, color_map, plot_kind) -> None:
            """Plot results for a single round."""
            ax.set_title(f"Round {round_num} Results")
            if data.empty:
                ax.text(0.5, 0.5, "No data for this round", ha="center", va="center")
                return
            candidate_ids_in_round = sorted(data["candidate_id"].unique())
            if plot_kind == "histogram":
                for cid in candidate_ids_in_round:
                    proportions = data.loc[data["candidate_id"] == cid, "proportion"]
                    ax.hist(
                        proportions.dropna(),
                        bins=25,
                        density=True,
                        alpha=0.7,
                        color=color_map[cid],
                        edgecolor="black",
                        label=f"Candidate {cid}",
                        histtype="stepfilled",
                    )
                ax.set_xlabel("Proportion of Votes")
                ax.set_ylabel("Density")
                ax.legend(frameon=True, title="Candidate ID")
                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0, top=400)  # Optional: set a fixed upper limit
            else:
                sns.stripplot(
                    data=data,
                    x="proportion",
                    y="candidate_id",
                    orient="h",
                    hue="candidate_id",
                    palette=color_map,
                    jitter=True,
                    alpha=0.5,
                    size=4,
                    ax=ax,
                    order=candidate_ids_in_round,
                    legend=False,
                )
                ax.set_ylabel("Candidate ID")
                ax.set_xlabel("Proportion of Votes")
                ax.set_xlim(left=0)

        _plot_single_round(axes[0], df_r1, 1, color_map, plot_kind)
        _plot_single_round(axes[1], df_r2, 2, color_map, plot_kind)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, axes
