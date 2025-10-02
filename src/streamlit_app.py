import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import streamlit as st

from agents import Voter
from analysis import SimulationVisualizer
from environment import Environment
from voting_systems import (
    PluralityVoting,
    QuadraticVoting,
    QuadraticVotingBudget,
    RankingVoting,
)

# === Sidebar: Parameters ===
st.sidebar.header("Simulation Parameters")
NUM_VOTERS = st.sidebar.number_input("Number of voters", 100, 10000, 100, step=100)
NUM_CANDIDATES = st.sidebar.number_input("Number of candidates", 2, 20, 3, step=1)
NUM_PREFERENCES = st.sidebar.number_input("Number of preferences", 1, 5, 2, step=1)
NUM_SIMULATIONS = st.sidebar.number_input(
    "Number of simulations", 10, 2000, 100, step=10
)


# === Voting systems available ===
def make_systems(use_tom):
    """Create a dictionary of available voting systems."""
    return {
        "Plurality": PluralityVoting(use_theory_of_mind=use_tom),
        "Ranking": RankingVoting(use_theory_of_mind=use_tom),
        "Quadratic": QuadraticVoting(use_theory_of_mind=use_tom),
        "Quadratic Budget": QuadraticVotingBudget(use_theory_of_mind=use_tom),
    }


available_systems_tom = make_systems(True)
available_systems_no_tom = make_systems(False)

# Sidebar: allow user to select systems
selected_systems = st.sidebar.multiselect(
    "Select voting systems to compare",
    options=list(available_systems_tom.keys()),
    default=["Plurality", "Ranking"],  # start with two for faster testing
)

st.title("Agent-Based Voting Simulation Dashboard")

if st.button("Run Simulations"):
    total_steps = (
        NUM_SIMULATIONS * len(selected_systems) * 2
    )  # double car ToM et No ToM
    progress_bar = st.progress(0)
    status_text = st.empty()

    messages = [
        "Generating voters",
        "Running first round",
        "Counting ballots",
        "Running second round",
        "Analyzing results",
    ]

    # Create tabs for each selected system
    tabs = st.tabs(selected_systems)

    def run_and_display(system, use_tom, label):
        """Run the simulation and display results."""
        env = Environment(
            num_voters=NUM_VOTERS,
            num_candidates=NUM_CANDIDATES,
            num_preferences=NUM_PREFERENCES,
            voting_system=system,
            scenario=1,
            rounds=2,
            use_theory_of_mind=use_tom,
        )

        def update_progress(step, total):
            progress_bar.progress(step / total)
            status_text.text(messages[step % len(messages)])

        env.run(num_steps=NUM_SIMULATIONS, progress_callback=update_progress)
        visualizer = SimulationVisualizer(env, env.datacollector)

        st.markdown(f"### {label}")

        # === Winner summary ===
        results_df = env.datacollector.get_long_dataframe()
        if "winner" in results_df.columns:
            winner = results_df["winner"].iloc[-1]
            st.metric(label="🏆 Winning Candidate", value=f"Candidate {winner}")

        # === Preferences distribution ===
        fig, axes = plt.subplots(NUM_PREFERENCES, 1, figsize=(6, 4), sharex=True)
        fig.suptitle(f"Preferences Distribution ({label})", fontsize=16)
        visualizer.plot_preference_distributions(num_voters_to_show=10, axes=axes)
        st.pyplot(fig)

        # === Belief trajectory ===
        my_fig = plt.figure(figsize=(6, 4))
        my_fig.suptitle(f"Belief Trajectory ({label})", fontsize=16)
        gs = gridspec.GridSpec(1, 5, figure=my_fig)
        axes = my_fig.add_subplot(gs[0, 0:4])
        density_ax = my_fig.add_subplot(gs[0, 4], sharey=axes)
        voter = next(agent for agent in env.agents if isinstance(agent, Voter))
        visualizer.plot_belief_trajectory(voter=voter, axes=(axes, density_ax))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(my_fig)

        # === Results side-by-side (Density vs Histogram) ===
        col1, col2 = st.columns(2)
        with col1:
            fig, axes = visualizer.plot_simulation_results_distribution(
                plot_kind="stripplot"
            )
            if fig is not None:
                axes[0].set_xlabel("Proportion of Votes (1st Round)")
                axes[1].set_xlabel("Proportion of Votes (2nd Round)")
                st.pyplot(fig)

        with col2:
            fig, axes = visualizer.plot_simulation_results_distribution(
                plot_kind="histogram"
            )
            if fig is not None:
                axes[0].set_ylim(0, 80)
                axes[1].set_ylim(0, 80)
                axes[0].set_xlabel("Proportion of Votes (1st Round)")
                axes[1].set_xlabel("Proportion of Votes (2nd Round)")
                st.pyplot(fig)

    # Boucle sur chaque système
    for i, name in enumerate(selected_systems):
        with tabs[i]:
            st.subheader(f"{name} Voting Comparison")

            col1, col2 = st.columns(2)

            with col1:
                run_and_display(available_systems_no_tom[name], False, "No ToM")

            with col2:
                run_and_display(available_systems_tom[name], True, "With ToM")

    progress_bar.empty()
    status_text.text("All simulations completed!")
