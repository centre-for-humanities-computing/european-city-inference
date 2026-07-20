import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

import jax
import pandas as pd
from matplotlib.figure import Figure

from eci.decision import ElectionData, response_function
from eci.environment import EnvConfig, Environment
from eci.metrics import batch_compute_metrics
from eci.plots import plot_belief_trajectory, plot_preference, plot_voting_metrics
from eci.utils import _extract_env_data_vectorized, get_voter_trajectory_data
from eci.voting import _vote_plurality, _vote_quadratic

# TODO: re-enable when strategic / random voting are restored.
# from eci.voting.plurality import strategic_vote
# from eci.voting.quadratic import strategic_quadratic_vote
# from eci.voting.random_voting import _vote_uniform_random


@dataclass(frozen=True)
class VotingSystemRun:
    """A named voting-system simulation with its dedicated PRNG key."""

    name: str
    voting_function: Callable
    key: Any


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for a simulation run."""
    parser = argparse.ArgumentParser(
        description="Run a voting simulation with multiple iterations."
    )
    parser.add_argument("--agents", type=int, default=100, help="Number of agents.")
    parser.add_argument(
        "--candidates", type=int, default=5, help="Number of candidates."
    )
    parser.add_argument(
        "--preferences", type=int, default=4, help="Number of preferences."
    )
    parser.add_argument(
        "--simulations", type=int, default=100, help="Number of simulations to run."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="experiment",
        help="Name of the run for saving files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../results",
        help="Directory to save the CSV and JSON.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="../figures",
        help="Directory to save the generated figures.",
    )
    return parser.parse_args()


def _run_voting_system(
    env: Environment,
    data: ElectionData,
    run: VotingSystemRun,
    n_simulations: int,
) -> pd.DataFrame:
    """Run one voting system and return its labeled metrics."""
    print(f"Running {run.name} Voting")
    simulation_results = env.run_n_simulation(
        run.voting_function,
        data,
        response_function,
        run.key,
        n_simulations,
    )
    metrics = batch_compute_metrics(simulation_results)
    metrics["voting_system"] = run.name
    return metrics


def _add_run_metadata(
    metrics: pd.DataFrame,
    args: argparse.Namespace,
    run_id: str,
) -> None:
    """Attach run-level configuration fields to every metrics row."""
    metrics["num_agents"] = args.agents
    metrics["num_candidates"] = args.candidates
    metrics["num_preferences"] = args.preferences
    metrics["seed"] = args.seed
    metrics["run_id"] = run_id


def _save_outputs(
    args: argparse.Namespace,
    run_id: str,
    metrics: pd.DataFrame,
    figures: Mapping[str, Figure],
) -> tuple[str, str]:
    """Persist run metrics, configuration and figures."""
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{run_id}_metrics.csv")
    json_path = os.path.join(args.out_dir, f"{run_id}_config.json")

    for figure_name, figure in figures.items():
        figure.savefig(
            os.path.join(args.fig_dir, f"{run_id}_{figure_name}.png"),
            dpi=150,
            bbox_inches="tight",
        )

    metrics.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=4)

    return csv_path, json_path


def _print_summary(metrics: pd.DataFrame, csv_path: str, json_path: str) -> None:
    """Print saved paths and per-system averages."""
    print("Simulation finished.")
    print(f"Results saved to: {csv_path}")
    print(f"Config saved to: {json_path}")
    print("\nSummary:")
    numeric_columns = metrics.select_dtypes(include="number").columns.tolist()
    print(metrics.groupby("voting_system")[numeric_columns].mean())


def main():
    """Run multiple simulations of ECI voting systems and save results."""
    args = _parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{timestamp}"

    print(f"Running {args.simulations} simulations")
    print(f"Run ID: {run_id}")
    print(
        f"   Agents: {args.agents} | Candidates: {args.candidates} | Seed: {args.seed}"
    )

    # Configuration and Initialization
    config = EnvConfig(
        num_voters=args.agents,
        num_candidates=args.candidates,
        num_preferences=args.preferences,
        seed=args.seed,
    )
    env = Environment(config)

    # Run agent perception inference (HGF)
    print("Running agent belief update")
    env._run_multi_agent_inference()

    # Vectorize the environment data once for all voting systems.
    data = _extract_env_data_vectorized(env)

    # Plot belief trajectory for the first voter as an example.
    # `get_voter_trajectory_data` returns a dict matching the kwargs of
    # `plot_belief_trajectory`; `plot_voting_metrics` expects a DataFrame
    # of vote outcomes and is called later on `combined_df`.
    traj_data = get_voter_trajectory_data(env, voter_id=0)
    fig_trajectories, _, _ = plot_belief_trajectory(**traj_data)

    print("Saving preference plot")
    fig_preference, _ = plot_preference(data)

    base_key = jax.random.PRNGKey(args.seed)
    key_quad, key_plur = jax.random.split(base_key, 2)

    voting_systems = [
        VotingSystemRun("Plurality", _vote_plurality, key_plur),
        VotingSystemRun("Quadratic", _vote_quadratic, key_quad),
    ]
    metrics_by_system = [
        _run_voting_system(env, data, run, args.simulations) for run in voting_systems
    ]

    # Combine all metrics into one DataFrame
    combined_df = pd.concat(metrics_by_system, ignore_index=True)
    _add_run_metadata(combined_df, args, run_id)

    print("Saving voting metrics plot")
    fig_voting_metrics, _ = plot_voting_metrics(combined_df)

    figures = {
        "preference": fig_preference,
        "voting_metrics": fig_voting_metrics,
        "trajectories": fig_trajectories,
    }
    csv_path, json_path = _save_outputs(
        args,
        run_id,
        combined_df,
        figures,
    )
    _print_summary(combined_df, csv_path, json_path)


if __name__ == "__main__":
    main()
