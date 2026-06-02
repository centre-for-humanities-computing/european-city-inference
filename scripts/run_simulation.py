import argparse
import json
import os
from datetime import datetime

import jax
import pandas as pd

from eci.decision import response_function
from eci.environment import EnvConfig, Environment
from eci.metrics import batch_compute_metrics
from eci.plots import plot_belief_trajectory, plot_preference, plot_voting_metrics
from eci.utils import _extract_env_data_vectorized, get_voter_trajectory_data
from eci.voting import _vote_plurality, _vote_quadratic

# TODO: re-enable when strategic / random voting are restored.
# from eci.voting.plurality import strategic_vote
# from eci.voting.quadratic import strategic_quadratic_vote
# from eci.voting.random_voting import _vote_uniform_random


def main():
    """Run multiple simulations of ECI voting systems and save results."""
    parser = argparse.ArgumentParser(
        description="Run an voting simulation with multiple iterations."
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
    args = parser.parse_args()

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
    # TODO: re-add `key_rand` when random voting is restored.
    key_quad, key_plur = jax.random.split(base_key, 2)

    # plurality
    print("Running Plurality Voting")
    sim_plurality = env.run_n_simulation(
        _vote_plurality, data, response_function, key_plur, args.simulations
    )
    metrics_plurality = batch_compute_metrics(sim_plurality)
    metrics_plurality["voting_system"] = "Plurality"

    # TODO: restore strategic plurality voting.
    # print("Running Strategic Plurality Voting")
    # sim_plurality_strat = env.run_n_simulation(
    #     strategic_vote, data, response_function, key_plur, args.simulations
    # )
    # metrics_plurality_strat = batch_compute_metrics(sim_plurality_strat)
    # metrics_plurality_strat["voting_system"] = "Plur_Strat"

    # quadratic
    print("Running Quadratic Voting")
    sim_qv = env.run_n_simulation(
        _vote_quadratic, data, response_function, key_quad, args.simulations
    )
    metrics_qv = batch_compute_metrics(sim_qv)
    metrics_qv["voting_system"] = "Quadratic"

    # TODO: restore strategic quadratic voting.
    # print("Running Strategic Quadratic Voting")
    # sim_qv_strat = env.run_n_simulation(
    #     strategic_quadratic_vote, data, response_function, key_quad, args.simulations
    # )
    # metrics_qv_strat = batch_compute_metrics(sim_qv_strat)
    # metrics_qv_strat["voting_system"] = "Quad_Strat"

    # TODO: restore uniform random voting.
    # print("Running Uniform Random Voting")
    # sim_rdm = env.run_n_simulation(
    #     _vote_uniform_random, data, response_function, key_rand, args.simulations
    # )
    # metrics_rdm = batch_compute_metrics(sim_rdm)
    # metrics_rdm["voting_system"] = "Rdm_Uni"

    # Combine all metrics into one DataFrame
    combined_df = pd.concat(
        [
            metrics_plurality,
            # metrics_plurality_strat,  # TODO: restore.
            metrics_qv,
            # metrics_qv_strat,         # TODO: restore.
            # metrics_rdm,              # TODO: restore.
        ],
        ignore_index=True,
    )

    # Tag every row with run-level params
    combined_df["num_agents"] = args.agents
    combined_df["num_candidates"] = args.candidates
    combined_df["num_preferences"] = args.preferences
    combined_df["seed"] = args.seed
    combined_df["run_id"] = run_id

    print("Saving voting metrics plot")
    fig_voting_metrics, _ = plot_voting_metrics(combined_df)

    # Save data, config, and figures
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{run_id}_metrics.csv")
    json_path = os.path.join(args.out_dir, f"{run_id}_config.json")

    fig_preference.savefig(
        os.path.join(args.fig_dir, f"{run_id}_preference.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig_voting_metrics.savefig(
        os.path.join(args.fig_dir, f"{run_id}_voting_metrics.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig_trajectories.savefig(
        os.path.join(args.fig_dir, f"{run_id}_trajectories.png"),
        dpi=150,
        bbox_inches="tight",
    )

    combined_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    print("Simulation finished.")
    print(f"Results saved to: {csv_path}")
    print(f"Config saved to: {json_path}")
    print("\nSummary:")
    numeric_cols = combined_df.select_dtypes(include="number").columns.tolist()
    print(combined_df.groupby("voting_system")[numeric_cols].mean())


if __name__ == "__main__":
    main()
