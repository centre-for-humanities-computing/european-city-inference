import argparse
import json
import os
from datetime import datetime

import jax
import pandas as pd

from eci.environment import EnvConfig, Environment
from eci.metrics import batch_compute_metrics
from eci.voting_system.plurality import _vote_plurality
from eci.voting_system.quadratic import _vote_quadratic
from eci.voting_system.random_voting import _vote_random


def main():
    """Run multiple simulations of ECI voting systems and save results."""
    parser = argparse.ArgumentParser(
        description="Run an ECI voting simulation with multiple iterations."
    )
    parser.add_argument(
        "--agents", type=int, default=1000, help="Number of agents (voters)."
    )
    parser.add_argument(
        "--candidates", type=int, default=5, help="Number of candidates."
    )
    parser.add_argument(
        "--preferences", type=int, default=6, help="Number of preferences."
    )
    parser.add_argument(
        "--simulations", type=int, default=100, help="Number of simulations to run."
    )
    parser.add_argument(
        "--budget", type=float, default=99.0, help="Budget for quadratic voting."
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
        default="results",
        help="Directory to save the CSV and JSON.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="figures",
        help="Directory to save the generated figures.",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{timestamp}"

    print(f"🚀 Running {args.simulations} simulations for all voting systems...")
    print(f"📊 Run ID: {run_id}")
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
    print("   -> Running agent perception inference...")
    env._run_multi_agent_inference()

    # Split JAX keys for fairness
    base_key = jax.random.PRNGKey(args.seed)
    key_quad, key_plur, key_rand = jax.random.split(base_key, 3)

    # Execute N simulations using your class method
    print("   -> Running Plurality Voting...")
    sim_plurality = env.run_n_simulation(_vote_plurality, key_plur, args.simulations)
    metrics_plurality = batch_compute_metrics(sim_plurality)
    metrics_plurality["system"] = "plurality"

    print("   -> Running Quadratic Voting...")
    sim_qv = env.run_n_simulation(
        _vote_quadratic, key_quad, args.simulations, budget=args.budget
    )
    metrics_qv = batch_compute_metrics(sim_qv)
    metrics_qv["system"] = "quadratic"

    print("   -> Running Random Voting...")
    sim_rdm = env.run_n_simulation(_vote_random, key_rand, args.simulations)
    metrics_rdm = batch_compute_metrics(sim_rdm)
    metrics_rdm["system"] = "random"

    # Compile metrics and ADD PARAMETERS to the DataFrame
    combined_df = pd.concat(
        [metrics_plurality, metrics_qv, metrics_rdm], ignore_index=True
    )

    # Adding parameters as columns so they are strictly tied to the data
    combined_df["num_agents"] = args.agents
    combined_df["num_candidates"] = args.candidates
    combined_df["num_preferences"] = args.preferences
    combined_df["budget"] = args.budget
    combined_df["seed"] = args.seed
    combined_df["run_id"] = run_id

    # 6. Save Data and Configuration
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, f"{run_id}_metrics.csv")
    json_path = os.path.join(args.out_dir, f"{run_id}_config.json")

    # Save CSV
    combined_df.to_csv(csv_path, index=False)

    # Save parameters as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    print("Simulation finished.")
    print(f"Results saved to: {csv_path}")
    print(f"Config saved to: {json_path}")
    print("\nSummary:")
    numeric_cols = combined_df.select_dtypes(include="number").columns.tolist()
    print(combined_df.groupby("system")[numeric_cols].mean())


if __name__ == "__main__":
    main()
