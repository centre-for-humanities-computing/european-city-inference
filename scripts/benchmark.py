import argparse
import os
import time

import jax
import pandas as pd

from eci.environment import EnvConfig, Environment
from eci.voting_system.plurality import _vote_plurality
from eci.voting_system.quadratic import _vote_quadratic
from eci.voting_system.random_voting import _vote_random


def measure_batch_time(voting_func, env, key, num_simulations, **kwargs):
    """Measures the time to run simulations."""
    start_time = time.perf_counter()

    results = env.run_n_simulation(voting_func, key, num_simulations, **kwargs)
    results[num_simulations - 1]["final_winner"].block_until_ready()

    end_time = time.perf_counter()
    return end_time - start_time


def main():
    """Benchmarks the performance across varying agent sizes."""
    parser = argparse.ArgumentParser(
        description="Benchmark JAX performance across multiple simulations."
    )
    parser.add_argument(
        "--simulations", type=int, default=1000, help="Number of iterations per batch."
    )
    parser.add_argument(
        "--candidates", type=int, default=8, help="Number of candidates."
    )
    parser.add_argument(
        "--preferences", type=int, default=8, help="Number of preferences."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_results.csv",
        help="Where to save the benchmark CSV.",
    )
    args = parser.parse_args()

    agent_sizes = [100, 1000, 5000, 10000]
    base_key = jax.random.PRNGKey(42)

    benchmark_data = []

    print(f"Starting Benchmark ({args.simulations} simulations/system)")
    print("=" * 65)
    print(
        f"{'Agents':<10} | {'System':<15} | {'Total Time (s)':<15} | {'Iter/sec':<10}"
    )
    print("-" * 65)

    for n_agents in agent_sizes:
        config = EnvConfig(
            num_voters=n_agents,
            num_candidates=args.candidates,
            num_preferences=args.preferences,
            seed=42,
        )
        env = Environment(config)

        env._run_multi_agent_inference()

        key_quad, key_plur, key_rand = jax.random.split(base_key, 3)

        # Benchmark Random
        rand_time = measure_batch_time(_vote_random, env, key_rand, args.simulations)
        rand_iter_sec = args.simulations / rand_time
        benchmark_data.append(
            {
                "agents": n_agents,
                "system": "Random",
                "total_time_s": rand_time,
                "iter_per_sec": rand_iter_sec,
            }
        )

        # Benchmark Plurality
        plur_time = measure_batch_time(_vote_plurality, env, key_plur, args.simulations)
        plur_iter_sec = args.simulations / plur_time
        benchmark_data.append(
            {
                "agents": n_agents,
                "system": "Plurality",
                "total_time_s": plur_time,
                "iter_per_sec": plur_iter_sec,
            }
        )

        # Benchmark Quadratic
        quad_time = measure_batch_time(
            _vote_quadratic, env, key_quad, args.simulations, budget=99.0
        )
        quad_iter_sec = args.simulations / quad_time
        benchmark_data.append(
            {
                "agents": n_agents,
                "system": "Quadratic",
                "total_time_s": quad_time,
                "iter_per_sec": quad_iter_sec,
            }
        )

        print("-" * 65)

    # Save the results to CSV
    df_results = pd.DataFrame(benchmark_data)

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(args.output, index=False)
    print(f"Benchmark complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
