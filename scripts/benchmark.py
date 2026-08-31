import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import pandas as pd

from eci.decision import ElectionData, response_function
from eci.environment import EnvConfig, Environment
from eci.utils import _extract_env_data_vectorized
from eci.voting import _vote_plurality, _vote_quadratic


@dataclass(frozen=True)
class BenchmarkRun:
    """One named voting-system benchmark with its key and options."""

    name: str
    voting_function: Callable
    key: Any
    voting_options: Mapping[str, Any]


def measure_batch_time(
    run: BenchmarkRun,
    environment: Environment,
    data: ElectionData,
    num_simulations: int,
) -> float:
    """Measure the time to run `num_simulations` of a given voting function.

    Signature follows the post-refactor API:
        env.run_n_simulation(func, data, response_function, key, n_simulations, ...)
    Each simulation result now exposes `winner` (was `final_winner`).
    """
    start_time = time.perf_counter()

    simulation_results = environment.run_n_simulation(
        run.voting_function,
        data,
        response_function,
        run.key,
        num_simulations,
        **run.voting_options,
    )
    simulation_results[num_simulations - 1]["winner"].block_until_ready()

    return time.perf_counter() - start_time


def _parse_args() -> argparse.Namespace:
    """Parse benchmark command-line options."""
    parser = argparse.ArgumentParser(
        description="Benchmark JAX performance across multiple simulations."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of iterations per batch.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help="Number of candidates.",
    )
    parser.add_argument(
        "--preferences",
        type=int,
        default=8,
        help="Number of preferences.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/benchmark_results.csv",
        help="Where to save the benchmark CSV.",
    )
    return parser.parse_args()


def _benchmark_system(
    run: BenchmarkRun,
    environment: Environment,
    data: ElectionData,
    simulation_count: int,
    agent_count: int,
) -> dict:
    """Benchmark one voting system and return a result row."""
    total_time = measure_batch_time(
        run,
        environment,
        data,
        simulation_count,
    )
    return {
        "agents": agent_count,
        "system": run.name,
        "total_time_s": total_time,
        "iter_per_sec": simulation_count / total_time,
    }


def main():
    """Benchmarks the performance across varying agent sizes."""
    args = _parse_args()

    agent_sizes = [100, 1000, 5000, 10000]
    base_key = jax.random.PRNGKey(42)
    benchmark_rows = []

    print(f"Starting Benchmark ({args.simulations} simulations/system)")
    print("=" * 65)
    print(
        f"{'Agents':<10} | {'System':<15} | {'Total Time (s)':<15} | {'Iter/sec':<10}"
    )
    print("-" * 65)

    for agent_count in agent_sizes:
        config = EnvConfig(
            num_voters=agent_count,
            num_candidates=args.candidates,
            num_preferences=args.preferences,
            seed=42,
        )
        environment = Environment(config)
        environment._run_multi_agent_inference()
        data = _extract_env_data_vectorized(environment)

        quadratic_key, plurality_key = jax.random.split(base_key, 2)
        benchmark_runs = [
            BenchmarkRun(
                name="Plurality",
                voting_function=_vote_plurality,
                key=plurality_key,
                voting_options={},
            ),
            BenchmarkRun(
                name="Quadratic",
                voting_function=_vote_quadratic,
                key=quadratic_key,
                voting_options={"budget": 99.0},
            ),
        ]
        benchmark_rows.extend(
            _benchmark_system(
                run,
                environment,
                data,
                args.simulations,
                agent_count,
            )
            for run in benchmark_runs
        )

        print("-" * 65)

    results_frame = pd.DataFrame(benchmark_rows)
    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    results_frame.to_csv(args.output, index=False)
    print(f"Benchmark complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
