# voting_simulation/environment/environment.py

from typing import List, Dict
import numpy as np
from agents.voter import Voter

class Environment:
    """Manages the world state and notifies observers of changes."""
    def __init__(self, num_dimensions: int):
        self._observers: List[Voter] = []
        self.num_dimensions = num_dimensions
        self.world_state: Dict[str, np.ndarray] = {
            'mu': np.random.rand(num_dimensions),
            'sigma': np.random.rand(num_dimensions) * 0.1 + 0.05
        }

    def register(self, observer: Voter):
        self._observers.append(observer)

    def notify(self, event: Dict[str, np.ndarray]):
        for observer in self._observers:
            observer.update(event)


def generate_observations(
    n_nodes,
    n_steps,
    scenario=1,
    shock_pattern=None,
    shock_time=None,
    recovery_time=None,
    trend_shape="linear",
):
    """
    Generate observations for nodes based on specified scenarios and shock patterns.

    Parameters
    ----------
    - n_nodes: int, number of nodes.
    - n_steps: int, number of time steps.
    - scenario: int, scenario type (1 or 2).
    - shock_pattern: str, pattern of shock ("phase", "sudden", "trend", or None).
    - shock_time: int, time step at which the shock occurs.
    - recovery_time: int, time step at which recovery begins.
    - trend_shape: str, shape of the trend ("linear" or other shapes).

    Returns
    -------
    - numpy.ndarray, array of generated observations.
    """
    np.random.seed(42)  # Fix seed for reproducibility
    node_observations = []
    # Default beta parameters for the nodes
    phase1_params = (5, 1)
    phase2_params = (2, 2)
    phase3_params = phase1_params

    def generate_beta(params, size):
        return np.random.beta(a=params[0], b=params[1], size=size)

    for node in range(n_nodes):
        # Scenario 1: Stable observations
        if scenario == 1:
            node_observations.append(generate_beta(phase1_params, n_steps))
        # Scenario 2: Shock scenarios
        elif scenario == 2:
            shock_time = shock_time or n_steps // 3
            recovery_time = recovery_time or 2 * n_steps // 3
            if shock_pattern in [None, "phase"]:
                phase1_end, phase2_end = (
                    (shock_time, recovery_time)
                    if recovery_time
                    else (n_steps // 3, 2 * n_steps // 3)
                )
                obs = np.concatenate(
                    [
                        generate_beta(phase1_params, phase1_end),
                        generate_beta(phase2_params, phase2_end - phase1_end),
                        generate_beta(phase3_params, n_steps - phase2_end),
                    ]
                )
            elif shock_pattern == "sudden":
                obs = np.concatenate(
                    [
                        generate_beta(phase1_params, shock_time),
                        generate_beta(phase2_params, recovery_time - shock_time),
                        generate_beta(phase3_params, n_steps - recovery_time),
                    ]
                )
            elif shock_pattern == "trend":
                obs = np.zeros(n_steps)
                for t in range(recovery_time):
                    weight = (
                        (t / recovery_time)
                        if trend_shape == "linear"
                        else (t / recovery_time) ** 2
                    )
                    alpha = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                    beta_param = (
                        phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)
                for t in range(recovery_time, n_steps):
                    weight = (
                        1 - ((t - recovery_time) / (n_steps - recovery_time))
                        if trend_shape == "linear"
                        else (1 - (t - recovery_time) / (n_steps - recovery_time)) ** 2
                    )
                    alpha = phase2_params[0] * (1 - weight) + phase1_params[0] * weight
                    beta_param = (
                        phase2_params[1] * (1 - weight) + phase1_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)
            else:
                raise ValueError("Invalid shock_pattern specified for scenario 2.")
            node_observations.append(obs)
        else:
            raise ValueError("Scenario must be 1 or 2.")
    return np.column_stack(node_observations)
