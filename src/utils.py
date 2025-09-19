from typing import Optional

import numpy as np
from scipy.stats import halfnorm, norm


def generate_observations(
    n_nodes: int,
    n_steps: int,
    scenario: int = 1,
    shock_pattern: Optional[str] = None,
    shock_time: Optional[int] = None,
    recovery_time: Optional[int] = None,
    trend_shape: str = "linear",
    dispersion: float = 1.0,
) -> np.ndarray:
    """Generate observations for nodes based on specified scenarios.

    Parameters
    ----------
    n_nodes : int
        Number of nodes for which observations are generated.
    n_steps : int
        Number of time steps for simulations.
    scenario : int, optional (default=1)
        Scenario identifier (1 or 2).
    shock_pattern : str, optional
        Pattern of shock for scenario 2 (None, "phase", "sudden", or "trend").
    shock_time : int, optional
        Time step at which shock begins.
    recovery_time : int, optional
        Time step at which recovery begins.
    trend_shape : str, optional (default="linear")
        Shape of the trend. Supported: "linear".
    dispersion : float, optional (default=1.0)
        Controls the dispersion of noise added to observations.

    Returns
    -------
    np.ndarray
        A 2D array with observations for each node across all time steps.
    """
    np.random.seed(42)  # Fix seed for reproducibility
    node_observations = []
    phase1_params = (15, 1)
    phase2_params = (2, 2)
    phase3_params = phase1_params

    def generate_beta(params, size):
        a, b = params
        obs = np.random.beta(a, b, size=size)
        # Add Gaussian noise for dispersion
        obs += np.random.normal(0, 0.05 * dispersion, size=size)
        return np.clip(obs, 0, 1)

    for node in range(n_nodes):
        if scenario == 1:
            # Generate observations using phase 1 parameters for all time steps
            node_observations.append(generate_beta(phase1_params, n_steps))
        elif scenario == 2:
            # Set default shock and recovery times if not provided
            shock_time = shock_time or n_steps // 3
            recovery_time = recovery_time or 2 * n_steps // 3

            if shock_pattern in [None, "phase"]:
                phase1_end, phase2_end = shock_time, recovery_time
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
                    # Calculate weight based on trend shape
                    if trend_shape == "linear":
                        weight = t / recovery_time
                    else:
                        weight = (t / recovery_time) ** 2
                    # Interpolate between phase1 and phase2 parameters
                    alpha = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                    beta_param = (
                        phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)[0]

                for t in range(recovery_time, n_steps):
                    if trend_shape == "linear":
                        weight = 1 - (t - recovery_time) / (n_steps - recovery_time)
                    else:
                        weight = (
                            1 - (t - recovery_time) / (n_steps - recovery_time)
                        ) ** 2
                    # Interpolate back between phase2 and phase1 parameters
                    alpha = phase2_params[0] * (1 - weight) + phase1_params[0] * weight
                    beta_param = (
                        phase2_params[1] * (1 - weight) + phase1_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)[0]
            else:
                raise ValueError("Invalid shock_pattern specified for scenario 2.")

            node_observations.append(obs)
        else:
            raise ValueError("Scenario must be 1 or 2.")

    # Stack node observations horizontally to form a 2D array
    return np.column_stack(node_observations)


def generate_candidates(
    n_candidates, n_preferences, manual_means=None, manual_precisions=None
):
    """Generate a list of candidates, each with preferences.

    - If manual_means and manual_precisions are provided, they must be
      numpy arrays (or nested lists) of shape (n_candidates, n_preferences).
    - Otherwise, candidates are generated randomly:
        mus ~ Normal(loc=2, scale=1)
        pis ~ HalfNormal(scale=1)

    Parameters
    ----------
    n_candidates : int
        The number of candidates to generate.
    n_preferences : int
        The number of preferences for each candidate.
    manual_means : array-like, optional
        Shape (n_candidates, n_preferences). If provided, overrides random generation.
    manual_precisions : array-like, optional
        Shape (n_candidates, n_preferences). If provided, overrides random generation.

    Returns
    -------
    list of tuple of numpy.ndarray
        A list of candidates. Each candidate is represented as a tuple
        (mus, pis), where mus and pis are numpy arrays of length n_preferences.
    """
    if manual_means is not None and manual_precisions is not None:
        manual_means = np.array(manual_means)
        manual_precisions = np.array(manual_precisions)

        if manual_means.shape != (n_candidates, n_preferences):
            raise ValueError("means must have shape (n_candidates, n_preferences)")

        if manual_precisions.shape != (n_candidates, n_preferences):
            raise ValueError("precisions must have shape (n_candidates, n_preferences)")

        candidates = [
            (manual_means[i], manual_precisions[i]) for i in range(n_candidates)
        ]

    else:
        mu_sigma = 1
        sigma_scale = 1

        candidates = []
        for _ in range(n_candidates):
            mus = norm.rvs(loc=2, scale=mu_sigma, size=n_preferences)
            pis = halfnorm.rvs(scale=sigma_scale, size=n_preferences)
            candidates.append((mus, pis))

    return candidates
