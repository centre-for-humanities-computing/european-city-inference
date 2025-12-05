from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.stats import halfnorm, norm


def kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> ArrayLike:
    """Calculate the KL divergence between two Gaussian distributions."""
    # Conversion to JAX arrays for broadcasting
    mean_belief = jnp.asarray(mean_belief)
    precision_belief = jnp.asarray(precision_belief)
    mean_pref = jnp.asarray(mean_pref)
    precision_pref = jnp.asarray(precision_pref)

    # compute variances
    var_belief = 1.0 / precision_belief
    var_pref = 1.0 / precision_pref

    # KL divergence formula
    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )
    return kl


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
    """Generate observations for nodes."""
    if scenario not in [1, 2]:
        raise ValueError("Scenario must be 1 or 2")

    if scenario == 2:
        valid_patterns = [None, "phase", "sudden", "trend"]
        if shock_pattern not in valid_patterns:
            raise ValueError("Invalid shock_pattern")

    np.random.seed(42)
    node_observations = []

    # Parameters
    phase1_params: tuple[float, float] = (15.0, 1.0)
    phase2_params: tuple[float, float] = (2.0, 2.0)
    phase3_params = phase1_params

    def generate_beta(params, size):
        if size <= 0:
            return np.array([])
        a, b = params
        obs = np.random.beta(a, b, size=size)
        obs += np.random.normal(0, 0.05 * dispersion, size=size)
        return np.clip(obs, 0, 1)

    for node in range(n_nodes):
        if scenario == 1:
            obs = generate_beta(phase1_params, n_steps)

        else:  # Scenario 2
            s_time = shock_time if shock_time is not None else n_steps // 3
            r_time = recovery_time if recovery_time is not None else 2 * n_steps // 3

            if shock_pattern in [None, "phase"]:
                # Normal -> Shock -> Recovery
                p1 = generate_beta(phase1_params, s_time)
                p2 = generate_beta(phase2_params, r_time - s_time)
                p3 = generate_beta(phase3_params, n_steps - r_time)
                obs = np.concatenate([p1, p2, p3])

            elif shock_pattern == "sudden":
                # Normal -> Shock (persists)
                p1 = generate_beta(phase1_params, s_time)
                p2 = generate_beta(phase2_params, r_time - s_time)
                p3 = generate_beta(phase3_params, n_steps - r_time)
                obs = np.concatenate([p1, p2, p3])

            elif shock_pattern == "trend":
                obs = np.zeros(n_steps)
                for t in range(n_steps):
                    # Determine phase and weight
                    if t < s_time:
                        params = phase1_params
                    elif t < r_time:
                        # Degrading
                        progress = (t - s_time) / (r_time - s_time)
                        weight = progress if trend_shape == "linear" else progress**2
                        a = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                        b = phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                        params = (a, b)
                    else:
                        # Recovering
                        progress = (t - r_time) / (n_steps - r_time)
                        weight = (
                            1 - progress
                            if trend_shape == "linear"
                            else (1 - progress) ** 2
                        )
                        a = phase1_params[0] * weight + phase2_params[0] * (1 - weight)
                        b = phase1_params[1] * weight + phase2_params[1] * (1 - weight)
                        params = (a, b)

                    obs[t] = generate_beta(params, 1)[0]

        node_observations.append(obs)

    return np.column_stack(node_observations)


def generate_candidates(
    n_candidates: int,
    n_preferences: int,
    manual_means: Optional[ArrayLike] = None,
    manual_precisions: Optional[ArrayLike] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate candidates with preferences."""
    if manual_means is not None and manual_precisions is not None:
        manual_means = np.asarray(manual_means)
        manual_precisions = np.asarray(manual_precisions)

        if manual_means.shape != (n_candidates, n_preferences):
            raise ValueError(f"means must have shape ({n_candidates}, {n_preferences})")

        if manual_precisions.shape != (n_candidates, n_preferences):
            raise ValueError(
                f"precisions must have shape ({n_candidates}, {n_preferences})"
            )

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
