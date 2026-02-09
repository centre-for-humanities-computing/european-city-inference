from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.stats import halfnorm, norm

# Type aliases for readability
Params = Tuple[float, float]
PhaseParams = Tuple[Params, Params]


def kl_divergence(
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
) -> ArrayLike:
    """Calculate the KL divergence between two Gaussian distributions.

    Parameters
    ----------
    mean_pref :
        Mean of the preferred distribution.
    precision_pref :
        Precision of the preferred distribution.
    mean_belief :
        Mean of the belief distribution.
    precision_belief :
        Precision of the belief distribution.

    Returns
    -------
    The KL divergence between the two distributions.
    """
    # Conversion to JAX arrays for broadcasting
    mean_belief = jnp.asarray(mean_belief)
    precision_belief = jnp.asarray(precision_belief)
    mean_pref = jnp.asarray(mean_pref)
    precision_pref = jnp.asarray(precision_pref)

    # compute variances
    var_belief = 1.0 / precision_belief
    var_pref = 1.0 / precision_pref

    # KL divergence formula
    kl = (  # To do the KL blablabl (1)
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )
    return kl


def _get_parameter_trajectory(
    n_steps: int,
    s_time: int,
    r_time: int,
    pattern: Optional[str],
    trend_shape: str,
    phase_params: PhaseParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the alpha and beta parameters based on the shock pattern.

    Parameters
    ----------
    n_steps :
        Total number of time steps.
    s_time :
        Time step when the shock begins.
    r_time :
        Time step when recovery begins or ends.
    pattern :
        The pattern of the shock.
    trend_shape :
        The shape of the transition for 'trend' patterns.
    phase_params :
        Parameters for the Beta distribution (alpha, beta).

    Returns
    -------
    alpha_t :
        Array of alpha parameters for each time step.
    beta_t :
        Array of beta parameters for each time step.
    """
    (a1, b1), (a2, b2) = phase_params

    # Initialize with normal phase parameters
    alpha_t = np.full(n_steps, a1, dtype=float)
    beta_t = np.full(n_steps, b1, dtype=float)

    if pattern in [None, "phase"]:
        # Shock occurs, then immediately reverts after recovery time
        alpha_t[s_time:r_time] = a2
        beta_t[s_time:r_time] = b2

    elif pattern == "sudden":
        # Shock occurs and persists indefinitely
        alpha_t[s_time:] = a2
        beta_t[s_time:] = b2

    elif pattern == "trend":
        t = np.arange(n_steps)
        # Masks for different phases
        mask_degrade = (t >= s_time) & (t < r_time)
        mask_recover = t >= r_time

        # 1. Degradation Phase
        if np.any(mask_degrade):
            prog = (t[mask_degrade] - s_time) / (r_time - s_time)
            w = prog if trend_shape == "linear" else prog**2
            alpha_t[mask_degrade] = a1 * (1 - w) + a2 * w
            beta_t[mask_degrade] = b1 * (1 - w) + b2 * w

        # 2. Recovery Phase
        if np.any(mask_recover):
            prog = (t[mask_recover] - r_time) / (n_steps - r_time)
            w = (1 - prog) if trend_shape == "linear" else (1 - prog) ** 2
            alpha_t[mask_recover] = a1 * (1 - w) + a2 * w
            beta_t[mask_recover] = b1 * (1 - w) + b2 * w

    return alpha_t, beta_t


def generate_observations(
    n_nodes: int,
    n_steps: int,
    scenario: int = 1,
    shock_pattern: Optional[Literal["phase", "sudden", "trend"]] = None,
    shock_time: Optional[int] = None,
    recovery_time: Optional[int] = None,
    trend_shape: Literal["linear", "quadratic"] = "linear",
    dispersion: float = 1.0,
    phase_params: PhaseParams = ((15.0, 1.0), (2.0, 2.0)),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic observations for a set of nodes over time.

    Parameters
    ----------
    n_nodes :
        The number of nodes to generate data for.
    n_steps :
        The number of time steps (observations) per node.
    scenario :
        The scenario type.
    shock_pattern :
        The pattern of the shock.
    shock_time :
        Time step when the shock begins.
    recovery_time :
        Time step when recovery begins or ends.
    trend_shape :
        The shape of the transition for 'trend' patterns.
    dispersion :
        Multiplicative factor for the Gaussian noise added to observations.
    phase_params :
        Parameters for the Beta distribution (alpha, beta).
    seed :
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    Array containing the generated observations.
    """
    if scenario not in [1, 2]:
        raise ValueError("Scenario must be 1 or 2")
    if scenario == 2 and shock_pattern not in [None, "phase", "sudden", "trend"]:
        raise ValueError(f"Invalid shock_pattern: {shock_pattern}")
    rng = np.random.default_rng(seed)

    # time logic
    s_time = shock_time if shock_time is not None else n_steps // 3
    r_time = recovery_time if recovery_time is not None else 2 * n_steps // 3

    # clipping
    s_time = np.clip(s_time, 0, n_steps)
    r_time = np.clip(r_time, s_time, n_steps)

    # Scenario 1 is just Scenario 2 with no shock pattern (effectively)
    pattern = shock_pattern if scenario == 2 else None

    alpha_t, beta_t = _get_parameter_trajectory(
        n_steps, s_time, r_time, pattern, trend_shape, phase_params
    )

    # generate observations
    obs = rng.beta(alpha_t[:, None], beta_t[:, None], size=(n_steps, n_nodes))

    if dispersion > 0:
        noise = rng.normal(0, 0.05 * dispersion, size=(n_steps, n_nodes))
        obs += noise

    return np.clip(obs, 0, 1)


def generate_candidates(
    n_candidates: int,
    n_preferences: int,
    manual_means: Optional[ArrayLike] = None,
    manual_precisions: Optional[ArrayLike] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate candidates with preferences.

    Parameters
    ----------
    n_candidates :
        Number of candidates to generate.
    n_preferences :
        Number of preferences per candidate.
    manual_means :
        Optional manual means for the candidates.
    manual_precisions :
        Optional manual precisions for the candidates.

    Returns
    -------
        List of tuples (means, precisions) for each candidate.
    """
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


def get_voter_trajectory_data(env, voter_id: int, pref_idx: int = 0):
    """Retrieve specific arrays for a single voter's belief trajectory."""
    voter = next(v for v in env.voters if v.id == voter_id)
    return {
        "means": voter.trajectory["expected_mean"][voter.id],
        "precisions": voter.trajectory["precision"][voter.id],
        "observations": env.input_data[:, pref_idx],
        "preference_params": (
            voter.preferences["mean"][pref_idx],
            voter.preferences["precision"][pref_idx],
        ),
        "title_suffix": f"for Voter {voter_id}",
    }


def _extract_env_data_vectorized(env):
    """
    Extract and vectorize belief, preference, and policy data from the environment.

    This function transforms the environment's agent and candidate data into
    dense JAX arrays, organized as matrices where rows typically represent
    agents and columns represent preference dimensions.

    Parameters
    ----------
    env :
        The simulation environment containing agents and candidates.

    Returns
    -------
    data : dict
        A dictionary containing JAX arrays:

    """
    # Matrice (agent, preference)
    pref_idx_list = env.preferences_idx

    # Extract Candidate Data
    # Shape: (n_candidates, n_features)
    policy_means = jnp.stack([c.policy["mean"].ravel() for c in env.candidates])
    policy_precs = jnp.stack([c.policy["precision"].ravel() for c in env.candidates])

    # Extract Voter Beliefs
    # Stack along axis -1 to ensure shape is (n_agents, n_preferences)
    means_belief = jnp.stack(
        [env.last_attributes[i]["expected_mean"] for i in pref_idx_list], axis=-1
    )
    precs_belief = jnp.stack(
        [env.last_attributes[i]["expected_precision"] for i in pref_idx_list], axis=-1
    )

    # Extract Voter Preferences
    p_idx_jax = jnp.array(pref_idx_list)
    agent_pref_means = env.last_attributes[-1]["preferences"]["mean"][:, p_idx_jax]
    agent_pref_precs = env.last_attributes[-1]["preferences"]["precision"][:, p_idx_jax]

    return {
        "beliefs": {"mean": means_belief, "precision": precs_belief},
        "preferences": {"mean": agent_pref_means, "precision": agent_pref_precs},
        "candidates": {"mean": policy_means, "precision": policy_precs},
    }
