from typing import Literal, Optional, Tuple

import numpy as np

Params = Tuple[float, float]
PhaseParams = Tuple[Params, Params]


def _get_parameter_trajectory(
    n_steps: int,
    s_time: int,
    r_time: int,
    pattern: Optional[str],
    trend_shape: str,
    phase_params: PhaseParams,
    recover: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the per-timestep ``(alpha, beta)`` Beta parameters."""
    (a1, b1), (a2, b2) = phase_params
    alpha_t = np.full(n_steps, a1, dtype=float)
    beta_t = np.full(n_steps, b1, dtype=float)

    if pattern is None:
        # Stable: parameters stay at phase-1 values (a1, b1) throughout — no shock.
        pass
    elif pattern == "phase":
        alpha_t[s_time:r_time] = a2
        beta_t[s_time:r_time] = b2
    elif pattern == "sudden":
        end = r_time if recover else n_steps  # recover → temporary; else permanent
        alpha_t[s_time:end] = a2
        beta_t[s_time:end] = b2
    elif pattern == "trend":
        t = np.arange(n_steps)
        mask_degrade = (t >= s_time) & (t < r_time)
        mask_recover = t >= r_time
        if np.any(mask_degrade):
            prog = (t[mask_degrade] - s_time) / (r_time - s_time)
            w = prog if trend_shape == "linear" else prog**2
            alpha_t[mask_degrade] = a1 * (1 - w) + a2 * w
            beta_t[mask_degrade] = b1 * (1 - w) + b2 * w
        if np.any(mask_recover):
            prog = (t[mask_recover] - r_time) / (n_steps - r_time)
            w = (1 - prog) if trend_shape == "linear" else (1 - prog) ** 2
            alpha_t[mask_recover] = a1 * (1 - w) + a2 * w
            beta_t[mask_recover] = b1 * (1 - w) + b2 * w

    return alpha_t, beta_t


def _validate_observation_args(scenario, shock_pattern):
    """Validate ``scenario`` / ``shock_pattern``, raising ``ValueError`` if invalid."""
    if scenario not in [1, 2]:
        raise ValueError("Scenario must be 1 or 2")
    if scenario == 2 and shock_pattern not in [None, "phase", "sudden", "trend"]:
        raise ValueError(f"Invalid shock_pattern: {shock_pattern}")


def _resolve_shock_times(n_steps, shock_time, recovery_time):
    """Resolve shock / recovery step indices."""
    s = shock_time if shock_time is not None else n_steps // 3
    r = recovery_time if recovery_time is not None else 2 * n_steps // 3
    s = np.clip(s, 0, n_steps)
    r = np.clip(r, s, n_steps)
    return s, r


def _sample_beta_signal(rng, alpha_t, beta_t, n_steps, n_nodes):
    """Raw signal in [0, 1] from a time-varying Beta distribution."""
    return rng.beta(alpha_t[:, None], beta_t[:, None], size=(n_steps, n_nodes))


def _rescale_and_add_noise(obs, rng, obs_low, obs_high, dispersion):
    """Affine-map ``[0, 1] → [obs_low, obs_high]`` then add Gaussian noise."""
    span = obs_high - obs_low
    obs = obs_low + span * obs
    if dispersion > 0:
        obs += rng.normal(0, 0.05 * dispersion * span, size=obs.shape)
    return np.clip(obs, obs_low, obs_high)


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
    obs_low: float = 0.0,
    obs_high: float = 1.0,
    recover: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic observation time series of shape ``(n_steps, n_nodes)``.

    Parameters
    ----------
    n_nodes
        Number of independent observation channels.
    n_steps
        Time-series length.
    scenario
        ``1`` = stable (no shock allowed). ``2`` = enables ``shock_pattern``.
    shock_pattern
        ``"phase"`` (windowed), ``"sudden"`` (permanent), or ``"trend"``
        (smooth transition). Only active when ``scenario == 2``.
    shock_time, recovery_time
        Time-step indices for the shock window. Default to ``n_steps // 3``
        and ``2 * n_steps // 3``.
    trend_shape
        Shape of the trend transition: ``"linear"`` or ``"quadratic"``.
    dispersion
        Gaussian noise σ multiplier (``σ = 0.05 * dispersion * (obs_high - obs_low)``).
    phase_params
        ``((α1, β1), (α2, β2))`` Beta parameters for the two phases.
    obs_low, obs_high
        Output range. Default ``[0, 1]`` matches the raw Beta range.
    recover
        Only affects ``shock_pattern="sudden"``: when ``True`` the sudden
        shock is temporary (the world returns to its baseline at
        ``recovery_time``); when ``False`` (default) it is permanent.
        ``"phase"`` and ``"trend"`` recover regardless.
    seed
        Reproducibility seed for ``np.random.default_rng``. ``None`` → fresh entropy.

    Returns
    -------
    Array of shape ``(n_steps, n_nodes)`` clipped to ``[obs_low, obs_high]``.
    """
    _validate_observation_args(scenario, shock_pattern)
    rng = np.random.default_rng(seed)
    s_time, r_time = _resolve_shock_times(n_steps, shock_time, recovery_time)
    # Scenario 1 is always stable (no shock). Scenario 2 produces a shock:
    # if no explicit shock_pattern is given, default to a "phase" shock.
    if scenario == 1:
        pattern = None
    else:  # scenario == 2
        pattern = shock_pattern if shock_pattern is not None else "phase"
    alpha_t, beta_t = _get_parameter_trajectory(
        n_steps, s_time, r_time, pattern, trend_shape, phase_params, recover=recover
    )
    raw = _sample_beta_signal(rng, alpha_t, beta_t, n_steps, n_nodes)
    return _rescale_and_add_noise(raw, rng, obs_low, obs_high, dispersion)
