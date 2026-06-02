from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp

from eci.observations import generate_observations


@dataclass(frozen=True)
class WorldConfig:
    """Parameters governing observation generation.

    Mirrors the observation-related fields of
    :class:`eci.environment.EnvConfig`. See
    :func:`eci.utils.generate_observations` for the full semantics.
    """

    num_steps: int = 362
    num_nodes: int = 1
    scenario: int = 2
    dispersion: float = 1.0
    shock_pattern: Optional[Literal["phase", "sudden", "trend"]] = None
    obs_low: float = 0.0
    obs_high: float = 1.0
    obs_seed: Optional[int] = None
    recover: bool = False

    @classmethod
    def from_env_config(cls, env_config) -> "WorldConfig":
        """Build a WorldConfig from a flat :class:`EnvConfig`."""
        return cls(
            num_steps=env_config.num_steps,
            num_nodes=env_config.num_preferences,
            scenario=env_config.scenario,
            dispersion=env_config.dispersion,
            shock_pattern=env_config.shock_pattern,
            obs_low=env_config.obs_low,
            obs_high=env_config.obs_high,
            obs_seed=env_config.obs_seed,
            recover=env_config.recover,
        )


@dataclass(frozen=True)
class World:
    """The observation stream the agents see.

    Attributes
    ----------
    observations : Array, shape (n_steps, n_nodes)
        Observation time series. Always in ``[obs_low, obs_high]``.
    """

    observations: jax.Array

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: WorldConfig) -> "World":
        """Generate a fresh observation stream from a :class:`WorldConfig`."""
        obs = generate_observations(
            n_nodes=config.num_nodes,
            n_steps=config.num_steps,
            scenario=config.scenario,
            shock_pattern=config.shock_pattern,
            dispersion=config.dispersion,
            obs_low=config.obs_low,
            obs_high=config.obs_high,
            recover=config.recover,
            seed=config.obs_seed,
        )
        return cls(observations=jnp.asarray(obs))

    @classmethod
    def from_env_config(cls, env_config) -> "World":
        """Build a :class:`World` directly from a flat EnvConfig."""
        return cls.from_config(WorldConfig.from_env_config(env_config))

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        """Number of time steps in the observation stream."""
        return self.observations.shape[0]

    @property
    def n_nodes(self) -> int:
        """Number of observation channels (preference dimensions)."""
        return self.observations.shape[1]


# Register World as a JAX pytree (single data field).
jax.tree_util.register_dataclass(
    World,
    data_fields=["observations"],
    meta_fields=[],
)
