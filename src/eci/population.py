from dataclasses import dataclass
from typing import List, Tuple

import jax

from eci.agents import Candidate, Voter


@dataclass(frozen=True)
class PopulationConfig:
    """Sampling parameters for a :class:`Population`.

    Attributes
    ----------
    num_voters, num_candidates, num_preferences :
        Shapes of the population.
    preference_mean_range, preference_precision_range :
        ``(low, high)`` uniform-sample ranges for voter preferences.
    policy_mean_range, policy_precision_range :
        ``(low, high)`` uniform-sample ranges for candidate policies.
    tonic_volatility_mean, tonic_volatility_std :
        Mean and std of the Gaussian sampling the per-voter tonic
        volatility (used by the perceptual model, but generated here
        because each voter owns one).
    seed :
        Master PRNG seed for sampling.
    """

    num_voters: int
    num_candidates: int
    num_preferences: int
    preference_mean_range: Tuple[float, float] = (-5.0, 5.0)
    preference_precision_range: Tuple[float, float] = (0.05, 1.0)
    policy_mean_range: Tuple[float, float] = (-5.0, 5.0)
    policy_precision_range: Tuple[float, float] = (0.05, 1.0)
    tonic_volatility_mean: float = -2.0
    tonic_volatility_std: float = 0.01
    seed: int = 42

    @classmethod
    def from_env_config(cls, env_config) -> "PopulationConfig":
        """Build a PopulationConfig from a flat :class:`EnvConfig`."""
        return cls(
            num_voters=env_config.num_voters,
            num_candidates=env_config.num_candidates,
            num_preferences=env_config.num_preferences,
            tonic_volatility_mean=env_config.tonic_volatility_mean,
            tonic_volatility_std=env_config.tonic_volatility_std,
            seed=env_config.seed,
        )


@dataclass(frozen=True)
class Population:
    """A voting population as a JAX pytree.

    All fields are JAX arrays (data fields). The pytree registration
    means ``jax.vmap(f)(population)`` walks every field with the same
    leading axis — useful for sweeping over a batch of populations.

    Attributes
    ----------
    voter_means : Array, shape (n_voters, n_pref)
    voter_precisions : Array, shape (n_voters, n_pref)
    voter_volatilities : Array, shape (n_voters,)
    candidate_means : Array, shape (n_candidates, n_pref)
    candidate_precisions : Array, shape (n_candidates, n_pref)
    """

    voter_means: jax.Array
    voter_precisions: jax.Array
    voter_volatilities: jax.Array
    candidate_means: jax.Array
    candidate_precisions: jax.Array

    @classmethod
    def random(cls, config: PopulationConfig, key=None) -> "Population":
        """Sample a fresh random population from a :class:`PopulationConfig`."""
        if key is None:
            key = jax.random.PRNGKey(config.seed)
        k_voters, k_candidates = jax.random.split(key)
        voter_means, voter_precisions, voter_vols = _sample_voter_arrays(
            k_voters, config
        )
        candidate_means, candidate_precisions = _sample_candidate_arrays(
            k_candidates, config
        )
        return cls(
            voter_means=voter_means,
            voter_precisions=voter_precisions,
            voter_volatilities=voter_vols,
            candidate_means=candidate_means,
            candidate_precisions=candidate_precisions,
        )

    def as_voters(self, start_id: int = 0) -> List[Voter]:
        """Materialise voters as a list of :class:`Voter` dataclasses.

        Lossy: trajectory / vote_round_* fields are reset to defaults.
        Useful only for code that iterates over agent objects.
        """
        means = jax.device_get(self.voter_means)
        precs = jax.device_get(self.voter_precisions)
        vols = jax.device_get(self.voter_volatilities)
        return [
            Voter(
                id=start_id + i,
                preferences={"mean": means[i], "precision": precs[i]},
                tonic_volatility=float(vols[i]),
            )
            for i in range(self.voter_means.shape[0])
        ]

    def as_candidates(self, start_id: int = 0) -> List[Candidate]:
        """Materialise candidates as a list of :class:`Candidate` dataclasses."""
        means = jax.device_get(self.candidate_means)
        precs = jax.device_get(self.candidate_precisions)
        return [
            Candidate(
                id=start_id + i,
                policy={"mean": means[i], "precision": precs[i]},
            )
            for i in range(self.candidate_means.shape[0])
        ]

    @property
    def n_voters(self) -> int:
        """Number of voters in the population."""
        return self.voter_means.shape[0]

    @property
    def n_candidates(self) -> int:
        """Number of candidates in the population."""
        return self.candidate_means.shape[0]

    @property
    def n_preferences(self) -> int:
        """Number of preference dimensions per agent."""
        return self.voter_means.shape[1]


# Register Population as a JAX pytree (all fields are data fields).
jax.tree_util.register_dataclass(
    Population,
    data_fields=[
        "voter_means",
        "voter_precisions",
        "voter_volatilities",
        "candidate_means",
        "candidate_precisions",
    ],
    meta_fields=[],
)


def _sample_voter_arrays(key, config: PopulationConfig):
    """Sample voter preference and volatility arrays.

    Parameters
    ----------
    key : jax.Array
        PRNG key, split three ways (means, precisions, volatilities).
    config : PopulationConfig
        Sampling ranges and volatility hyperparameters.

    Returns
    -------
    means : jax.Array, shape (n_voters, n_preferences)
    precisions : jax.Array, shape (n_voters, n_preferences)
    volatilities : jax.Array, shape (n_voters,)
    """
    k_mean, k_prec, k_vol = jax.random.split(key, 3)
    means = jax.random.uniform(
        k_mean,
        shape=(config.num_voters, config.num_preferences),
        minval=config.preference_mean_range[0],
        maxval=config.preference_mean_range[1],
    )
    precisions = jax.random.uniform(
        k_prec,
        shape=(config.num_voters, config.num_preferences),
        minval=config.preference_precision_range[0],
        maxval=config.preference_precision_range[1],
    )
    volatilities = (
        jax.random.normal(k_vol, shape=(config.num_voters,))
        * config.tonic_volatility_std
        + config.tonic_volatility_mean
    )
    return means, precisions, volatilities


def _sample_candidate_arrays(key, config: PopulationConfig):
    """Sample candidate policy arrays.

    Parameters
    ----------
    key : jax.Array
        PRNG key, split two ways (means, precisions).
    config : PopulationConfig
        Policy sampling ranges.

    Returns
    -------
    means : jax.Array, shape (n_candidates, n_preferences)
    precisions : jax.Array, shape (n_candidates, n_preferences)
    """
    k_mean, k_prec = jax.random.split(key)
    means = jax.random.uniform(
        k_mean,
        shape=(config.num_candidates, config.num_preferences),
        minval=config.policy_mean_range[0],
        maxval=config.policy_mean_range[1],
    )
    precisions = jax.random.uniform(
        k_prec,
        shape=(config.num_candidates, config.num_preferences),
        minval=config.policy_precision_range[0],
        maxval=config.policy_precision_range[1],
    )
    return means, precisions
