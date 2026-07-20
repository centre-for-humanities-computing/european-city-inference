from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import jax
import jax.numpy as jnp
import tqdm
from jax.tree_util import tree_map

from eci.agents import Agent, Candidate, Voter
from eci.perceptual import PerceptualModel
from eci.population import Population, PopulationConfig
from eci.world import World


@dataclass
class EnvConfig:
    """
    Centralized configuration for the simulation environment.

    Kept as a flat dataclass for backward compatibility. Internally,
    :class:`Environment` decomposes it into
    :class:`~eci.population.PopulationConfig`,
    :class:`~eci.world.WorldConfig` and
    :class:`~eci.perceptual.HGFConfig`.

    Attributes
    ----------
    num_voters : int
        Number of voting agents.
    num_candidates : int
        Number of candidate agents.
    num_preferences : int
        Number of preference dimensions.
    num_steps : int, optional
        Number of time steps in the simulation. Default is 362.
    scenario : int, optional
        Scenario identifier for input generation. Default is 2.
        Note: with `shock_pattern=None`, scenario 1 and 2 are equivalent.
    seed : int, optional
        Random seed for JAX agent generation. Default is 42.
        Does NOT control observation noise — see `obs_seed`.
    dispersion : float, optional
        Multiplier for the Gaussian noise added to observations
        (σ = 0.05 * dispersion * (obs_high - obs_low)). Default is 1.0.
    shock_pattern : {"phase", "sudden", "trend"} or None, optional
        Type of shock pattern injected in the observations. Only active
        when `scenario == 2`. Default is None.
    obs_seed : int or None, optional
        Seed for the numpy RNG used inside `generate_observations`.
        None → non-reproducible. Default is None.
    obs_low, obs_high : float, optional
        Output range for observations. Defaults to ``[0.0, 1.0]``.
    precision_state : float, optional
        Precision parameter for the HGF state nodes. Default is 100.0.
    tonic_volatility_mean : float, optional
        Mean of the tonic volatility distribution. Default is -2.0.
    tonic_volatility_std : float, optional
        Standard deviation of the tonic volatility distribution. Default is 0.01.
    """

    num_voters: int
    num_candidates: int
    num_preferences: int
    num_steps: int = 362
    scenario: int = 2
    seed: int = 42

    # Observation generation
    dispersion: float = 1.0
    shock_pattern: Optional[Literal["phase", "sudden", "trend"]] = None
    obs_seed: Optional[int] = None
    obs_low: float = 0.0
    obs_high: float = 1.0
    recover: bool = False

    # HGF Model Parameters
    precision_state: float = 100.0
    tonic_volatility_mean: float = -2.0
    tonic_volatility_std: float = 0.01


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Environment:
    """High-level simulation environment.

    Composes a :class:`Population`, a :class:`World` and a
    :class:`PerceptualModel`. All three are accessible as attributes
    (``env.population``, ``env.world``, ``env.perceptual``) for code
    that wants the new API; legacy attributes are preserved as proxies.

    Attributes
    ----------
    config : EnvConfig
        The flat config that was used to build the environment.
    population : Population
        Voter & candidate arrays (JAX pytree).
    world : World
        Observation stream (JAX pytree).
    perceptual : PerceptualModel
        HGF wrapper. Stateful; produces JAX pytrees on ``.run()``.

    Legacy attributes
    -----------------
    voters : list[Voter]
    candidates : list[Candidate]
    agents : list[Agent]
    input_data : jax.Array
    network : pyhgf.model.Network
    node_trajectories : Any, set after inference
    last_attributes : Any, set after inference
    preferences_idx : list[int], set after inference
    """

    def __init__(self, config: EnvConfig):
        """Build the population, world and perceptual model from ``config``.

        Parameters
        ----------
        config : EnvConfig
            Flat configuration for the whole simulation.
        """
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)

        # --- composable building blocks ------------------------------
        self.population = Population.random(
            PopulationConfig.from_env_config(config),
            key=self.key,
        )
        self.world = World.from_env_config(config)
        self.perceptual = PerceptualModel.from_env_config(config)

        self.voters: List[Voter] = self.population.as_voters(start_id=0)
        self.candidates: List[Candidate] = self.population.as_candidates(
            start_id=len(self.voters)
        )
        self.agents: List[Agent] = [*self.voters, *self.candidates]

        self.node_trajectories: Optional[Any] = None
        self.last_attributes: Optional[Any] = None
        self.preferences_idx: Optional[List[int]] = None
        self.winner_id: Optional[int] = None
        self.sim_result: Optional[Dict] = None

    @property
    def input_data(self) -> jax.Array:
        """Observation stream — proxy to ``self.world.observations``."""
        return self.world.observations

    @input_data.setter
    def input_data(self, value) -> None:
        """Override the observation stream with a manual array.

        Backward-compat shim for the ``env.input_data = manual_obs``
        pattern used in the tutorials: rebuilds ``self.world`` around the
        supplied array so the (frozen, pytree) :class:`World` invariant
        is preserved.
        """
        self.world = World(observations=jnp.asarray(value))

    @property
    def network(self):
        """The underlying HGF network — proxy to ``self.perceptual.network``."""
        return self.perceptual.network

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_multi_agent_inference(self) -> None:
        """Run HGF inference for the whole population (vmapped)."""
        result = self.perceptual.run(self.population, self.world)
        self.last_attributes = result["last_attributes"]
        self.node_trajectories = result["node_trajectories"]
        self.preferences_idx = result["preferences_idx"]
        for i, voter in enumerate(self.voters):
            voter.trajectory = tree_map(lambda x, _i=i: x[_i], self.node_trajectories)

    def _run_single_agent_inference(
        self,
        preference_means,
        preference_precisions,
        tonic_volatility,
        network=None,
    ):
        """Legacy single-agent inference path. Delegates to PerceptualModel.

        Kept for tests that mock-patch this method. New code should use
        ``self.perceptual.run(...)``.
        """
        return self.perceptual._run_one_agent(
            preference_means,
            preference_precisions,
            tonic_volatility,
            self.world.observations,
            network if network is not None else self.perceptual.network,
        )

    # ------------------------------------------------------------------
    # Simulation runners (unchanged signatures, unchanged semantics)
    # ------------------------------------------------------------------
    def run_one_simulation(self, func, key, *args, **kwargs) -> dict:
        """Run a single simulation using the provided function and key."""
        self.sim_result = func(self, key, *args, **kwargs)
        return self.sim_result

    def run_n_simulation(
        self,
        func,
        data,
        response_function,
        key,
        n_simulations: int,
        *args,
        **kwargs,
    ) -> Dict[int, Any]:
        """Run ``n_simulations`` simulations sequentially, returning a dict.

        TODO: replace the Python loop with ``jax.vmap`` over PRNG keys.
        """
        all_results: Dict[int, Any] = {}
        current_key = key
        for i in tqdm.tqdm(range(n_simulations), desc="Running Simulations"):
            current_key, subkey = jax.random.split(current_key)
            all_results[i] = func(data, response_function, subkey, *args, **kwargs)
        self.sim_result = all_results
        return self.sim_result

    def vote_outcome_over_time(
        self,
        response_function,
        voting_function,
        n_simulations: int = 100,
        metric: str = "win",
        key=None,
        **vote_kwargs,
    ) -> jax.Array:
        """Per-candidate election outcome at each timestep over the population.

        At every timestep ``t``, all voters vote using their belief **at that
        timestep** (preferences and candidate policies stay fixed); the voting
        rule aggregates the population's votes into an election. Over
        ``n_simulations`` stochastic runs we report, per candidate:

        - ``metric="win"`` (default): the **win frequency** — fraction of
          elections that candidate wins (its ``P(win)``).
        - ``metric="share"``: the mean **vote share** (fraction of votes /
          credits it receives).

        This is the population generalisation of the single-voter, per-timestep
        vote distribution shown in tutorial 5.

        Requires :meth:`_run_multi_agent_inference` to have been called first.

        Parameters
        ----------
        response_function:
            A :class:`~eci.decision.ResponseFunction`.
        voting_function:
            ``_vote_plurality`` or ``_vote_quadratic``.
        n_simulations:
            Stochastic elections averaged per timestep.
        metric:
            ``"win"`` (P(win) per candidate) or ``"share"`` (mean vote share).
        key:
            JAX PRNG key (defaults to ``PRNGKey(0)``).
        **vote_kwargs:
            Forwarded to ``voting_function`` (e.g. ``num_votes=None`` for the
            adaptive QV allocation, or ``budget=...``).

        Returns
        -------
        outcome : jax.Array, shape (n_candidates, n_steps)
            Per-candidate win frequency (``metric="win"``) or mean vote share
            (``metric="share"``) at each timestep. Columns sum to 1.
        """
        from eci.utils import _extract_env_data_vectorized

        if metric not in ("win", "share"):
            raise ValueError(f"metric must be 'win' or 'share', got {metric!r}")
        if self.node_trajectories is None or self.preferences_idx is None:
            raise RuntimeError(
                "Call _run_multi_agent_inference() before vote_outcome_over_time()."
            )
        if key is None:
            key = jax.random.PRNGKey(0)

        base_data = _extract_env_data_vectorized(self)
        n_candidates = base_data["candidates"]["mean"].shape[0]
        n_steps = self.input_data.shape[0]
        belief_means, belief_precisions = self._belief_trajectories()

        outcomes_by_timestep = []
        for timestep in range(n_steps):
            timestep_data = self._data_at_timestep(
                base_data,
                belief_means,
                belief_precisions,
                timestep,
            )
            simulation_keys = jax.random.split(
                jax.random.fold_in(key, timestep), n_simulations
            )
            election_results = self._run_election_batch(
                timestep_data,
                response_function,
                voting_function,
                simulation_keys,
                vote_kwargs,
            )
            outcome = self._summarize_election_results(
                election_results,
                metric,
                n_candidates,
                n_simulations,
            )
            outcomes_by_timestep.append(outcome)

        return jnp.stack(outcomes_by_timestep, axis=1)

    def _belief_trajectories(self) -> tuple[jax.Array, jax.Array]:
        """Stack HGF belief trajectories as ``(agent, step, preference)`` arrays."""
        preference_node_indices = self.preferences_idx
        node_trajectories = self.node_trajectories
        if preference_node_indices is None or node_trajectories is None:
            raise RuntimeError(
                "Belief trajectories are not available before inference."
            )

        belief_means = jnp.stack(
            [
                node_trajectories[node_idx]["expected_mean"]
                for node_idx in preference_node_indices
            ],
            axis=-1,
        )
        belief_precisions = jnp.stack(
            [
                node_trajectories[node_idx]["expected_precision"]
                for node_idx in preference_node_indices
            ],
            axis=-1,
        )
        return belief_means, belief_precisions

    @staticmethod
    def _data_at_timestep(
        base_data,
        belief_means: jax.Array,
        belief_precisions: jax.Array,
        timestep: int,
    ) -> dict:
        """Combine fixed election data with beliefs from one timestep."""
        return {
            "beliefs": {
                "mean": belief_means[:, timestep, :],
                "precision": belief_precisions[:, timestep, :],
            },
            "preferences": base_data["preferences"],
            "candidates": base_data["candidates"],
        }

    @staticmethod
    def _run_election_batch(
        timestep_data,
        response_function,
        voting_function,
        simulation_keys: jax.Array,
        vote_kwargs: dict,
    ):
        """Run a batch of stochastic elections for one timestep."""
        return jax.vmap(
            lambda simulation_key: voting_function(
                timestep_data,
                response_function,
                simulation_key,
                **vote_kwargs,
            )
        )(simulation_keys)

    @staticmethod
    def _summarize_election_results(
        election_results,
        metric: str,
        n_candidates: int,
        n_simulations: int,
    ) -> jax.Array:
        """Summarize an election batch as candidate win frequencies or vote shares."""
        if metric == "win":
            winners = election_results["winner"]
            return jnp.bincount(winners, length=n_candidates) / n_simulations

        votes_per_candidate = election_results["votes_per_candidate"].astype(
            jnp.float32
        )
        total_votes = jnp.sum(votes_per_candidate, axis=1, keepdims=True)
        vote_shares = votes_per_candidate / jnp.maximum(total_votes, 1e-9)
        return jnp.mean(vote_shares, axis=0)

    def _gather_agent_data(self):
        """Return ``(voter_means, voter_precisions, voter_volatilities)``.

        Now reads directly from ``self.population`` instead of iterating
        over the legacy ``self.voters`` list.
        """
        return (
            self.population.voter_means,
            self.population.voter_precisions,
            self.population.voter_volatilities,
        )
