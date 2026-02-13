from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import tqdm
from jax import vmap
from jax.tree_util import Partial, tree_map
from pyhgf.model import Network

from eci.agents import Agent, Candidate, Voter
from eci.utils import generate_observations


@dataclass
class EnvConfig:
    """
    Centralized configuration for the simulation environment.

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
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    precision_state : float, optional
        Precision parameter for the HGF state nodes. Default is 10.0.
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

    # HGF Model Parameters
    precision_state: float = 10.0
    tonic_volatility_mean: float = -2.0
    tonic_volatility_std: float = 0.01


class Environment:
    """
    Simulation environment for the election scenario.

    Manages agent creation, HGF network configuration, and simulation execution.

    Attributes
    ----------
    config : EnvConfig
        Configuration object containing simulation parameters.
    key : jax.Array
        JAX PRNG key for random number generation.
    voters : List[Voter]
        List of voter agents.
    candidates : List[Candidate]
        List of candidate agents.
    agents : List[Agent]
        Combined list of all agents.
    network : Network
        The underlying PyHGF network model.
    input_data : jax.Array
        Observation data for the simulation steps.
    """

    def __init__(self, config: EnvConfig):
        """
        Initialize the simulation environment.

        Parameters
        ----------
        config : EnvConfig
            Configuration object containing all simulation parameters.
        """
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)

        # State containers
        self.voters: List[Voter] = []
        self.candidates: List[Candidate] = []
        self.agents: List[Agent] = []

        # Simulation artifacts
        self.node_trajectories: Optional[Any] = None
        self.preferences_idx: Optional[List[int]] = None
        self.winner_id: Optional[int] = None
        self.sim_result: Optional[Dict] = None

        # Initialization
        self._init_agents()
        self.network = self._setup_network()

        # Input data generation
        self.input_data = generate_observations(
            n_nodes=self.config.num_preferences,
            n_steps=self.config.num_steps,
            scenario=self.config.scenario,
        )

    def _setup_network(self) -> Network:
        """
        Configure and return the Hierarchical Gaussian Filter (HGF) network.

        Returns
        -------
        Network
            Configured PyHGF network instance.
        """
        network = Network(update_type="unbounded")

        # Add continuous state nodes
        network.add_nodes(
            kind="continuous-state",
            n_nodes=self.config.num_preferences,
            precision=self.config.precision_state,
            expected_precision=self.config.precision_state,
        )

        # Configure hierarchy
        for i in range(self.config.num_preferences):
            network.add_nodes(value_children=i)
            network.add_nodes(volatility_children=i)

        return network

    def _init_agents(self) -> None:
        """
        Create voters and candidates using vectorized operations.

        Generates random parameters for all agents in a single JAX call
        to improve initialization speed compared to iterative loops.
        """
        # Vectorized generation of random parameters
        self.key, k1, k2, k3, k4, k5, k6 = jax.random.split(self.key, 7)

        # Voter parameters
        v_means = jax.random.uniform(
            k1,
            shape=(self.config.num_voters, self.config.num_preferences),
            minval=0.0,
            maxval=2.0,
        )
        v_precs = jax.random.uniform(
            k2,
            shape=(self.config.num_voters, self.config.num_preferences),
            minval=0.4,
            maxval=1.0,
        )
        v_vols = (
            jax.random.normal(k3, shape=(self.config.num_voters,))
            * self.config.tonic_volatility_std
        ) + self.config.tonic_volatility_mean

        # Candidate parameters
        c_means = jax.random.uniform(
            k4,
            shape=(self.config.num_candidates, self.config.num_preferences),
            minval=0.0,
            maxval=2.0,
        )
        c_precs = jax.random.uniform(
            k5,
            shape=(self.config.num_candidates, self.config.num_preferences),
            minval=0.3,
            maxval=1.0,
        )

        # Instantiate Python objects
        next_id = 0

        # Create Voters
        for i in range(self.config.num_voters):
            voter = Voter(
                id=next_id,
                preferences={"mean": v_means[i], "precision": v_precs[i]},
                tonic_volatility=float(v_vols[i]),
            )
            self.voters.append(voter)
            next_id += 1

        # Create Candidates
        for i in range(self.config.num_candidates):
            candidate = Candidate(
                id=next_id,
                policy={"mean": c_means[i], "precision": c_precs[i]},
            )
            self.candidates.append(candidate)
            next_id += 1

        self.agents = []
        self.agents.extend(self.voters)
        self.agents.extend(self.candidates)

    def _gather_agent_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Gather data from Voter objects into JAX arrays for batch processing.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            A tuple containing (means, precisions, volatilities) as JAX arrays.
        """
        all_mus = jnp.array([v.preferences["mean"] for v in self.voters])
        all_pis = jnp.array([v.preferences["precision"] for v in self.voters])
        all_volatilities = jnp.array([v.tonic_volatility for v in self.voters])
        return all_mus, all_pis, all_volatilities

    def run_one_simulation(self, func, key, *args, **kwargs) -> dict:
        """Run a single simulation using the provided function and key."""
        self.sim_result = func(self, key, *args, **kwargs)
        return self.sim_result

    def run_n_simulation(
        self, func, key, n_simulations: int, *args, **kwargs
    ) -> Dict[int, Any]:
        """
        Run multiple simulations and aggregate the results.

        Parameters
        ----------
        func : callable
            The simulation function to execute.
        key : jax.Array
            Initial PRNG key.
        n_simulations : int
            Number of simulations to run.
        *args, **kwargs
            Additional arguments passed to `func`.

        Returns
        -------
        Dict[int, Any]
            Dictionary containing results from all simulation runs.
        """
        all_results = {}

        current_key = key
        for i in tqdm.tqdm(range(n_simulations), desc="Running Simulations"):
            current_key, subkey = jax.random.split(current_key)
            all_results[i] = func(self, subkey, *args, **kwargs)

        self.sim_result = all_results
        return self.sim_result

    def _run_single_agent_inference(self, mu, pi, tonic_volatility, network):
        """
        Prepare network and run inference for a single agent.

        Helper function designed to be used with `vmap`.

        Parameters
        ----------
        mu : jax.Array
            Preference means.
        pi : jax.Array
            Preference precisions.
        tonic_volatility : float
            Agent's tonic volatility.
        network : Network
            The HGF network instance.

        Returns
        -------
        Tuple
            Last attributes and node trajectories.
        """
        network.attributes[-1]["preferences"] = {"mean": mu, "precision": pi}
        preferences_idx = network.input_idxs
        for idx in preferences_idx:
            network.attributes[idx]["tonic_volatility"] = tonic_volatility

        network.input_data(input_data=self.input_data)

        return network.last_attributes, network.node_trajectories

    def _run_multi_agent_inference(self) -> None:
        """
        Execute inference for all agents in parallel using vmap.

        Updates the `trajectory` attribute of each Voter object.
        """
        all_mus, all_pis, all_volatilities = self._gather_agent_data()

        # Partial application to fix the network argument
        vmap_create_net = Partial(
            self._run_single_agent_inference, network=self.network
        )

        # Batch execution
        self.last_attributes, self.node_trajectories = vmap(vmap_create_net)(
            all_mus, all_pis, all_volatilities
        )

        self.preferences_idx = self.network.input_idxs

        # Distribute batched results back to individual agent objects
        for i, voter in enumerate(self.voters):
            voter.trajectory = tree_map(lambda x: x[i], self.node_trajectories)
