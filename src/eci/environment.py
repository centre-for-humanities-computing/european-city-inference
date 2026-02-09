from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax import vmap
from jax.tree_util import Partial
from pyhgf.model import Network

from eci.agents import Candidate, Voter
from eci.utils import generate_observations


class Environment:
    """Simulation environment for the election scenario."""

    def __init__(
        self,
        num_voters: int,
        num_candidates: int,
        num_preferences: int,
    ):
        """Initialize the simulation environment."""
        # attributes
        self.num_simulations = 1
        self.key = jax.random.PRNGKey(42)
        self.num_preferences = num_preferences
        self.next_agent_id = 1
        self.num_candidates = num_candidates
        self.node_trajectories = None
        self.preferences_idx = None
        self.voters: List[Voter] = []
        self.candidates: List[Candidate] = []
        self._create_voters(num_voters)
        self._create_candidates(num_candidates)
        self.agents = self.voters + self.candidates
        self.winner_id: Optional[int] = None
        self.last_round1_results: Optional[Dict[Any, Any]] = None
        self.last_round2_results: Optional[Dict[Any, Any]] = None
        self.winner_id = None
        self.df = None

        # model setup
        network = Network(update_type="unbounded")
        network.add_nodes(
            kind="continuous-state",
            n_nodes=self.num_preferences,
            precision=10,
            expected_precision=10,
        )
        for i in range(self.num_preferences):
            network.add_nodes(value_children=i)
            network.add_nodes(volatility_children=i)

        self.network = network

        # input data setup
        self.input_data = generate_observations(
            n_nodes=self.num_preferences, n_steps=362, scenario=2
        )

    def _get_new_agent_id(self) -> int:
        """Generate and returns a unique ID for a new agent."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id - 1

    def _create_voters(self, num_voters: int) -> None:
        """Create Voters with random preferences in a vectorized way."""
        for _ in range(num_voters):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            mean = jax.random.uniform(
                subkey1, shape=(self.num_preferences,), minval=0, maxval=2.0
            )
            precision = jax.random.uniform(
                subkey2, shape=(self.num_preferences,), minval=0.4, maxval=1.0
            )
            preferences = {"mean": mean, "precision": precision}
            tonic_volatility = np.random.normal(-2.0, 0.01)
            voter = Voter(
                id=self._get_new_agent_id(),
                preferences=preferences,
                tonic_volatility=tonic_volatility,
            )

            voter.vote_round_1 = []
            voter.vote_round_2 = []
            voter.softmax_probs_1 = []
            voter.softmax_probs_2 = []
            voter.dissatisfactions = []

            self.voters.append(voter)

    def _create_candidates(self, num_candidates: int) -> None:
        """Create Candidates with random policy platforms."""
        for i in range(num_candidates):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            mean = jax.random.uniform(
                subkey1, shape=(self.num_preferences,), minval=0, maxval=2.0
            )
            precision = jax.random.uniform(
                subkey2, shape=(self.num_preferences,), minval=0.3, maxval=1.0
            )
            policy_data = {"mean": mean, "precision": precision}
            candidate = Candidate(
                id=i + 1,
                policy=policy_data,
            )
            self.candidates.append(candidate)

    def _gather_agent_data(self) -> tuple:
        """Gathers data from all Voter objects into JAX arrays."""
        all_mus = jnp.array([v.preferences["mean"] for v in self.voters])
        all_pis = jnp.array([v.preferences["precision"] for v in self.voters])
        all_volatilities = jnp.array([v.tonic_volatility for v in self.voters])
        return all_mus, all_pis, all_volatilities

    # ok
    def run_one_simulation(self, func, key, *args, **kwargs) -> dict:
        """Run a single simulation using the provided function and key."""
        self.sim_result = func(self, key, *args, **kwargs)
        return self.sim_result

    # ok
    def run_n_simulation(self, func, key, n_simulations, *args, **kwargs) -> dict:
        """Run multiple simulations and aggregate the results."""
        # Dictionary to store the results of all n simulations
        all_results = {}

        # Loop n_simulations times
        for i in tqdm.tqdm((range(n_simulations)), desc="Running Simulations"):
            key, subkey = jax.random.split(key)
            # Run the simulation function once
            single_run_result = func(self, subkey, *args, **kwargs)

            # Store the result dict using the simulation index (i) as the key
            all_results[i] = single_run_result

        # Store the complete dictionary (of dictionaries) as the main sim_result
        self.sim_result = all_results
        # Return the dictionary containing all simulation results
        return self.sim_result

    # TODO: _update_agents(self)
    # TODO:  This has to be checked / tested and simplified for JAX
    def _run_single_agent_inference(self, mu, pi, tonic_volatility, network):
        """Prepare network for voting."""
        network.attributes[-1]["preferences"] = {"mean": mu, "precision": pi}
        preferences_idx = network.input_idxs
        for idx in preferences_idx:
            network.attributes[idx]["tonic_volatility"] = tonic_volatility
        network.input_data(input_data=self.input_data)

        return network.last_attributes, network.node_trajectories

    def _run_multi_agent_inference(self):
        """Initialize the network attributes and trajectories."""
        all_mus, all_pis, all_volatilities = self._gather_agent_data()
        vmap_create_net = Partial(
            self._run_single_agent_inference, network=self.network
        )
        self.last_attributes, self.node_trajectories = vmap(vmap_create_net)(
            all_mus, all_pis, all_volatilities
        )
        self.preferences_idx = self.network.input_idxs
        for agent_idx in range(self.voters.__len__()):
            self.agents[agent_idx].trajectory = self.node_trajectories[0]
