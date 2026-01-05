from typing import Any, Counter, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
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
        self.num_simulations = 1
        self.key = jax.random.PRNGKey(42)
        self.num_preferences = num_preferences
        self.next_agent_id = 0
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

        # network setup
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
        self.df = None
        self.input_data = generate_observations(
            n_nodes=self.num_preferences, n_steps=362, scenario=2
        )
        self.winner_id = None

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
        for _ in range(num_candidates):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            mean = jax.random.uniform(
                subkey1, shape=(self.num_preferences,), minval=0, maxval=2.0
            )
            precision = jax.random.uniform(
                subkey2, shape=(self.num_preferences,), minval=0.3, maxval=1.0
            )
            policy_data = {"mean": mean, "precision": precision}
            candidate = Candidate(
                id=self._get_new_agent_id(),
                policy=policy_data,
            )
            self.candidates.append(candidate)

    def _gather_agent_data(self) -> tuple:
        """Gathers data from all Voter objects into JAX arrays."""
        all_mus = jnp.array([v.preferences["mean"] for v in self.voters])
        all_pis = jnp.array([v.preferences["precision"] for v in self.voters])
        all_volatilities = jnp.array([v.tonic_volatility for v in self.voters])
        return all_mus, all_pis, all_volatilities

    def run_one_simulation(self, func, key, *args, **kwargs) -> dict:
        """Run a single simulation using the provided function and key."""
        self.sim_result = func(self, key, *args, **kwargs)
        return self.sim_result

    def run_n_simulation(self, func, key, n_simulations, *args, **kwargs) -> dict:
        """Run multiple simulations and aggregate the results."""
        # Dictionary to store the results of all n simulations
        all_results = {}

        # Loop n_simulations times
        for i in range(n_simulations):
            key, subkey = jax.random.split(key)
            # Run the simulation function once
            single_run_result = func(self, subkey, *args, **kwargs)

            # Store the result dict using the simulation index (i) as the key
            all_results[i] = single_run_result

        # Store the complete dictionary (of dictionaries) as the main sim_result
        self.sim_result = all_results
        # Return the dictionary containing all simulation results
        return self.sim_result

    def _update_agents(self) -> None:
        """Update agents with the results from the simulations."""
        for simulation_number in self.sim_result.keys():
            sim_data = self.sim_result[simulation_number]
            if "vote_round_1" in sim_data:
                for agent_idx in range(len(self.voters)):
                    # Create a local reference for cleaner code
                    voter = self.voters[agent_idx]

                    # 1. Vote Round 1
                    if voter.vote_round_1 is None:
                        voter.vote_round_1 = []
                    voter.vote_round_1.append(
                        self.sim_result[simulation_number]["vote_round_1"][agent_idx]
                    )

                    # 2. Vote Round 2
                    if voter.vote_round_2 is None:
                        voter.vote_round_2 = []
                    voter.vote_round_2.append(
                        self.sim_result[simulation_number]["vote_final_round_2"][
                            agent_idx
                        ]
                    )

                    # 3. Softmax Probs 1
                    if voter.softmax_probs_1 is None:
                        voter.softmax_probs_1 = []
                    voter.softmax_probs_1.append(
                        self.sim_result[simulation_number]["softmax_probs_round_1"][
                            agent_idx
                        ]
                    )

                    # 4. Softmax Probs 2
                    if voter.softmax_probs_2 is None:
                        voter.softmax_probs_2 = []
                    voter.softmax_probs_2.append(
                        self.sim_result[simulation_number][
                            "softmax_probs_final_round_2"
                        ][agent_idx]
                    )

            elif "vote_matrix" in sim_data:
                for agent_idx in range(len(self.voters)):
                    # Create a local reference for cleaner code
                    voter = self.voters[agent_idx]

                    # 1. Vote Round 1
                    if voter.vote_matrix is None:
                        voter.vote_round_1 = []
                    voter.vote_round_1.append(
                        self.sim_result[simulation_number]["vote_matrix"][agent_idx]
                    )

                    # 5. Dissatisfactions
                    if voter.dissatisfactions is None:
                        voter.dissatisfactions = []
                    voter.dissatisfactions.append(
                        self.sim_result[simulation_number]["dissatisfaction"][agent_idx]
                    )

    def create_network(self, mu, pi, tonic_volatility, network):
        """Prepare network for voting."""
        network.attributes[-1]["preferences"] = {"mean": mu, "precision": pi}
        preferences_idx = [
            network.edges[idx].value_parents[0] for idx in network.input_idxs
        ]
        for idx in preferences_idx:
            network.attributes[idx]["tonic_volatility"] = tonic_volatility
        network.input_data(input_data=self.input_data)

        return network.last_attributes, network.node_trajectories

    # TODO: Could be another name like running observations
    def initialize_network(self):
        """Initialize the network attributes and trajectories."""
        all_mus, all_pis, all_volatilities = self._gather_agent_data()
        vmap_create_net = Partial(self.create_network, network=self.network)
        self.last_attributes, self.node_trajectories = vmap(vmap_create_net)(
            all_mus, all_pis, all_volatilities
        )
        self.preferences_idx = [
            self.network.edges[idx].value_parents[0] for idx in self.network.input_idxs
        ]
        for agent_idx in range(self.voters.__len__()):
            self.agents[agent_idx].trajectory = self.node_trajectories[0]

    def get_winners(self, vote_list, top_n):
        """Determine the top N winners from a list of votes."""
        clean_votes = [v.item() if hasattr(v, "item") else v for v in vote_list]
        counter = Counter(clean_votes)
        top_candidates = counter.most_common(top_n)
        winners = [candidat for candidat, nb_voix in top_candidates]
        return winners

    def create_data_frame(self):
        """DataFrame summarizing the simulation results."""
        results_list = []
        for simulation_idx in range(self.num_simulations):
            votes_r1 = [voter.vote_round_1[simulation_idx] for voter in self.voters]
            votes_r2 = [voter.vote_round_2[simulation_idx] for voter in self.voters]
            top2_r1 = self.get_winners(votes_r1, 2)
            top1_r2 = self.get_winners(votes_r2, 1)
            results_list.append(
                {
                    "simulation_id": simulation_idx,
                    "vote_round_1": votes_r1,
                    "vote_round_2": votes_r2,
                    "winners_round_1": top2_r1,
                    "winner_round_2": top1_r2[0] if top1_r2 else None,
                }
            )
        return pd.DataFrame(results_list)
