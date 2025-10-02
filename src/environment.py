from typing import Any, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from pyhgf.model import Network

from agents import Agent, Candidate, Voter
from analysis import DataCollector
from core_logic import individual_vote
from utils import generate_observations
from voting_systems import VotingSystem


class Scheduler:
    """A scheduler to manage the activation order of agents.

    This scheduler activates agents in a pre-determined sequence. It is a
    "simultaneous" or "synchronous" scheduler, meaning all agents are
    activated once per step, in the order they are provided.

    Attributes
    ----------
    agents : list[Agent]
        The list of all agent objects managed by the scheduler.
    step_count : int
        The number of steps the scheduler has executed.
    """

    def __init__(self, agents: Sequence[Agent]):
        """Initialize the scheduler.

        Parameters
        ----------
        agents : list[Agent]
            A list of agents to be activated in each step. The order of
            activation is determined by the order of this list.
        """
        self.agents = agents
        self.step_count = 0

    def step(self, environment) -> None:
        """Activates all agents for one simulation step.

        This method iterates through the list of agents and calls the `step`
        method on each one, passing the environment. After all agents have
        been activated, the internal step counter is incremented.

        Parameters
        ----------
        environment : Any
            The simulation environment, which is passed to each agent's
            `step` method.
        """
        for agent in self.agents:
            agent.step(environment)
        self.step_count += 1


class Environment:
    """Orchestrates an agent-based simulation of a voting process.

    This class is the main controller for the simulation. It is responsible
    for creating and managing agents (Voters and Candidates), setting up the
    simulation components, and running the step-by-step execution of the model.
    It heavily utilizes the JAX library to perform efficient, vectorized
    computations for the voting process.

    Parameters
    ----------
    num_voters : int
        The number of voter agents to create.
    num_candidates : int
        The number of candidate agents to create.
    num_preferences : int
        The dimensionality of the preference/policy space.
    voting_system : VotingSystem
        An instantiated voting system object that will be used to count votes.
    scenario : int, optional
        An identifier for the scenario to run, used for data generation.
        Default is 1.

    Attributes
    ----------
    key : jax.random.PRNGKey
        The master JAX random key for ensuring reproducibility.
    voters : list[Voter]
        The list of all Voter agents in the simulation.
    candidates : list[Candidate]
        The list of all Candidate agents in the simulation.
    scheduler : Scheduler
        The scheduler object that manages agent activation.
    datacollector : DataCollector
        The object responsible for collecting data at each step.
    vmap_get_votes_fn : callable
        A pre-compiled, vectorized JAX function for efficiently computing
        the votes of all agents simultaneously.
    winner_id : int or None
        The ID of the winning candidate from the most recent step.
    """

    def __init__(
        self,
        num_voters: int,
        num_candidates: int,
        num_preferences: int,
        voting_system: "VotingSystem",
        scenario: int = 1,
        rounds: int = 2,
        use_theory_of_mind: bool = False,
    ):
        """Initialize the simulation environment."""
        self.key = jax.random.PRNGKey(42)
        self.num_preferences = num_preferences
        self.next_agent_id = 0
        self.rounds = rounds
        self.num_candidates = num_candidates
        # --- Create Agents ---
        self.voters: List[Voter] = []
        self.candidates: List[Candidate] = []
        self._create_voters(num_voters)
        self._create_candidates(num_candidates)
        self.agents = self.voters + self.candidates
        self.use_theory_of_mind = use_theory_of_mind
        self.winner_id: Optional[int] = None
        self.last_round1_results: Optional[Dict[Any, Any]] = None
        self.last_round2_results: Optional[Dict[Any, Any]] = None
        # Starts as a uniform distribution (no prior knowledge)
        self.public_poll = jnp.ones(self.num_candidates) / self.num_candidates

        # --- JAX Pre-computation and Setup ---
        # This section prepares and pre-compiles the core JAX function
        # to avoid compilation overhead during the simulation run.
        print("Pre-compiling JAX function...")
        self.input_data = generate_observations(
            n_nodes=self.num_preferences, n_steps=500, scenario=scenario
        )
        network_template = Network()
        network_template.add_nodes(kind="binary-state", n_nodes=self.num_preferences)
        for i in range(self.num_preferences):
            network_template.add_nodes(value_children=i)

        candidate_list = [
            (c.policy["mean"], c.policy["precision"]) for c in self.candidates
        ]

        def get_votes_fn(
            mus, pis, tonic_volatility, key, mask, perceived_outcome, budget
        ):
            return individual_vote(
                mus=mus,
                pis=pis,
                tonic_volatility=tonic_volatility,
                key=key,
                network=network_template,
                candidates=candidate_list,
                n_preferences=self.num_preferences,
                input_data=self.input_data,
                mask=mask,  # The mask is now a dynamic argument
                voting_system=voting_system.name,
                average_proportions_vector=perceived_outcome,
                budget=budget,
            )

        self.vmap_get_votes_fn = vmap(get_votes_fn, in_axes=(0, 0, 0, 0, None, 0, 0))  #

        # self.vmap_get_votes_fn = vmap(get_votes_fn)
        print("Compilation complete.")

        # --- Simulation Components ---
        self.voting_system = voting_system
        self.scheduler = Scheduler(self.agents)
        self.datacollector = DataCollector()
        self.winner_id = None
        self.last_round1_results = None
        self.last_round2_results = None

    def _update_public_poll(self, vote_counts: dict) -> None:
        """Update the public poll with the latest election results."""
        if not self.use_theory_of_mind or not vote_counts:
            return

        total_votes = sum(vote_counts.values())
        if total_votes == 0:
            return

        # Create a vector of vote proportions from the results dict
        candidate_ids = [c.id for c in self.candidates]
        self.public_poll = jnp.array(
            [vote_counts.get(cid, 0) / total_votes for cid in candidate_ids]
        )
        print(f"Public poll updated: {self.public_poll}")

    def _get_new_agent_id(self) -> int:
        """Generate and returns a unique ID for a new agent."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

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
            tonic_volatility = np.random.normal(-3.0, 1.0)
            voter = Voter(
                id=self._get_new_agent_id(),  # <-- Corrected argument name
                preferences=preferences,
                tonic_volatility=tonic_volatility,
            )
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
            policy_data = {"mean": mean, "precision": precision}  # Renamed for clarity
            candidate = Candidate(
                id=self._get_new_agent_id(),  # <-- Corrected argument name
                policy=policy_data,  # <-- Corrected argument name
            )
            self.candidates.append(candidate)

    def _check_preference_diversity(self):
        means = jax.numpy.stack([voter.preferences["mean"] for voter in self.voters])
        distances = jax.numpy.mean(jax.numpy.std(means, axis=0))
        return distances > 0.5  # Seuil arbitraire, à ajuster

    def _gather_agent_data(self) -> tuple:
        """Gathers data from all Voter objects into JAX arrays."""
        all_mus = jnp.array([v.preferences["mean"] for v in self.voters])
        all_pis = jnp.array([v.preferences["precision"] for v in self.voters])
        all_volatilities = jnp.array([v.tonic_volatility for v in self.voters])
        all_budgets = jnp.array([v.budget for v in self.voters])  # <-- ADD THIS LINE
        return all_mus, all_pis, all_volatilities, all_budgets  # <-- RETURN IT

    def _scatter_results(self, results: tuple) -> None:
        """Distributes results from the vectorized computation.

        Parameters
        ----------
        results : tuple
            A tuple containing the outputs from the JAX computation, such as
            votes, probabilities, dissatisfactions, and trajectories.
        """
        votes, softmax_probs, dissatisfactions, node_trajectories = results

        # Get the voting system name for conditional logic
        system_name = self.voting_system.name

        for i, voter in enumerate(self.voters):
            if "Ranking" in system_name or "Quadratic" in system_name:
                voter.last_vote = votes[i]
            else:
                voter.last_vote = int(votes[i])

            voter.last_softmax_probs = {
                cid: prob for cid, prob in enumerate(softmax_probs[i])
            }
            voter.last_dissatisfactions = {
                cid: diss for cid, diss in enumerate(dissatisfactions[i])
            }
            voter.traj = node_trajectories

    def _distribute_poll_to_voters(self) -> None:
        """Update the `perceived_outcome` attribute on each Voter object.

        Useful for inspection, but not necessary for the simulation.
        """
        if not self.use_theory_of_mind:
            return

        # Convert the JAX array to a NumPy array for storage
        poll_to_distribute = np.array(self.public_poll)

        for voter in self.voters:
            voter.perceived_outcome = poll_to_distribute

    def step(self) -> None:
        """Execute a full election process."""
        if self.rounds == 2:
            self._run_two_round_election()
        else:
            self._run_single_round_election()

    def _run_single_round_election(self) -> None:
        """Execute a standard, single-round election."""
        all_mus, all_pis, all_volatilities, all_budgets = (
            self._gather_agent_data()
        )  # <-- GET BUDGETS
        self.key, *agent_keys = jax.random.split(self.key, len(self.voters) + 1)

        # Prepare the ToM data to be passed to the JAX function
        perceived_outcomes = jnp.tile(self.public_poll, (len(self.voters), 1))

        mask = jnp.ones(len(self.candidates))

        # Pass the `perceived_outcomes` to the vmapped function
        results = self.vmap_get_votes_fn(
            all_mus,
            all_pis,
            all_volatilities,
            jnp.array(agent_keys),
            mask,
            perceived_outcomes,
            all_budgets,
        )

        self._scatter_results(results)

        self.scheduler.step(self)
        self.winner_id, vote_counts = self.voting_system.counting_votes(
            self.voters, self.candidates
        )
        self.last_round1_results = vote_counts
        self.last_round2_results = None

        # Update the poll for the next step
        self._update_public_poll(vote_counts)
        print(f"Winner is Candidate ID: {self.winner_id}.")

    def _run_two_round_election(self) -> None:
        """Execute a two-round election process."""
        all_mus, all_pis, all_volatilities, all_budgets = (
            self._gather_agent_data()
        )  # <-- GET BUDGETS
        self.key, *agent_keys = jax.random.split(self.key, len(self.voters) + 1)
        agent_keys_array = jnp.array(agent_keys)

        perceived_outcomes = jnp.tile(self.public_poll, (len(self.voters), 1))

        # --- ROUND 1 ---
        print("--- Round 1 ---")
        mask_round1 = jnp.ones(len(self.candidates))
        results_round1 = self.vmap_get_votes_fn(
            all_mus,
            all_pis,
            all_volatilities,
            agent_keys_array,
            mask_round1,
            perceived_outcomes,
            all_budgets,
        )

        self._scatter_results(results_round1)

        # Count votes after round 1
        _, vote_counts_round1 = self.voting_system.counting_votes(
            self.voters, self.candidates
        )
        self.last_round1_results = vote_counts_round1
        print(f"Round 1 Results: {vote_counts_round1}")

        sorted_candidates = sorted(
            vote_counts_round1.items(), key=lambda item: item[1], reverse=True
        )

        if len(sorted_candidates) < 2:
            print("Not enough candidates for a second round. Winner is from round 1.")
            self.winner_id = sorted_candidates[0][0] if sorted_candidates else -1
            self.scheduler.step(self)
            return

        finalist_ids = [cid for cid, _ in sorted_candidates[:2]]
        print(
            f"Finalists for Round 2: Candidates {finalist_ids[0]} and {finalist_ids[1]}"
        )

        # --- ROUND 2 ---
        print("\n--- Round 2 ---")
        candidate_id_to_index = {c.id: i for i, c in enumerate(self.candidates)}
        finalist_indices = [candidate_id_to_index[fid] for fid in finalist_ids]
        mask_round2 = (
            jnp.zeros(len(self.candidates)).at[jnp.array(finalist_indices)].set(1.0)
        )

        results_round2 = self.vmap_get_votes_fn(
            all_mus,
            all_pis,
            all_volatilities,
            agent_keys_array,
            mask_round2,
            perceived_outcomes,
            all_budgets,
        )

        self._scatter_results(results_round2)

        self.scheduler.step(self)

        self.winner_id, final_counts = self.voting_system.counting_votes(
            self.voters, self.candidates
        )
        self.last_round2_results = final_counts

        self._update_public_poll(final_counts)

        print(f"Round 2 Results: {final_counts}")
        print(f"Final Winner is Candidate ID: {self.winner_id}.")

    def run(self, num_steps: int, progress_callback=None) -> None:
        """Run the simulation for a specified number of steps."""
        print("Starting simulation...")
        for i in range(num_steps):
            print(f"--- Step {i + 1}/{num_steps} ---")
            self.step()
            self.datacollector.collect(self)
            self._distribute_poll_to_voters()

            # Update external progress bar if provided
            if progress_callback is not None:
                progress_callback(i + 1, num_steps)

        print("Simulation finished.")
