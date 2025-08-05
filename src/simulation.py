# simulation.py
from tqdm import tqdm
import jax

from agent import HGFAgent
from environement import Environment
from data_collector import DataCollector


class Simulation:
    """Orchestrates the running of multiple election simulations."""

    def __init__(
        self,
        sim_params: dict,
        agent_params: list,
        candidates: list,
        base_hgf_network,
        decision_strategy,
        voting_system,
        input_data,
    ):
        self.n_simulations = sim_params["n_simulations"]
        self.agent_params = agent_params
        self.base_hgf_network = base_hgf_network
        self.decision_strategy = decision_strategy
        self.voting_system = voting_system
        self.data_collector = DataCollector()
        self.environment = Environment(candidates, input_data)

    def run(self, key):
        """Launch the series of simulations."""
        agents = [
            HGFAgent(
                agent_id=i,
                base_hgf_network=self.base_hgf_network,
                decision_strategy=self.decision_strategy,
                preferences=params["preferences"],
                tonic_volatility=params["tonic_volatility"],
            )
            for i, params in enumerate(self.agent_params)
        ]

        # --- NEW LEARNING STEP ADDED HERE ---
        print("Running the initial learning phase for all agents...")
        for agent in tqdm(agents, desc="Agent Learning"):
            agent.learn(self.environment.input_data)
        print("Learning phase complete.")
        # --- END OF NEW STEP ---

        simulation_keys = jax.random.split(key, self.n_simulations)

        for i in tqdm(range(self.n_simulations), desc="Simulations"):
            # Now, run_election only performs JAX-compatible operations
            election_results = self.voting_system.run_election(
                agents, self.environment, simulation_keys[i]
            )
            self.data_collector.record_simulation(i + 1, election_results)

        return self.data_collector.get_dataframe()
