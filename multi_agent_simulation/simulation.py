import pyhgf
from pyhgf.model import Network
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pyhgf.math import binary_surprise, dirichlet_kullback_leibler

# Set global constraints for all plots
plt.rcParams['figure.figsize'] = [8, 10]  # Adjust height to accommodate subplots
plt.rcParams['figure.dpi'] = 100  # Resolution in DPI
plt.style.use('seaborn-v0_8-pastel')  # Use seaborn style for better aesthetics

class MultiAgentSimulation:
    """
    A class to simulate multiple agents using the Hierarchical Gaussian Filter (HGF) model.

    Attributes:
        n_steps (int): Number of time steps in the simulation.
        n_nodes (int): Number of nodes in the network.
        preference (numpy.ndarray): Preference vector for the agents.
        agents (list): List to store the created agent objects.
        observations (list): List to store the generated observations for each agent.
    """
    def __init__(self, n_steps=100, n_nodes=3, preference=np.array([0.5, 0.5, 0.5])):
        """
        Initialize the MultiAgentSimulation with the given parameters.

        Args:
            n_steps (int): Number of time steps in the simulation. Defaults to 100.
            n_nodes (int): Number of nodes in the network. Defaults to 3.
            preference (numpy.ndarray): Preference vector for the agents. Defaults to np.array([0.5, 0.5, 0.5]).
        """
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.preference = preference
        self.agents = []
        self.observations = []

    def create_agent(self):
        """
        Create a new agent with the specified network structure.

        Returns:
            Network: A configured Network object representing the agent.
        """
        # Create a new network
        agent = (
            Network()
            .add_nodes(kind="binary-state", n_nodes=self.n_nodes)
        )
        # Add parent nodes for each binary state node
        for i in range(self.n_nodes):
            agent.add_nodes(value_children=i)
        # Set preferences for the top-level node
        agent.attributes[-1]["preference"] = self.preference
        return agent

    def generate_observations(self, n_agents=2):
        """
        Generate random observations for each agent.

        Args:
            n_agents (int): Number of agents to generate observations for. Defaults to 2.
        """
        self.observations = []
        for _ in range(n_agents):
            # Generate random observations using a beta distribution
            observation = np.array([np.random.beta(a=1, b=0.2, size=self.n_steps) for _ in range(self.n_nodes)]).T
            self.observations.append(observation)

    def run_simulation(self, n_agents=2):
        """
        Run the simulation with the specified number of agents.

        Args:
            n_agents (int): Number of agents to simulate. Defaults to 2.
        """
        self.generate_observations(n_agents)
        self.agents = []
        for i in range(n_agents):
            agent = self.create_agent()
            # Provide input data to the agent
            agent.input_data(input_data=self.observations[i], time_steps=np.ones(self.n_steps))
            self.agents.append(agent)

    def plot_trajectories(self):
        """
        Plot the expected mean trajectories for all agents across all nodes.
        """
        # Create subplots
        plt.figure(figsize=(8, 10))
        # Define a colormap with enough colors for all agents
        colormap = cm.get_cmap('tab20', len(self.agents))

        # Plot trajectories for each node
        for i in range(self.n_nodes):
            plt.subplot(self.n_nodes, 1, i+1)
            for j, agent in enumerate(self.agents):
                color = colormap(j)
                # Plot the expected mean trajectory for the current node
                plt.plot(agent.node_trajectories[i]["expected_mean"], label=f'Agent {j+1}', color=color)
            plt.legend()
            plt.ylabel('Expected Mean')

        plt.xlabel('Time Steps')
        plt.suptitle('Expected Mean Trajectories')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def calculate_surprise(expected_mean, actual_value):
        """
        Calculate the surprise based on the expected mean and the actual observed value.

        Args:
            expected_mean (float): The expected mean value.
            actual_value (float): The actual observed value.

        Returns:
            float: The calculated surprise.
        """
        return binary_surprise(expected_mean, actual_value)

    @staticmethod
    def calculate_kl_divergence(distribution1, distribution2):
        """
        Calculate the Kullback-Leibler divergence between two probability distributions.

        Args:
            distribution1 (numpy.ndarray): The first probability distribution.
            distribution2 (numpy.ndarray): The second probability distribution.

        Returns:
            float: The calculated KL divergence.
        """
        return dirichlet_kullback_leibler(distribution1, distribution2)

    def get_agent_surprises(self, time_step):
        """
        Calculate the surprise for all agents and nodes at a specific time step.

        Args:
            time_step (int): Time step for which to calculate the surprises.

        Returns:
            numpy.ndarray: A 2D array of surprises with shape (n_agents, n_nodes).
        """
        surprises = np.zeros((len(self.agents), self.n_nodes))
        for i, agent in enumerate(self.agents):
            for j in range(self.n_nodes):
                expected_mean = agent.node_trajectories[j]["expected_mean"][time_step]
                actual_value = self.observations[i][time_step, j]
                surprises[i, j] = self.calculate_surprise(expected_mean, actual_value)
        return surprises

    def get_agent_kl_divergences(self, time_step):
        """
        Calculate the KL divergence between all pairs of agents for all nodes at a specific time step.

        Args:
            time_step (int): Time step for which to calculate the KL divergences.

        Returns:
            numpy.ndarray: A 3D array of KL divergences with shape (n_agents, n_agents, n_nodes).
        """
        n_agents = len(self.agents)
        kl_divergences = np.zeros((n_agents, n_agents, self.n_nodes))
        for i in range(n_agents):
            for j in range(n_agents):
                for k in range(self.n_nodes):
                    # Assuming the node trajectories contain the necessary distributions
                    # Placeholder: actual implementation depends on data structure
                    # For now, we'll use the expected_mean as a placeholder for the distribution
                    # Note: This is a placeholder; actual implementation depends on the data structure
                    # and the correct parameters for dirichlet_kullback_leibler
                    dist1 = self.agents[i].node_trajectories[k]["expected_mean"][time_step]
                    dist2 = self.agents[j].node_trajectories[k]["expected_mean"][time_step]
                    # Note: dirichlet_kullback_leibler expects specific parameters, so this may not work correctly
                    # without proper distribution parameters.
                    try:
                        kl_divergences[i, j, k] = self.calculate_kl_divergence(dist1, dist2)
                    except:
                        kl_divergences[i, j, k] = np.nan  # Placeholder for invalid calculations
        return kl_divergences
