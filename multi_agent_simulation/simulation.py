import pyhgf
from pyhgf.model import Network
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyhgf.math import binary_surprise, dirichlet_kullback_leibler

# Set global constraints for all plots
plt.rcParams['figure.figsize'] = [8, 10]  # Adjust height
plt.rcParams['figure.dpi'] = 100  # Resolution in DPI
plt.style.use('seaborn-v0_8-pastel')  # Use seaborn style

class MultiAgentSimulation:
    def __init__(self, n_steps=100, n_nodes=3, n_agents_per_population=5):
        """
        Initialize the MultiAgentSimulation with the given parameters.

        Args:
            n_steps (int): Number of time steps in the simulation. Defaults to 100.
            n_nodes (int): Number of nodes in the network. Defaults to 3.
            n_agents_per_population (int): Number of agents in each population. Defaults to 5.
        """
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.n_agents_per_population = n_agents_per_population
        self.agents = []
        self.observations = []

    def create_agent(self, tonic_volatility, preferences=None, beta_params=None):
        """
        Create a new agent with a given tonic volatility, preferences, and beta parameters.

        Args:
            tonic_volatility (float): The tonic volatility to set for the agent.
            preferences (numpy.ndarray): The preferences to set for the agent.
            beta_params (tuple): The beta distribution parameters for generating observations.

        Returns:
            Network: A configured Network object representing the agent.
        """
        agent = Network().add_nodes(kind="binary-state", n_nodes=self.n_nodes)
        for i in range(self.n_nodes):
            agent.add_nodes(value_children=i)

        # Set preferences and beta parameters for the top-level node
        if preferences is not None:
            agent.attributes[-1]["preference"] = preferences
        else:
            agent.attributes[-1]["preference"] = np.array([np.random.rand() for _ in range(self.n_nodes)])

        if beta_params is not None:
            agent.attributes[-1]["preference_beta"] = beta_params

        agent.attributes[2]["tonic_volatility"] = tonic_volatility + np.random.normal(0, 0.5)  # Add some noise to the tonic volatility
        agent.attributes[3]["tonic_volatility"] = tonic_volatility + np.random.normal(0, 0.5)  # Add some noise to the tonic volatility

        return agent

    def generate_observations(self, n_agents, node_beta_params, scenario=1, shock_pattern=None,
                            shock_time=None, recovery_time=None, trend_shape='linear'):
        """
        Generate observations for the given number of agents with different scenarios.

        Args:
            n_agents (int): Number of agents to generate observations for.
            node_beta_params (list): List of beta parameters for each node's phases.
            scenario (int): Scenario to use for observation generation (1 or 2).
            shock_pattern (str): Type of shock pattern to use in scenario 2 (None, 'phase', 'sudden', 'trend').
            shock_time (int): Time step at which shock occurs (for 'sudden' pattern).
            recovery_time (int): Time step at which recovery occurs (for scenario 2 patterns).
            trend_shape (str): Shape of the trend ('linear', 'quadratic', 'sigmoid').
        """
        np.random.seed(42)  # Use a fixed seed for reproducibility
        self.observations = []

        if scenario == 1:
            # Scenario 1: Stable observations from a beta distribution
            node_observations = []
            for node in range(self.n_nodes):
                # Get the beta parameters for this node
                if node < len(node_beta_params):
                    beta_params = node_beta_params[node][0]  # Use only the first phase parameters for stability
                else:
                    # Default values if not specified for this node
                    beta_params = (5, 1)

                # Generate stable observations for the entire duration
                complete_observation = np.random.beta(a=beta_params[0], b=beta_params[1], size=self.n_steps)
                node_observations.append(complete_observation)

        elif scenario == 2:
            if shock_pattern is None or shock_pattern == 'phase':
                # Scenario 2 with three phases: normal, shock, recovery
                if recovery_time is None:
                    # Divide time into three equal phases if recovery_time is not specified
                    phase1_end = self.n_steps // 3
                    phase2_end = 2 * self.n_steps // 3
                else:
                    if shock_time is None:
                        # If shock_time is not specified, assume shock happens at 1/3 of the time
                        shock_time = self.n_steps // 3
                    phase1_end = shock_time
                    phase2_end = recovery_time

                node_observations = []
                for node in range(self.n_nodes):
                    if node < len(node_beta_params):
                        phase1_params, phase2_params = node_beta_params[node]
                        # For recovery phase, we'll use the same parameters as phase 1 (return to normal)
                        phase3_params = phase1_params
                    else:
                        phase1_params = (5, 1)
                        phase2_params = (2, 2)
                        phase3_params = phase1_params

                    # Generate observations for each phase
                    observation_phase1 = np.random.beta(a=phase1_params[0], b=phase1_params[1], size=phase1_end)
                    observation_phase2 = np.random.beta(a=phase2_params[0], b=phase2_params[1],
                                                    size=phase2_end - phase1_end)
                    observation_phase3 = np.random.beta(a=phase3_params[0], b=phase3_params[1],
                                                    size=self.n_steps - phase2_end)

                    complete_observation = np.concatenate((observation_phase1, observation_phase2, observation_phase3))
                    node_observations.append(complete_observation)

            elif shock_pattern == 'sudden':
                # Scenario 2 with sudden shock and recovery
                if shock_time is None:
                    shock_time = self.n_steps // 3
                if recovery_time is None:
                    recovery_time = 2 * self.n_steps // 3
                elif recovery_time <= shock_time or recovery_time >= self.n_steps:
                    raise ValueError("recovery_time must be between shock_time and n_steps-1")

                node_observations = []
                for node in range(self.n_nodes):
                    if node < len(node_beta_params):
                        phase1_params, phase2_params = node_beta_params[node]
                        # For recovery phase, use phase1 parameters (return to normal)
                        phase3_params = phase1_params
                    else:
                        phase1_params = (5, 1)
                        phase2_params = (2, 2)
                        phase3_params = phase1_params

                    # Generate observations for each period
                    observation_pre_shock = np.random.beta(a=phase1_params[0], b=phase1_params[1], size=shock_time)
                    observation_post_shock = np.random.beta(a=phase2_params[0], b=phase2_params[1],
                                                        size=recovery_time - shock_time)
                    observation_recovery = np.random.beta(a=phase3_params[0], b=phase3_params[1],
                                                        size=self.n_steps - recovery_time)

                    complete_observation = np.concatenate((observation_pre_shock, observation_post_shock, observation_recovery))
                    node_observations.append(complete_observation)

            elif shock_pattern == 'trend':
                # Scenario 2 with trend and return to normal
                if recovery_time is None:
                    # If recovery_time is not specified, assume it's 2/3 of the way through
                    recovery_time = 2 * self.n_steps // 3

                node_observations = []
                for node in range(self.n_nodes):
                    if node < len(node_beta_params):
                        phase1_params, phase2_params = node_beta_params[node]
                        # For recovery phase, use phase1 parameters (return to normal)
                        phase3_params = phase1_params
                    else:
                        phase1_params = (5, 1)
                        phase2_params = (2, 2)
                        phase3_params = phase1_params

                    complete_observation = np.zeros(self.n_steps)

                    # Phase 1: Stable at initial parameters (before trend starts)
                    for t in range(recovery_time // 2):  # First half of recovery_time is phase 1
                        alpha = phase1_params[0]
                        beta_param = phase1_params[1]
                        complete_observation[t] = np.random.beta(a=alpha, b=beta_param)

                    # Phase 2: Trend from phase1 to phase2 parameters
                    trend_start = recovery_time // 2
                    trend_end = recovery_time

                    for t in range(trend_start, trend_end):
                        # Calculate progress through the trend phase
                        progress = (t - trend_start) / (trend_end - trend_start)

                        # Calculate weight based on time and trend shape
                        if trend_shape == 'linear':
                            weight = progress
                        elif trend_shape == 'quadratic':
                            weight = progress ** 2
                        elif trend_shape == 'sigmoid':
                            x = (progress - 0.5) * 10  # Scale and center
                            weight = 1 / (1 + np.exp(-x))  # Sigmoid function
                        else:
                            weight = progress  # Default to linear

                        # Interpolate between phase1 and phase2 parameters
                        alpha = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                        beta_param = phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                        complete_observation[t] = np.random.beta(a=alpha, b=beta_param)

                    # Phase 3: Transition back to normal (from phase2 to phase1 parameters)
                    recovery_start = recovery_time
                    recovery_end = self.n_steps

                    for t in range(recovery_start, recovery_end):
                        # Calculate progress through the recovery phase
                        progress = (t - recovery_start) / (recovery_end - recovery_start)

                        # Calculate weight (we'll reverse the trend)
                        if trend_shape == 'linear':
                            weight = 1 - progress
                        elif trend_shape == 'quadratic':
                            weight = (1 - progress) ** 2
                        elif trend_shape == 'sigmoid':
                            x = (progress - 0.5) * 10  # Scale and center
                            weight = 1 - (1 / (1 + np.exp(-x)))  # Inverted sigmoid
                        else:
                            weight = 1 - progress  # Default to linear

                        # Interpolate between phase2 and phase1 parameters (recovery)
                        alpha = phase2_params[0] * (1 - weight) + phase1_params[0] * weight
                        beta_param = phase2_params[1] * (1 - weight) + phase1_params[1] * weight
                        complete_observation[t] = np.random.beta(a=alpha, b=beta_param)

                    node_observations.append(complete_observation)

            else:
                raise ValueError("Invalid shock_pattern specified for scenario 2.")
        else:
            raise ValueError("Scenario must be 1 or 2.")

        # Stack observations horizontally (each column corresponds to a node)
        observations_matrix = np.column_stack(node_observations)

        # Assign the same observations to all agents
        for _ in range(n_agents):
            self.observations.append(observations_matrix.copy())


    def run_simulation(self, scenario=1, shock_pattern=None, shock_time=None, recovery_time=None, trend_shape='linear'):
        """
        Run the simulation with two populations, each with different tonic volatilities and consistent preferences within populations.

        Args:
            scenario (int): Scenario to use for observation generation (1 or 2).
            shock_pattern (str): Type of shock pattern to use in scenario 2 (None, 'phase', 'sudden', 'trend').
            shock_time (int): Time step at which shock occurs (for 'sudden' pattern).
            recovery_time (int): Time step at which recovery occurs (for scenario 2 patterns).
            trend_shape (str): Shape of the trend ('linear', 'quadratic', 'sigmoid').
        """
        n_agents = 2 * self.n_agents_per_population  # Total number of agents

        # Define tonic volatilities for each population
        volatility_1 = -3
        volatility_2 = -1

        # Define preferences for each population (same for all agents within a population)
        population_1_preferences = np.array([0.1, 0.4])  # Example preferences for Population 1
        population_2_preferences = np.array([0.8, 0.1])  # Example preferences for Population 2

        # Define beta parameters for each node's volatility phases
        # For each node, we define a tuple of phase parameters (phase1, phase2)
        node_beta_params = [
            ( (10, 2), (30, 30) ),  # Node 0: low then high volatility
            ( (30, 1), (20,100) ),  # Node 1: high then low volatility
            # Additional nodes will use default pattern
        ]

        # Generate observations for all agents using the node-specific beta parameters and selected scenario
        self.generate_observations(n_agents, node_beta_params, scenario, shock_pattern,
                                shock_time, recovery_time, trend_shape)

        # Rest of the method remains unchanged...
        beta_params_1 = (5, 1)  # Beta parameters for Population 1
        beta_params_2 = (2, 1)  # Beta parameters for Population 2

        population_1 = [self.create_agent(volatility_1, population_1_preferences, beta_params_1)
                    for _ in range(self.n_agents_per_population)]
        population_2 = [self.create_agent(volatility_2, population_2_preferences, beta_params_2)
                    for _ in range(self.n_agents_per_population)]

        self.agents = population_1 + population_2

        for i, agent in enumerate(self.agents):
            agent.input_data(input_data=self.observations[i], time_steps=np.ones(self.n_steps))

    def plot_trajectories(self):
        """
        Plot the expected mean trajectories for all agents across all nodes.
        Enhanced version with dynamic figure sizing and improved readability.
        """
        # Define line styles for agents within each population
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)),
                    (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1))]

        # Use a colormap with more distinct colors
        colormap = cm.get_cmap('tab20', 20)  # Supports up to 20 distinct populations

        # Adjust figure size dynamically based on number of nodes
        fig_height = max(10, 2 * self.n_nodes)
        plt.rcParams['figure.figsize'] = [12, fig_height]

        fig, axs = plt.subplots(self.n_nodes, 1, figsize=(12, fig_height))

        # Ensure axs is always iterable (for the case of 1 node)
        if self.n_nodes == 1:
            axs = [axs]

        for i in range(self.n_nodes):
            ax = axs[i] if self.n_nodes > 1 else axs[0]
            handles = []
            labels = []
            added_agents = set()

            for j, agent in enumerate(self.agents):
                population = (j // self.n_agents_per_population) + 1
                color = colormap(population - 1)
                agent_in_pop = j % self.n_agents_per_population
                linestyle = line_styles[agent_in_pop % len(line_styles)]

                line, = ax.plot(agent.node_trajectories[i]["expected_mean"],
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.5,
                            alpha=0.7)

                # Add to legend (limited to first 12 unique agent types)
                if len(added_agents) < 12 and (population, agent_in_pop) not in added_agents:
                    label = f'Agent {j+1} (Pop {population})'
                    handles.append(line)
                    labels.append(label)
                    added_agents.add((population, agent_in_pop))

            # Handle legend creation based on number of agents
            if len(self.agents) > 12:
                handles = []
                labels = []
                # Calculate number of populations dynamically
                n_populations = len(set([(j // self.n_agents_per_population) + 1
                                        for j in range(len(self.agents))]))
                for pop in range(1, n_populations + 1):
                    color = colormap(pop - 1)
                    handles.append(plt.Line2D([0], [0], color=color, lw=2))
                    labels.append(f'Population {pop}')
                ax.legend(handles=handles, labels=labels,
                    title='Populations', bbox_to_anchor=(1.05, 1), loc='upper left')
            elif handles:  # Only add legend if there are handles to show
                ax.legend(handles=handles, labels=labels,
                    title='Agents', bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_ylabel(f'Node {i+1}\nExpected Mean')
            ax.grid(True, linestyle='--', alpha=0.7)

        # Set xlabel only for the bottom subplot
        axs[-1].set_xlabel('Time Steps')
        plt.suptitle('Expected Mean Trajectories by Node and Agent', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        plt.show()


    def plot_observations(self):
        """
        Plot the observations for all agents across all nodes at each time step.
        """
        plt.figure(figsize=(8, 10))
        colormap = cm.get_cmap('tab10', 2)  # Two distinct colors for the two populations

        for i in range(self.n_nodes):
            plt.subplot(self.n_nodes, 1, i+1)
            handles = []
            labels = []

            for j, observations in enumerate(self.observations):
                # Determine the population of the agent
                population = 1 if j < self.n_agents_per_population else 2
                # Select color based on population
                color = colormap(population - 1)

                # Plot observations as dots for the current node
                plt.scatter(range(self.n_steps), observations[:, i], color=color, alpha=0.6, s=10)

                # Add to legend handles and labels if it's the first appearance of the population
                if population == 1 and j == 0:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Population {population}', markerfacecolor=color, markersize=10))
                    labels.append(f'Population {population}')
                elif population == 2 and j == self.n_agents_per_population:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Population {population}', markerfacecolor=color, markersize=10))
                    labels.append(f'Population {population}')

            plt.legend(handles=handles, labels=labels, title='Agents by Population')
            plt.ylabel('Observation Value')
            plt.xlabel('Time Steps')
            plt.title(f'Node {i+1} Observations')

        plt.suptitle('Observations Over Time by Population')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def calculate_surprise(expected_mean, preference):
        """
        Calculate the binary surprise based on the expected mean and preference.

        Args:
            expected_mean (float): The expected mean value.
            preference (float): The agent's preference value.

        Returns:
            float: The calculated binary surprise.
        """
        return binary_surprise(expected_mean, preference)

    # Exemple dans get_agent_surprises
    def get_agent_surprises(self, time_step):
        if time_step >= self.n_steps:
            raise ValueError("time_step exceeds the number of steps in the simulation")
        surprises = np.zeros((len(self.agents), self.n_nodes))
        for i, agent in enumerate(self.agents):
            for j in range(self.n_nodes):
                if j >= len(agent.node_trajectories):
                    continue  # Skip if node index is out of range
                if time_step >= len(agent.node_trajectories[j]["expected_mean"]):
                    continue  # Skip if time_step is out of range
                expected_mean = agent.node_trajectories[j]["expected_mean"][time_step]
                if j < len(agent.attributes[-1]["preference"]):
                    preference = agent.attributes[-1]["preference"][j]
                else:
                    preference = 0.5  # Default preference if index is out of range
                surprises[i, j] = self.calculate_surprise(expected_mean, preference)
        return surprises

    def plot_agent_data(self):
        n_agents = len(self.agents)
        time_step = self.n_steps - 1
        surprises = self.get_agent_surprises(time_step)

        fig, axs = plt.subplots(n_agents, 1, figsize=(10, 5 * n_agents))
        if n_agents == 1:
            axs = [axs]

        for i, agent in enumerate(self.agents):
            population = 1 if i < self.n_agents_per_population else 2
            agent_last_expected_mean = agent.node_trajectories[0]["expected_mean"][time_step]

            categories = []
            surprise_values = []
            preference_values = []

            for j in range(self.n_nodes):
                categories.append(f'Preference {j+1}')
                surprise_values.append(surprises[i, j])
                preference_values.append(agent.attributes[-1]["preference"][j])

            width = 0.35
            x = np.arange(len(categories))

            axs[i].bar(x - width/2, preference_values, width, label='Preference', color='blue', alpha=0.6)
            axs[i].bar(x + width/2, surprise_values, width, label='Surprise', color='red', alpha=0.6)
            axs[i].axhline(y=agent_last_expected_mean, color='green', linestyle='--', label='Last Expected Mean')

            axs[i].set_title(f'Agent {i+1} (Population {population}): Preferences, Surprise Values, and Last Expected Mean')
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(categories)
            axs[i].set_ylabel('Values')
            axs[i].legend()

        plt.tight_layout()
        plt.show()
