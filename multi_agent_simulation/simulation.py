from pyhgf.model import Network
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.cm as cm  
from pyhgf.math import binary_surprise
from scipy.stats import norm  
from scipy.special import expit

# Set global constraints for all plots
plt.rcParams['figure.figsize'] = [8, 10]  # Adjust height
plt.rcParams['figure.dpi'] = 100  # Resolution in DPI
plt.style.use('seaborn-v0_8-pastel')  # Use seaborn style for better-looking plots

class Agent:
    def __init__(self, simulation, tonic_volatility, preferences=None, beta_params=None, preference_normal=None):
        """
        Initialize an Agent object.

        Args:
            simulation: The simulation environment the agent belongs to.
            tonic_volatility: A parameter representing the baseline volatility of the agent's beliefs.
            preferences: The agent's preferences (optional).
            beta_params: Parameters for beta distribution (optional).
            preference_normal: Parameters for normal distribution of preferences (optional).
        """
        self.simulation = simulation
        self.tonic_volatility = tonic_volatility
        self.preferences = preferences
        self.beta_params = beta_params
        self.preference_normal = preference_normal
        self.network = self._create_network()

    def _create_network(self):
        """
        Create and return a network structure for the agent.

        Returns:
            A Network object with binary-state nodes and value children.
        """
        network = Network().add_nodes(kind="binary-state", n_nodes=self.simulation.n_nodes)
        for i in range(self.simulation.n_nodes):
            network.add_nodes(value_children=i)
        return network

    def configure(self):
        """
        Configure the agent's network with preferences and tonic volatility.

        Returns:
            The configured network object.
        """
        if self.preference_normal is None:
            mean = np.random.normal(loc=0.5, scale=0.2, size=self.simulation.n_nodes)
            precision = np.random.gamma(shape=2.0, scale=1.0, size=self.simulation.n_nodes)
            self.preference_normal = (mean, precision)
        else:
            mean, precision = self.preference_normal

        self.network.attributes[-1]["preference_normal"] = (mean, precision)

        if self.preferences is not None:
            self.network.attributes[-1]["preference"] = self.preferences
        else:
            std = 1.0 / np.sqrt(precision)
            sampled_preferences = np.random.normal(loc=mean, scale=std)
            self.network.attributes[-1]["preference"] = np.clip(sampled_preferences, 0, 1)

        if self.beta_params is not None:
            self.network.attributes[-1]["preference_beta"] = self.beta_params

        self.network.attributes[2]["tonic_volatility"] = self.tonic_volatility + np.random.normal(0, 0.5)
        self.network.attributes[3]["tonic_volatility"] = self.tonic_volatility + np.random.normal(0, 0.5)

        return self.network

class Voter(Agent):
    def __init__(self, simulation, tonic_volatility, preferences=None, beta_params=None, preference_normal=None):
        """
        Initialize a Voter object.

        Args:
            simulation: The simulation environment the voter belongs to.
            tonic_volatility: A parameter representing the baseline volatility of the voter's beliefs.
            preferences: The voter's preferences (optional).
            beta_params: Parameters for beta distribution (optional).
            preference_normal: Parameters for normal distribution of preferences (optional).
        """
        super().__init__(simulation, tonic_volatility, preferences, beta_params, preference_normal)
        self.voting_history = []  # To keep track of voting history
        self.party_affiliation = None  # Party affiliation (if any)

    def calculate_dissatisfaction(self):
        """
        Calculate the dissatisfaction of the voter based on their preferences and beliefs.

        Returns:
            A list of dissatisfaction values for each node.
        """
        dissatisfactions = []
        n_nodes = self.simulation.n_nodes

        mean_pref, precision_pref = self.network.attributes[-1]["preference_normal"]
        std_pref = 1 / np.sqrt(precision_pref)
        var_pref = std_pref ** 2

        for i in range(n_nodes):
            if i < len(self.network.node_trajectories):
                mu_belief = self.network.node_trajectories[i]["expected_mean"][-1]
                prec_belief = self.network.node_trajectories[i]["expected_precision"][-1]
                std_belief = 1 / np.sqrt(prec_belief)
                var_belief = std_belief ** 2

                kl = (
                    np.log(std_belief / std_pref[i])
                    + (var_pref[i] + (mean_pref[i] - mu_belief) ** 2) / (2 * var_belief)
                    - 0.5
                )
                dissatisfactions.append(kl)
            else:
                dissatisfactions.append(0)

        return dissatisfactions

    def vote(self, candidates):
        if not candidates:
            return None

        current_dissatisfaction = self.calculate_dissatisfaction()
        total_current_dissatisfaction = sum(current_dissatisfaction)

        best_candidate = None
        min_total_dissatisfaction = float('inf')

        # Display current dissatisfaction for each node
        print(f"\nVoter preferences: {self.network.attributes[-1]['preference']}")

        for i, candidate in enumerate(candidates):
            expected_dissatisfaction = []
            n_nodes = self.simulation.n_nodes

            mean_pref, precision_pref = self.network.attributes[-1]["preference_normal"]
            std_pref = 1 / np.sqrt(precision_pref)
            var_pref = std_pref ** 2

            candidate_mean_pref, candidate_precision_pref = candidate.network.attributes[-1]["preference_normal"]
            candidate_std_pref = 1 / np.sqrt(candidate_precision_pref)
            candidate_var_pref = candidate_std_pref ** 2

            # Display candidate preferences (for debugging)
            print(f"\nCandidate {i} preferences (mean, precision):")
            for node in range(n_nodes):
                if node < len(candidate_mean_pref):
                    print(f"  Node {node}: mean = {candidate_mean_pref[node]:.2f}, precision = {candidate_precision_pref[node]:.2f}")

            for node in range(n_nodes):
                if node < len(candidate_mean_pref):
                    kl = (
                        np.log(candidate_std_pref[node] / std_pref[node])
                        + (var_pref[node] + (mean_pref[node] - candidate_mean_pref[node]) ** 2) / (2 * candidate_var_pref[node])
                        - 0.5
                    )
                    expected_dissatisfaction.append(kl)
                else:
                    expected_dissatisfaction.append(0)

            total_expected_dissatisfaction = sum(expected_dissatisfaction)

            print(f"Total expected dissatisfaction with Candidate {i}: {total_expected_dissatisfaction:.2f}")

            if total_expected_dissatisfaction < min_total_dissatisfaction:
                min_total_dissatisfaction = total_expected_dissatisfaction
                best_candidate = candidate

        self.voting_history.append({
            'candidates': candidates,
            'chosen_candidate': best_candidate,
            'current_dissatisfaction': total_current_dissatisfaction,
            'expected_dissatisfaction_with_chosen': min_total_dissatisfaction if best_candidate else float('inf')
        })

        return best_candidate

class Candidate(Agent):
    def __init__(self, simulation, tonic_volatility, preferences=None, beta_params=None, preference_normal=None):
        """
        Initialize a Candidate object.

        Args:
            simulation: The simulation environment the candidate belongs to.
            tonic_volatility: A parameter representing the baseline volatility of the candidate's beliefs.
            preferences: The candidate's preferences (optional).
            beta_params: Parameters for beta distribution (optional).
            preference_normal: Parameters for normal distribution of preferences (optional).
        """
        super().__init__(simulation, tonic_volatility, preferences, beta_params, preference_normal)
        self.campaign_strategy = None  # Campaign strategy (if any)
        self.party_affiliation = None  # Party affiliation (if any)
        self.campaign_budget = 0  # Campaign budget

    def configure(self):
        """
        Configure the candidate's network.

        Returns:
            The configured network object.
        """
        network = super().configure()
        return network

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, simulation, tonic_volatility, preferences=None, beta_params=None, preference_normal=None):
        """
        Factory method to create an agent of a specific type.

        Args:
            agent_type: Type of agent to create ('voter' or 'candidate').
            simulation: The simulation environment the agent belongs to.
            tonic_volatility: A parameter representing the baseline volatility of the agent's beliefs.
            preferences: The agent's preferences (optional).
            beta_params: Parameters for beta distribution (optional).
            preference_normal: Parameters for normal distribution of preferences (optional).

        Returns:
            An instance of the specified agent type.

        Raises:
            ValueError: If an invalid agent type is provided.
        """
        if agent_type == "voter":
            return Voter(simulation, tonic_volatility, preferences, beta_params, preference_normal)
        elif agent_type == "candidate":
            return Candidate(simulation, tonic_volatility, preferences, beta_params, preference_normal)
        else:
            raise ValueError("Invalid agent type")

class MultiAgentSimulation:
    def __init__(self, n_steps=100, n_nodes=3, n_agents_per_population=5, n_candidates=None):
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.n_agents_per_population = n_agents_per_population
        self.n_candidates = n_candidates if n_candidates is not None else n_agents_per_population
        self.agents = []  # List to store agents
        self.agent_networks = []  # List to store the networks of agents
        self.observations = []


    def create_agent(self, agent_type, tonic_volatility, preferences=None, beta_params=None, preference_normal=None):
        """
        Create an agent of a specific type and add it to the simulation.

        Args:
            agent_type: Type of agent to create ('voter' or 'candidate').
            tonic_volatility: A parameter representing the baseline volatility of the agent's beliefs.
            preferences: The agent's preferences (optional).
            beta_params: Parameters for beta distribution (optional).
            preference_normal: Parameters for normal distribution of preferences (optional).

        Returns:
            The created agent object.
        """
        agent = AgentFactory.create_agent(agent_type, self, tonic_volatility, preferences, beta_params, preference_normal)
        agent_network = agent.configure()
        self.agents.append(agent)  # Store the agent
        self.agent_networks.append(agent_network)  # Store the agent's network
        return agent  # Return the agent rather than its network

    def generate_observations(self, n_agents, node_beta_params, scenario=1, shock_pattern=None,
                             shock_time=None, recovery_time=None, trend_shape='linear'):
        """
        Generate observations for the given number of agents with different scenarios.

        Args:
            n_agents: Number of agents.
            node_beta_params: Beta parameters for each node.
            scenario: Scenario type (1 or 2).
            shock_pattern: Pattern of shock ('phase', 'sudden', 'trend').
            shock_time: Time step at which shock occurs.
            recovery_time: Time step at which recovery starts.
            trend_shape: Shape of the trend ('linear', 'quadratic', 'sigmoid').
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
        n_voters_per_population = self.n_agents_per_population
        n_candidates = self.n_candidates
        total_agents = 2 * n_voters_per_population + n_candidates

        # Définir les volatilités pour chaque population
        volatility_pop1 = -3
        volatility_pop2 = -1
        volatility_candidates = -2  # Une volatilité différente pour les candidats

        # Définir les préférences pour chaque population
        population_1_preferences = np.array([0.1, 0.4])  # Exemple de préférences pour la Population 1
        population_2_preferences = np.array([0.8, 0.1])  # Exemple de préférences pour la Population 2

        # Définir les paramètres beta pour chaque population
        beta_params_pop1 = (5, 1)
        beta_params_pop2 = (2, 1)
        beta_params_candidates = (3, 1)

        # Définir les paramètres pour les nœuds
        node_beta_params = [
            ( (10, 2), (30, 30) ),  # Node 0: low then high volatility
            ( (30, 1), (20,100) ),  # Node 1: high then low volatility
        ]

        # Générer des observations pour tous les agents
        self.generate_observations(total_agents, node_beta_params, scenario, shock_pattern,
                                   shock_time, recovery_time, trend_shape)

        # Créer les agents
        # Population 1 : électeurs
        population_1 = [self.create_agent("voter", volatility_pop1, population_1_preferences, beta_params_pop1)
                        for _ in range(n_voters_per_population)]
        # Population 2 : électeurs
        population_2 = [self.create_agent("voter", volatility_pop2, population_2_preferences, beta_params_pop2)
                        for _ in range(n_voters_per_population)]

        # Create candidates
        candidates = []
        for i in range(n_candidates):
            # Définir les préférences pour les candidats
            mean = np.random.uniform(0, 1, size=self.n_nodes)  # Moyenne aléatoire entre 0 et 1 pour chaque nœud
            precision = np.random.gamma(shape=2.0, scale=1.0, size=self.n_nodes)  # Précision aléatoire pour chaque nœud
            preference_normal = (mean, precision)

            # Créer le candidat avec ces préférences
            candidate = self.create_agent("candidate", volatility_candidates,
                                          preferences=None,  # Les préférences seront générées à partir de preference_normal
                                          beta_params=beta_params_candidates,
                                          preference_normal=preference_normal)
            candidates.append(candidate)

            # Afficher les préférences générées pour chaque candidat (pour le débogage)
            print(f"Candidate {i} preferences (mean, precision):")
            for node in range(self.n_nodes):
                print(f"  Node {node}: mean = {mean[node]:.2f}, precision = {precision[node]:.2f}")

        self.agents = population_1 + population_2 + candidates

        # Assigner les observations à chaque agent
        for i, agent in enumerate(self.agents):
            self.agent_networks[i].input_data(input_data=self.observations[i], time_steps=np.ones(self.n_steps))

    def simulate_voting(self):
        n = self.n_agents_per_population
        total_voters = 2 * n
        voters = self.agents[:total_voters]
        candidates = self.agents[total_voters:]

        print("\nList of voters and candidates:")
        for i, agent in enumerate(self.agents):
            if i < total_voters:
                print(f"Agent {i}: Voter (Population {'1' if i < n else '2'})")
            else:
                print(f"Agent {i}: Candidate")

        votes = []
        for voter in voters:
            chosen_candidate = voter.vote(candidates)
            if chosen_candidate:
                votes.append(candidates.index(chosen_candidate))
                voter_index = voters.index(voter)
                candidate_index = candidates.index(chosen_candidate)
                population = '1' if voter_index < n else '2'
                print(f"Voter {voter_index} (Population {population}) voted for Candidate {candidate_index}")
            else:
                votes.append(None)
                voter_index = voters.index(voter)
                population = '1' if voter_index < n else '2'
                print(f"Voter {voter_index} (Population {population}) did not vote.")

        if candidates:
            vote_counts = [0] * len(candidates)
            for vote in votes:
                if vote is not None:
                    vote_counts[vote] += 1
            print("Vote counts:", vote_counts)
        else:
            print("No candidates available for voting.")

    def plot_trajectories(self):
        # Déterminer le nombre de votants
        n_voters_per_population = self.n_agents_per_population
        total_voters = 2 * n_voters_per_population

        # Définir les styles de ligne pour les agents au sein de chaque population
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)),
                       (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1))]

        # Utiliser une colormap avec plus de couleurs distinctes
        colormap = cm.get_cmap('tab20', 20)  # Supports jusqu'à 20 populations distinctes

        # Ajuster la taille de la figure dynamiquement en fonction du nombre de nœuds
        fig_height = max(10, 2 * self.n_nodes)
        plt.rcParams['figure.figsize'] = [12, fig_height]

        fig, axs = plt.subplots(self.n_nodes, 1, figsize=(12, fig_height))

        # S'assurer que axs est toujours itérable (pour le cas d'un seul nœud)
        if self.n_nodes == 1:
            axs = [axs]

        for i in range(self.n_nodes):
            ax = axs[i] if self.n_nodes > 1 else axs[0]
            handles = []
            labels = []
            added_agents = set()

            # Parcourir uniquement les votants
            for j, agent in enumerate(self.agents[:total_voters]):
                population = 1 if j < n_voters_per_population else 2
                color = colormap(population - 1)
                agent_in_pop = j % n_voters_per_population
                linestyle = line_styles[agent_in_pop % len(line_styles)]

                line, = ax.plot(self.agent_networks[j].node_trajectories[i]["expected_mean"],
                                color=color,
                                linestyle=linestyle,
                                linewidth=1.5,
                                alpha=0.7)

                # Ajouter à la légende (limité aux 12 premiers types d'agents uniques)
                if len(added_agents) < 12 and (population, agent_in_pop) not in added_agents:
                    label = f'Agent {j+1} (Pop {population})'
                    handles.append(line)
                    labels.append(label)
                    added_agents.add((population, agent_in_pop))

            # Gérer la création de la légende en fonction du nombre d'agents
            if len(self.agents[:total_voters]) > 12:
                handles = []
                labels = []
                # Calculer le nombre de populations dynamiquement
                n_populations = len(set([(j // n_voters_per_population) + 1
                                       for j in range(len(self.agents[:total_voters]))]))
                for pop in range(1, n_populations + 1):
                    color = colormap(pop - 1)
                    handles.append(plt.Line2D([0], [0], color=color, lw=2))
                    labels.append(f'Population {pop}')
                ax.legend(handles=handles, labels=labels,
                          title='Populations', bbox_to_anchor=(1.05, 1), loc='upper left')
            elif handles:  # Ajouter la légende uniquement s'il y a des handles à afficher
                ax.legend(handles=handles, labels=labels,
                          title='Agents', bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_ylabel(f'Node {i+1}\nExpected Mean')
            ax.grid(True, linestyle='--', alpha=0.7)

        # Définir xlabel uniquement pour le sous-graphe du bas
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

    def plot_agent_data(self):
        """
        Plot agent-specific data such as preferences, surprise values, and last expected mean.
        """
        n_agents = len(self.agents)
        time_step = self.n_steps - 1
        surprises = self.get_agent_surprises(time_step)

        fig, axs = plt.subplots(n_agents, 1, figsize=(10, 5 * n_agents))
        if n_agents == 1:
            axs = [axs]

        for i, agent in enumerate(self.agents):
            population = 1 if i < self.n_agents_per_population else 2
            agent_last_expected_mean = self.agent_networks[i].node_trajectories[0]["expected_mean"][time_step]

            categories = []
            surprise_values = []
            preference_values = []

            # Obtenez les préférences de l'agent
            preferences = agent.network.attributes[-1].get("preference", [])
            if not isinstance(preferences, (np.ndarray, list)):
                preferences = []
            preferences = np.array(preferences) if isinstance(preferences, list) else preferences

            # Déterminez le nombre de dimensions de préférence disponibles
            n_prefs = len(preferences) if hasattr(preferences, '__len__') else 0

            # Déterminez le nombre de nœuds à afficher (le minimum entre n_nodes et la taille de surprises[i])
            n_nodes_to_plot = min(self.n_nodes, surprises.shape[1] if i < surprises.shape[0] else 0)

            for j in range(n_nodes_to_plot):
                categories.append(f'Preference {j+1}')
                if j < surprises.shape[1] and i < surprises.shape[0]:
                    surprise_values.append(surprises[i, j])
                else:
                    surprise_values.append(0)  # Valeur par défaut si l'index est hors limites

                if j < n_prefs:
                    preference_values.append(preferences[j])
                else:
                    preference_values.append(0)  # Valeur par défaut si l'index est hors limites

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

    @staticmethod
    def calculate_surprise(expected_mean, preference):
        """
        Calculate the binary surprise based on the expected mean and preference.

        Args:
            expected_mean: The expected mean value.
            preference: The preference value.

        Returns:
            The calculated surprise value.
        """
        return binary_surprise(expected_mean, preference)

    def get_agent_surprises(self, time_step):
        """
        Calculate surprise values for all agents at a given time step.

        Args:
            time_step: The time step to calculate surprises for.

        Returns:
            A numpy array of surprise values for all agents and nodes.

        Raises:
            ValueError: If the time_step exceeds the number of steps in the simulation.
        """
        if time_step >= self.n_steps:
            raise ValueError("time_step exceeds the number of steps in the simulation")
        surprises = np.zeros((len(self.agent_networks), self.n_nodes))
        for i, agent_network in enumerate(self.agent_networks):
            agent = self.agents[i]
            for j in range(self.n_nodes):
                if j >= len(agent_network.node_trajectories):
                    continue  # Skip if node index is out of range
                if time_step >= len(agent_network.node_trajectories[j]["expected_mean"]):
                    continue  # Skip if time_step is out of range
                expected_mean = agent_network.node_trajectories[j]["expected_mean"][time_step]
                preferences = agent.network.attributes[-1].get("preference", [])
                if j < len(preferences):
                    preference = preferences[j]
                else:
                    preference = 0.5  # Default preference if index is out of range
                surprises[i, j] = self.calculate_surprise(expected_mean, preference)
        return surprises

    @staticmethod
    def plot_preference_normals_all_agents(simulation, sigmoid=False, plot_beliefs=False):
        """
        Plot the normal distribution (or sigmoid-transformed normal distribution) of preferences
        for each agent in the simulation.agents list. One subplot per agent.

        Args:
            simulation: The simulation object containing agents.
            sigmoid: If True, apply sigmoid transformation to the normal distribution.
            plot_beliefs: If True, also plot beliefs.
        """
        n_agents = len(simulation.agents)
        n_cols = 3
        n_rows = (n_agents + n_cols - 1) // n_cols

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for idx, agent in enumerate(simulation.agents):
            mean_pref, precision_pref = agent.network.attributes[-1]["preference_normal"]
            std_pref = 1 / np.sqrt(precision_pref)
            n_dims = len(mean_pref)

            ax = plt.subplot(n_rows, n_cols, idx + 1)

            for i in range(n_dims):
                x_pref = np.linspace(mean_pref[i] - 4*std_pref[i], mean_pref[i] + 4*std_pref[i], 500)
                pdf_pref = norm.pdf(x_pref, loc=mean_pref[i], scale=std_pref[i])

                if not sigmoid:
                    ax.plot(x_pref, pdf_pref, label=f'Pref Dim {i}: N({mean_pref[i]:.2f}, {std_pref[i]**2:.3f})')
                else:
                    y_pref = expit(x_pref)
                    pdf_y_pref = pdf_pref / (y_pref * (1 - y_pref))
                    ax.plot(y_pref, pdf_y_pref, label=f'Pref Dim {i}: Sigmoid N({mean_pref[i]:.2f}, {std_pref[i]**2:.3f})')

            if plot_beliefs:
                mean_belief = agent.network.node_trajectories[1]["expected_mean"][-1]
                precision_belief = agent.network.node_trajectories[1]["expected_precision"][-1]
                std_belief = 1 / np.sqrt(precision_belief)

                x_belief = np.linspace(mean_belief - 4*std_belief, mean_belief + 4*std_belief, 500)
                pdf_belief = norm.pdf(x_belief, loc=mean_belief, scale=std_belief)

                if not sigmoid:
                    ax.plot(x_belief, pdf_belief, label=f'Belief: N({mean_belief:.2f}, {std_belief**2:.3f})', linestyle='--')
                else:
                    y_belief = expit(x_belief)
                    pdf_y_belief = pdf_belief / (y_belief * (1 - y_belief))
                    ax.plot(y_belief, pdf_y_belief, label=f'Belief: Sigmoid N({mean_belief:.2f}, {std_belief**2:.3f})', linestyle='--')

            ax.set_title(f"Agent {idx}")
            ax.set_xlabel("Value")
            ax.set_ylabel("PDF Density")
            ax.legend(fontsize='small')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_kl_divergence_preferences(simulation, node_idx=1):
        """
        Calculate the KL divergence between the preference distribution (normal)
        and the beliefs of each agent for a given node.

        Args:
            simulation: The simulation object containing agents.
            node_idx: The index of the node to calculate KL divergence for.

        Returns:
            A list of KL divergence values for each agent.
        """
        kl_divergences = []

        for agent in simulation.agents:
            # Real preferences
            mu_1, prec_1 = agent.network.attributes[-1]["preference_normal"]
            mu_1 = np.array(mu_1)
            prec_1 = np.array(prec_1)
            std_1 = 1 / np.sqrt(prec_1)
            var_1 = std_1 ** 2

            # Agent's beliefs
            mu_2 = agent.network.node_trajectories[node_idx]["expected_mean"][-1]
            prec_2 = agent.network.node_trajectories[node_idx]["expected_precision"][-1]

            # Ensure they are arrays, even if they are scalars
            mu_2 = np.atleast_1d(mu_2)
            prec_2 = np.atleast_1d(prec_2)
            std_2 = 1 / np.sqrt(prec_2)
            var_2 = std_2 ** 2

            # KL divergence between independent univariate normals
            kl = np.sum(
                np.log(std_2 / std_1)
                + (var_1 + (mu_1 - mu_2) ** 2) / (2 * var_2)
                - 0.5
            )

            kl_divergences.append(kl)

        return kl_divergences
