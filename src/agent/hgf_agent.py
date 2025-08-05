# agents/hgf_agent.py
import copy
import numpy as np

class HGFAgent:
    """Represents an agent using a model HGF for learning."""
    
    def __init__(self, agent_id: int, base_hgf_network, decision_strategy, preferences: tuple, tonic_volatility: float):
        self.id = agent_id
        self.preferences = preferences
        self.tonic_volatility = tonic_volatility
        self.decision_strategy = decision_strategy
        self.hgf_network = copy.deepcopy(base_hgf_network)
        self._configure_network()
        
    def _configure_network(self):
        """Applies specific parameters to this agent's HGF network."""
        for i in range(len(self.preferences), 2 * len(self.preferences)):
             if 'tonic_volatility' in self.hgf_network.attributes[i]:
                self.hgf_network.attributes[i]['tonic_volatility'] = self.tonic_volatility

    def learn(self, input_data):
        """Makes the agent learn by providing data."""
        self.hgf_network.input_data(input_data=input_data)
        
    # NEW HELPER METHOD
    def get_decision_parameters(self):
        """Extracts all necessary parameters for a JAX-compatible decision."""
        n_prefs = len(self.preferences)
        mu_belief = np.array([self.hgf_network.node_trajectories[i + n_prefs]["expected_mean"][-1] for i in range(n_prefs)])
        prec_belief = np.array([self.hgf_network.node_trajectories[i + n_prefs]["expected_precision"][-1] for i in range(n_prefs)])
        
        mean_pref = np.array([p[0] for p in self.preferences])
        precision_pref = np.array([1.0 / (p[1]**2) for p in self.preferences])
        
        return mu_belief, prec_belief, mean_pref, precision_pref
        
    def vote(self, environment, candidates: list, key) -> int:
        """This method will no longer be used by the vmap logic."""
        return self.decision_strategy.decide(self, environment, candidates, key)