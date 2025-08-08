import jax
import numpy as np
from scipy.stats import norm, halfnorm
from typing import List, Tuple
from .agent import Agent
from pyhgf.model import Network
from utils.voting_logic import get_votes 

class Voter(Agent):
    """
    Represents a voter as an autonomous agent with its own HGF network.
    """
    
    # Make sure this __init__ method is present and correct
    def __init__(self, id: int, n_preferences: int):
        """
        Builds a Voter object by generating its network and preferences.
        """
        # 1. Basic initialization (ID and JAX key)
        super().__init__(id)
        self.key = jax.random.PRNGKey(id)

        # 2. Creation and configuration of the internal HGF network
        self.network = Network()
        self.network.add_nodes(kind="binary-state", n_nodes=n_preferences)
        for i in range(n_preferences):
            self.network.add_nodes(value_children=i)

        # 3. Generation and assignment of unique preferences
        mus = norm.rvs(loc=2, scale=1, size=n_preferences)
        pis = halfnorm.rvs(loc=0, scale=1, size=n_preferences)
        
        self.preferences = {
            "mean": np.array(mus),
            "precision": np.array(pis)
        }
        self.network.attributes[-1]["preferences"] = self.preferences

    def decide_vote(self, candidates: List['Candidate']) -> int:
        """
        Calls the JAX voting logic to decide whom to vote for.
        """
        self.key, subkey = jax.random.split(self.key)
        
        network_state = {
            'attributes': self.network.attributes,
            'edges': self.network.edges,
            'node_trajectories': {},  # To be filled with real data
            'input_idxs': self.network.input_idxs
        }
        
        formatted_proposals = [(c.proposal['mean'], c.proposal['precision']) for c in candidates]

        chosen_candidate_index = get_votes(
            key=subkey,
            attributes=network_state['attributes'],
            edges=network_state['edges'],
            node_trajectories=network_state['node_trajectories'],
            input_idxs=network_state['input_idxs'],
            candidates=formatted_proposals,
        )
        return int(chosen_candidate_index)
