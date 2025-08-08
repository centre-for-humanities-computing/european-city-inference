from .agent import Agent
import numpy as np
from scipy.stats import norm, halfnorm
from typing import List, Tuple 

class Candidate(Agent):
    """
    Represents a candidate and contains the logic to generate a list 
    of random candidates.
    """
    
    # Constructor that uses "Tuple"
    def __init__(self, id: int, preferences_data: Tuple[np.ndarray, np.ndarray]):
        """Initialize a single Candidate object from specific data."""
        super().__init__(id)
        mus, sigmas = preferences_data
        self.preferences = {'mu': mus, 'sigma': sigmas}
    
    # Factory that uses "List"
    @classmethod
    def create_random_list(cls, n_candidates: int, n_preferences: int) -> List['Candidate']:
        """
        Class method acting as a factory to create a list
        of candidates with random preferences.
        """
        mu_sigma = 1.0
        sigma_scale = 1.0
        
        candidate_list = []
        for i in range(n_candidates):
            mus = norm.rvs(loc=2, scale=mu_sigma, size=n_preferences)
            sigmas = halfnorm.rvs(scale=sigma_scale, size=n_preferences)
            new_candidate = cls(id=i, preferences_data=(mus, sigmas))
            candidate_list.append(new_candidate)
            
        return candidate_list
