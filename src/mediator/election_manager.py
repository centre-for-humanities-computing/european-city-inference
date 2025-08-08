# mediator/election_manager.py

import pandas as pd
import jax
from jax import vmap
from typing import List, Dict
from agents.voter import Voter
from agents.candidate import Candidate
from voting_systems.voting_system import VotingSystem
from utils.voting_logic import get_votes  

class ElectionManager:
    """
    Orchestrates the entire electoral process, from voting to results.
    Acts as a Mediator between agents and the voting system.
    """
    def __init__(
        self,
        voters: List[Voter],
        candidates: List[Candidate],
        voting_system: VotingSystem
    ):
        self.voters = voters
        self.candidates = candidates
        self.voting_system = voting_system
        self.key = jax.random.PRNGKey(42)  # Main JAX random key
        self.results = None  # Will store the results DataFrame

    def run_election(self) -> Candidate:
        """
        Executes a full election cycle.
        
        Returns:
            The winning Candidate object.
        """
        print("--- Start of the election ---")
        
        # 1. Prepare voter-specific data for JAX
        # NOTE: This part is conceptual. You'll need to adapt it to how you obtain
        # the 'attributes', 'edges', and 'node_trajectories' for each voter.
        # Let's assume each voter's network holds this state.
        voter_networks = [v.network for v in self.voters]
        
        # Create a subkey for each voter to ensure different random choices
        voter_keys = jax.random.split(self.key, len(self.voters))

        # 2. Collect votes using JAX's `vmap` for high performance
        # `vmap` can apply `get_votes` to all voters' data in parallel.
        # This is highly efficient but requires `get_votes` and its inputs
        # to be structured correctly for batching.
        print(f"Collecting votes from {len(self.voters)} voters...")
        
        # For simplicity here, let's use a standard loop first.
        # vmap is a more advanced optimization.
        votes = []
        for i, voter in enumerate(self.voters):
            # This is a conceptual call; you'll need to pass the real network state
            network_state = { 
                'attributes': voter.network.attributes,
                'edges': voter.network.edges,
                'node_trajectories': {},  # Populate with real trajectory data
                'input_idxs': voter.network.input_idxs
            }
            formatted_candidates = [(c.preferences['mu'], c.preferences['sigma']) for c in self.candidates]
            
            # Each voter uses their unique key
            chosen_candidate_idx = get_votes(
                voter_keys[i], 
                network_state,
                formatted_candidates
            )
            votes.append(int(chosen_candidate_idx))
        
        # 3. Tally the votes using the chosen voting strategy
        print("Tallying votes...")
        # The voting system needs the list of votes and candidates
        vote_counts = self.voting_system.tally(votes, self.candidates)

        # 4. Store detailed results in a DataFrame
        self.results = pd.DataFrame({
            'candidate_id': [c.id for c in self.candidates],
            'vote_count': [vote_counts.get(c, 0) for c in self.candidates]
        }).sort_values(by='vote_count', ascending=False)
        
        print("\n--- Election Results ---")
        print(self.results)

        # 5. Declare the winner
        winner_id = self.results.iloc[0]['candidate_id']
        winner = self.candidates[winner_id]
        
        print(f"\nThe winner is Candidate {winner.id}!")
        return winner
