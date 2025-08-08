# voting_simulation/voting_systems/majority_voting.py

from typing import List, Dict
from agents.voter import Voter
from agents.candidate import Candidate
from .voting_system import VotingSystem

class MajorityVoting(VotingSystem):
    """One-round majority voting implementation."""
    def execute_vote(self, voters: List[Voter], candidates: List[Candidate]) -> Dict[Candidate, int]:
        results = {candidate: 0 for candidate in candidates}
        for voter in voters:
            chosen_candidate = voter.vote(candidates)
            if chosen_candidate:
                results[chosen_candidate] += 1
        return results
