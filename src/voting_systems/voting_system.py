# voting_simulation/voting_systems/voting_system.py

from abc import ABC, abstractmethod
from typing import List, Dict
from agents.voter import Voter
from agents.candidate import Candidate

class VotingSystem(ABC):
    """Interface for voting systems (Strategy Pattern)."""
    @abstractmethod
    def execute_vote(self, voters: List[Voter], candidates: List[Candidate]) -> Dict[Candidate, int]:
        """Execute a voting round and return the results."""
        pass
