from abc import ABC, abstractmethod


class VotingSystem(ABC):
    """Classe de base abstraite pour un système de vote."""

    @abstractmethod
    def run_election(self, agents: list, environment, key) -> dict:
        """
        Orchestre une élection complète et retourne les résultats finaux.
        """
        pass
