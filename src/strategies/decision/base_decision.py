from abc import ABC, abstractmethod

class DecisionStrategy(ABC):
    """Classe de base abstraite pour la stratégie de décision d'un agent."""

    @abstractmethod
    def decide(self, agent, environment, candidates: list, key) -> int:
        """
        Prend une décision de vote.
        """
        pass