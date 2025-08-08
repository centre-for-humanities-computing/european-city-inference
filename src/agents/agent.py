# agents/agent.py

class Agent:
    """Base class for all agents in the simulation."""
    def __init__(self, id: int):
        self.id = id
        self.preferences = {}

    def __repr__(self) -> str:
        """Return a clear string representation of the object."""
        return f"{self.__class__.__name__}(id={self.id})"
