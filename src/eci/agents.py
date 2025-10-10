from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class Agent(ABC):
    """An abstract class for all agents in the simulation.

    This class provides the basic structure for any agent.

    Attributes
    ----------
    id : int
        The agent's unique identifier.
    """

    id: int

    @abstractmethod
    def step(self, env: Any) -> None:
        """Define the agent's action during a single simulation step.

        This method must be implemented by all subclasses. It contains the
        logic for what the agent does during its activation.

        Parameters
        ----------
        environment
            The environment in which the agent exists, providing access to
            global state and other agents if needed.
        """
        pass


@dataclass
class Voter(Agent):
    """Represents a voter agent in the simulation.

    Voters have a set of preferences and a volatility level, which are used
    to compute their voting decisions.

    Attributes
    ----------
    id : int
        The agent's unique identifier.
    preferences : Dict[str, Any]
        The voter's preferences, typically containing 'mean' and
        'precision' vectors.
    tonic_volatility : float
        A parameter representing the voter's baseline level of choice volatility.
    last_vote : Optional[int]
        The ID of the candidate the voter chose in the last step.
    last_softmax_probs : Optional[Dict[int, float]]
        A dictionary mapping candidate IDs to choice probabilities from the last step.
    last_dissatisfactions : Optional[Dict[int, float]]
        A dictionary mapping candidate IDs to dissatisfaction scores from the last step.
    traj : Optional[Any]
        Stores trajectory data from the last simulation step, if any.
    observation : Optional[Any]
        Stores observation data from the last simulation step, if any.
    """

    # Initialization
    preferences: Dict[str, Any]
    tonic_volatility: float
    budget: float = 100.0

    # Attribute for Theory of Mind
    perceived_outcome: Optional[np.ndarray] = None

    # State attributes (with default values, not needed at initialization)
    last_vote: Optional[Union[int, np.ndarray, List[int]]] = None
    last_softmax_probs: Optional[Dict[int, float]] = field(default_factory=dict)
    last_dissatisfactions: Optional[Dict[int, float]] = field(default_factory=dict)
    traj: Optional[Any] = None
    observation: Optional[Any] = None

    def step(self, env: Any) -> None:
        """Perform the voter's action for a step.

        Parameters
        ----------
        environment
            The environment in which the agent exists, providing access to
            candidate policies and other relevant information.

        Notes
        -----
        In this model, the core voting logic is handled externally by the
        vectorized JAX functions in the `Environment`. This method is a
        placeholder for any additional, non-vectorized actions a voter
        might take.
        """
        pass


@dataclass
class Candidate(Agent):
    """Represents a candidate agent who can be elected.

    Candidates have a policy platform that voters evaluate.

    Attributes
    ----------
    id : int
        The agent's unique identifier.
    policy : Dict[str, Any]
        The candidate's policy platform, typically including 'mean' and
        'precision' vectors.
    vote_count : int
        A simple counter for votes. Note: The official tally is managed
        by the `VotingSystem`.
    """

    # Required attribute
    policy: Dict[str, Any]

    # State attribute with a default value
    vote_count: int = 0

    def step(self, env: Any) -> None:
        """Perform the candidate's action for a step.

        Parameters
        ----------
        environment : Any
            The environment in which the agent exists, providing access to
            global state, voter data, and other agents.

        Notes
        -----
        This method could be used to implement dynamic behaviors, such as
        a candidate changing their policy platform in response to voter
        opinions (i.e., campaigning).
        """
        pass
