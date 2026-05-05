from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Agent:
    """An abstract class for all agents in the simulation.

    This class provides the basic structure for any agent.

    Attributes
    ----------
    id :
        The agent's unique identifier.
    """

    id: int


@dataclass
class Voter(Agent):
    """Represents a voter agent in the simulation.

    Voters have a set of preferences and a volatility level, which are used
    to compute their voting decisions.

    Attributes
    ----------
    id :
        The agent's unique identifier.
    preferences :
        The voter's preferences, typically containing 'mean' and
        'precision' vectors.
    tonic_volatility :
        A parameter representing the voter's baseline level of choice volatility.
    budget :
        The voter's budget for influencing their decision, by default 100.0.
    perceived_outcome :
        The voter's perception of the election outcome, used for Theory of Mind,
        by default None.
    vote_round_1 :
        The voter's choice in the first round of voting, by default None.
    vote_round_2 :
        The voter's choice in the second round of voting, by default None.
    softmax_probs_1 :
        The softmax probabilities for the first round of voting.
    softmax_probs_2 :
        The softmax probabilities for the second round of voting.
    dissatisfactions :
        The voter's dissatisfaction levels.
    trajectory :
        The voter's trajectory data, by default None.
    observation:
        The voter's observation data, by default None.
    """

    # Initialization
    preferences: Dict[str, Any]
    tonic_volatility: float

    # Attribute for Theory of Mind
    perceived_outcome: Optional[np.ndarray] = None

    # State attributes
    vote_round_1: Optional[List[int]] = field(default_factory=list)
    vote_round_2: Optional[List[int]] = field(default_factory=list)
    softmax_probs_1: Optional[List[int]] = field(default_factory=list)
    softmax_probs_2: Optional[List[int]] = field(default_factory=list)
    dissatisfactions: Optional[List[int]] = field(default_factory=list)
    trajectory: Optional[Any] = None
    observation: Optional[Any] = None


@dataclass
class Candidate(Agent):
    """Represents a candidate agent who can be elected.

    Candidates have a policy platform that voters evaluate.

    Attributes
    ----------
    id :
        The agent's unique identifier.
    policy :
        The candidate's policy including 'mean' and
        'precision' vectors.
    vote_count :
        Counter for votes.
    """

    policy: Dict[str, Any]

    # State attribute with a default value
    vote_count: int = 0
