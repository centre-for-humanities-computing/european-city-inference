"""Top-level package for the voting simulation project.

This package provides agent definitions, voting systems, simulation
environment, analysis tools, and utility functions.
"""

__version__ = "0.1.0"

# Core agent classes
from .agents import Agent, Candidate, Voter

# Analysis tools
from .analysis import DataCollector, SimulationVisualizer

# Core logic functions
from .core_logic import (
    get_votes,
    init_preferences,
    kl_divergence,
    total_dissatisfaction_per_candidate,
)

# Environment
from .environment import Environment, Scheduler

# Utility functions
from .utils import generate_candidates, generate_observations

# Voting systems
from .voting_systems import (
    PluralityVoting,
    QuadraticVoting,
    QuadraticVotingBudget,
    RankingVoting,
    VotingSystem,
)

__all__ = [
    # Agents
    "Agent",
    "Voter",
    "Candidate",
    # Environment
    "Environment",
    "Scheduler",
    # Voting systems
    "VotingSystem",
    "PluralityVoting",
    "RankingVoting",
    "QuadraticVoting",
    "QuadraticVotingBudget",
    # Analysis
    "DataCollector",
    "SimulationVisualizer",
    # Core logic
    "get_votes",
    "init_preferences",
    "kl_divergence",
    "total_dissatisfaction_per_candidate",
    # Utils
    "generate_candidates",
    "generate_observations",
]
