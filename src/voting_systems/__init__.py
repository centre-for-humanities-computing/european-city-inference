# voting_systems/__init__.py

from .voting_system import VotingSystem
from .majority_voting import MajorityVoting
from .quadratic_voting import QuadraticVoting

__all__ = ['VotingSystem', 'MajorityVoting', 'QuadraticVoting']