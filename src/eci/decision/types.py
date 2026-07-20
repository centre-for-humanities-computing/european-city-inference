from typing import TypeAlias, TypedDict

import jax
from jax.typing import ArrayLike


class GaussianParameters(TypedDict):
    """Mean and precision arrays describing Gaussian distributions."""

    mean: jax.Array
    precision: jax.Array


class ElectionData(TypedDict):
    """Belief, preference, and candidate distributions used in an election."""

    beliefs: GaussianParameters
    preferences: GaussianParameters
    candidates: GaussianParameters


ResponseResult: TypeAlias = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
"""Sampled votes, probabilities, candidate utilities, and the next PRNG key."""
