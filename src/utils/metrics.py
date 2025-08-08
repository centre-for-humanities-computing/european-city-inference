# voting_simulation/utils/metrics.py

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def calculate_kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> Array:
    """Compute the KL divergence between two Gaussian distributions."""
    # Convert precision to variance
    var_belief = 1.0 / precision_belief
    var_pref = 1.0 / precision_pref

    # Compute KL divergence using the analytic formula for Gaussians
    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2.0 * var_pref)
        - 0.5
    )
    return kl
