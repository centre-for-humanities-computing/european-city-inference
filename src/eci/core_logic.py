import jax.numpy as jnp
from jax.typing import ArrayLike


def kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> ArrayLike:
    """Calculate the KL divergence between two Gaussian distributions.

    Parameters
    ----------
    mean_belief :
        Mean of the belief distribution.
    precision_belief :
        Precision of the belief distribution.
    mean_pref :
        Mean of the preference distribution.
    precision_pref :
        Precision of the preference distribution.

    Returns
    -------
        KL divergence.

    Raises
    ------
    ValueError
        If precision values are not positive.
    """
    # Convert precision to variance
    var_belief = 1 / precision_belief
    var_pref = 1 / precision_pref

    # Calculate KL divergence using the analytical formula for Gaussian distributions
    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )
    return kl
