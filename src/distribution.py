import jax.numpy as jnp

def calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref):
    """
    Calculate the KL divergence between two Gaussian distributions.

    Args:
        mu_belief: Mean of the belief distribution.
        prec_belief: Precision of the belief distribution.
        mean_pref: Mean of the preference distribution.
        precision_pref: Precision of the preference distribution.

    Returns:
        KL divergence.
    """
    # Convert precision to variance
    var_belief = jnp.where(prec_belief > 0, 1 / prec_belief, jnp.inf)
    var_pref = jnp.where(precision_pref > 0, 1 / precision_pref, jnp.inf)

    # Calculate KL divergence using the analytical formula for Gaussian distributions
    kl = jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief)) + \
         (var_belief + (mu_belief - mean_pref) ** 2) / (2 * var_pref) - 0.5
    return kl
