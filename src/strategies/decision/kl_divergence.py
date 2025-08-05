import jax.numpy as jnp


def calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref):
    """Calcule la divergence KL entre deux distributions Gaussiennes."""
    var_belief = jnp.where(prec_belief > 0, 1.0 / prec_belief, jnp.inf)
    var_pref = jnp.where(precision_pref > 0, 1.0 / precision_pref, jnp.inf)

    var_belief = jnp.maximum(var_belief, 1e-9)
    var_pref = jnp.maximum(var_pref, 1e-9)

    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mu_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )

    return jnp.sum(kl)
