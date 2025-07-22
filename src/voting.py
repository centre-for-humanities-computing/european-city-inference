# my_module/voting.py
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm, halfnorm
from numpy.typing import ArrayLike
from plot import plot_distributions, plot_kl_divergences, plot_time_series
from environement import generate_observations


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


def get_votes(tonic_volatility: float, key: jax.random.PRNGKey, network: object, input_data: ArrayLike, n_preferences: int, candidates: list) -> tuple:
    """
    Get votes based on network attributes and input data.

    Args:
        tonic_volatility: float, tonic volatility parameter.
        key: jax.random.PRNGKey, random key for JAX.
        network: object, network object with attributes and node_trajectories.
        input_data: ArrayLike, input data for the network.
        n_preferences: int, number of preferences.
        candidates: list, list of candidate preferences.

    Returns:
        tuple: Updated network attributes and node trajectories.
    """
    # Initialize the last node's attributes if not already present
    if 'preferences' not in network.attributes[-1]:
        network.attributes[-1]['preferences'] = []

    # Loop to sample and store tuples
    for _ in range(n_preferences):
        mu = norm.rvs(2, 1)
        sigma = halfnorm.rvs(0, 1)
        network.attributes[-1]['preferences'].append((np.float64(mu), np.float64(sigma)))

    # Set different parameters for this agent
    for i in [3, 4, 5]:
        network.attributes[i]["tonic_volatility"] = tonic_volatility

    network.input_data(input_data=input_data)  # Add observations

    # Get the preferences from the last node of the network
    preferences = network.attributes[-1].get("preferences", [])
    mean_pref = jnp.array([float(pref[0]) for pref in preferences])
    precision_pref = jnp.array([float(pref[1]) for pref in preferences])

    # Get the beliefs from the network
    start_index = len(preferences)
    mu_belief = jnp.array([network.node_trajectories[i + start_index]["expected_mean"][-1] for i in range(len(preferences))])
    prec_belief = jnp.array([network.node_trajectories[i + start_index]["expected_precision"][-1] for i in range(len(preferences))])

    current_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref)
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)

    # Initialize a list to store the voting decisions for each candidate
    candidate_preferences = []

    # For each candidate, calculate the expected dissatisfaction in a vectorized manner
    for candidate in candidates:
        candidate_mean_pref = jnp.array([float(preference[0]) for preference in candidate])
        candidate_precision_pref = jnp.array([float(preference[1]) for preference in candidate])

        # Calculate the expected dissatisfaction for this candidate in a vectorized manner
        expected_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, candidate_mean_pref, candidate_precision_pref)
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_preferences.append(total_current_dissatisfaction - total_expected_dissatisfaction)

    # Convert candidate_preferences to JAX array
    candidate_preferences = jnp.array(candidate_preferences)

    # Softmax of candidate_preferences to get probabilities
    softmax_probs = jax.nn.softmax(candidate_preferences)

    # Log of softmax_probs for the voting distribution
    log_softmax_probs = jnp.log(softmax_probs)
    vote_decisions = jax.random.categorical(key, log_softmax_probs)

    # Update the network attributes with the vote decisions
    network.attributes[-1]["votes"] = vote_decisions

    return network.attributes, network.node_trajectories
