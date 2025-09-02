import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from pyhgf.typing import Attributes, Edges
from jax.typing import ArrayLike
from functools import partial
from scipy.stats import halfnorm, norm

def calculate_kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> Array:
    """Calculate the KL divergence between two Gaussian distributions.

    Parameters
    ----------
    mu_belief :
        Mean of the belief distribution.
    prec_belief :
        Precision of the belief distribution.
    mean_pref :
        Mean of the preference distribution.
    precision_pref :
        Precision of the preference distribution.

    Returns
    -------
        KL divergence.
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


@partial(jit, static_argnames=("edges", "input_idxs"))
def get_votes(
    key: jax.random.PRNGKey,
    attributes: Attributes,
    edges: Edges,
    node_trajectories: dict,
    input_idxs: tuple,
    candidates: list,
    mask: Array,
) -> Array:
    """Get votes based on network attributes and input data.

    Parameters
    ----------
    key :
        Random key for JAX.
    Network:
        The network containing the attributes and node trajectories.
    candidates :
        List of candidate preferences.

    Returns
    -------
        The index of the selected candidate.

    """
    # get continuous nodes matching preferences
    preferences_idx = [
        edges[idx].value_parents[0] for idx in input_idxs
    ]

    # Get the beliefs from the network
    expected_mean = jnp.array(
        [node_trajectories[i]["expected_mean"][-1] for i in preferences_idx]
    )
    expected_precision = jnp.array(
        [
            node_trajectories[i]["expected_precision"][-1]
            for i in preferences_idx
        ]
    )

    # compute the dissatisfaction based on the current beliefs
    current_dissatisfaction = calculate_kl_divergence(
        expected_mean,
        expected_precision,
        attributes[-1]["preferences"]["mean"],
        attributes[-1]["preferences"]["precision"],
    )
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)

    # Initialize a list to store the voting decisions for each candidate
    candidate_preferences = []

    # For each candidate, calculate the expected dissatisfaction in a vectorized manner
    for candidate in candidates:

        candidate_mean_pref, candidate_precision_pref = candidate

        # Calculate the expected dissatisfaction for this candidate
        expected_dissatisfaction = calculate_kl_divergence(
            expected_mean,
            expected_precision,
            candidate_mean_pref,
            candidate_precision_pref
        )
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_preferences.append(
            total_current_dissatisfaction - total_expected_dissatisfaction
        )

    # Convert candidate_preferences to JAX array
    candidate_preferences = jnp.array(candidate_preferences)
    # Softmax of candidate_preferences to get probabilities
    softmax_probs = jax.nn.softmax(candidate_preferences)

    # Log of softmax_probs for the voting distribution
    log_softmax_probs = jnp.log(softmax_probs)
    log_softmax_probs = jnp.where(mask, log_softmax_probs, -jnp.inf)

    vote = jax.random.categorical(key, log_softmax_probs)

    return vote

def generate_observations(
    n_nodes,
    n_steps,
    scenario=1,
    shock_pattern=None,
    shock_time=None,
    recovery_time=None,
    trend_shape="linear",
):
    """
    Generate observations for nodes based on specified scenarios and shock patterns.

    Parameters
    ----------
    - n_nodes: int, number of nodes.
    - n_steps: int, number of time steps.
    - scenario: int, scenario type (1 or 2).
    - shock_pattern: str, pattern of shock ("phase", "sudden", "trend", or None).
    - shock_time: int, time step at which the shock occurs.
    - recovery_time: int, time step at which recovery begins.
    - trend_shape: str, shape of the trend ("linear" or other shapes).

    Returns
    -------
    - numpy.ndarray, array of generated observations.
    """
    np.random.seed(42)  # Fix seed for reproducibility
    node_observations = []
    # Default beta parameters for the nodes
    phase1_params = (5, 1)
    phase2_params = (2, 2)
    phase3_params = phase1_params

    def generate_beta(params, size):
        return np.random.beta(a=params[0], b=params[1], size=size)

    for node in range(n_nodes):
        # Scenario 1: Stable observations
        if scenario == 1:
            node_observations.append(generate_beta(phase1_params, n_steps))
        # Scenario 2: Shock scenarios
        elif scenario == 2:
            shock_time = shock_time or n_steps // 3
            recovery_time = recovery_time or 2 * n_steps // 3
            if shock_pattern in [None, "phase"]:
                phase1_end, phase2_end = (
                    (shock_time, recovery_time)
                    if recovery_time
                    else (n_steps // 3, 2 * n_steps // 3)
                )
                obs = np.concatenate(
                    [
                        generate_beta(phase1_params, phase1_end),
                        generate_beta(phase2_params, phase2_end - phase1_end),
                        generate_beta(phase3_params, n_steps - phase2_end),
                    ]
                )
            elif shock_pattern == "sudden":
                obs = np.concatenate(
                    [
                        generate_beta(phase1_params, shock_time),
                        generate_beta(phase2_params, recovery_time - shock_time),
                        generate_beta(phase3_params, n_steps - recovery_time),
                    ]
                )
            elif shock_pattern == "trend":
                obs = np.zeros(n_steps)
                for t in range(recovery_time):
                    weight = (
                        (t / recovery_time)
                        if trend_shape == "linear"
                        else (t / recovery_time) ** 2
                    )
                    alpha = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                    beta_param = (
                        phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)
                for t in range(recovery_time, n_steps):
                    weight = (
                        1 - ((t - recovery_time) / (n_steps - recovery_time))
                        if trend_shape == "linear"
                        else (1 - (t - recovery_time) / (n_steps - recovery_time)) ** 2
                    )
                    alpha = phase2_params[0] * (1 - weight) + phase1_params[0] * weight
                    beta_param = (
                        phase2_params[1] * (1 - weight) + phase1_params[1] * weight
                    )
                    obs[t] = generate_beta((alpha, beta_param), 1)
            else:
                raise ValueError("Invalid shock_pattern specified for scenario 2.")
            node_observations.append(obs)
        else:
            raise ValueError("Scenario must be 1 or 2.")
    return jnp.column_stack(node_observations)

def generate_candidates(n_candidates, n_preferences):
    """Generates a list of candidates, each with random preferences.

    Each candidate is assigned a set of preferences, modeled as Gaussian
    distributions. The means (mu) are drawn from a normal distribution,
    and the standard deviations (sigma) from a half-normal distribution.

    Parameters
    ----------
    n_candidates : int
        The number of candidates to generate.
    n_preferences : int
        The number of preferences for each candidate.

    Returns
    -------
    list of tuple of numpy.ndarray
        A list of candidates. Each candidate is represented as a tuple
        containing two numpy arrays: one for the 'mu' values and one for
        the 'sigma' values.
    """
    mu_sigma = 1
    sigma_scale = 1
    
    candidates = []
    for _ in range(n_candidates):
        mus = norm.rvs(loc=2, scale=mu_sigma, size=n_preferences)
        pis = halfnorm.rvs(scale=sigma_scale, size=n_preferences)
        candidates.append((mus, pis))
    return candidates

def individual_vote(
    tonic_volatility: float, 
    key, 
    network, 
    candidates,
    preferences_idx: list, 
    input_data,
    mask,
):
    
    # update the tonic volatilities for this agent
    for idx in preferences_idx:
        network.attributes[idx]["tonic_volatility"] = tonic_volatility

    network.input_data(input_data=input_data);  # Add observations

    return get_votes(
        key, 
        network.attributes,
        network.edges,
        network.node_trajectories,
        network.input_idxs,
        candidates,
        mask
    )