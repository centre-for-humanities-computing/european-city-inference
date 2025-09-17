import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from pyhgf.typing import Attributes, Edges
from jax.typing import ArrayLike
from functools import partial
from scipy.stats import halfnorm, norm

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

    # Ensure variances are positive and finite
    #if jnp.any(var_belief <= 0) or jnp.any(var_pref <= 0):
    #    raise ValueError("Variances must be positive.")

    #if jnp.any(jnp.isinf(var_belief)) or jnp.any(jnp.isinf(var_pref)):
    #    raise ValueError("Variances must be finite.")

    # Calculate KL divergence using the analytical formula for Gaussian distributions
    kl = (
        jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief))
        + (var_belief + (mean_belief - mean_pref) ** 2) / (2 * var_pref)
        - 0.5
    )
    return kl

@partial(jit, static_argnames=("edges", "input_idxs", "voting_system"))
def get_votes(
    key: jax.random.PRNGKey,
    attributes: Attributes,
    edges: Edges,
    node_trajectories: dict,
    input_idxs: tuple,
    candidates: list,
    mask: ArrayLike,
    voting_system: str = "basic",  # "basic", "ranked", "quadratic"
    average_proportions_vector = None,  # only for "basic (ToM)"
) -> ArrayLike:
    """Get votes based on network attributes and input data, using different voting systems.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX.
    attributes : Attributes
        Network-level attributes, including preference priors.
    edges : Edges
        Network structure mapping nodes to parents.
    node_trajectories : dict
        Dictionary of node belief trajectories (expected_mean, expected_precision).
    input_idxs : tuple
        Indices of input nodes whose preferences are considered.
    candidates : list
        List of candidate preferences (mean, precision).
    mask : Array
        Boolean mask of valid candidates.
    voting_system : str, optional
        Voting system to use:
        - "basic" (default): softmax sampling of a single candidate.
        - "ranked": full ranking of candidates (descending preference).
        - "quadratic": quadratic vote allocation based on preference strength.

    Returns
    -------
    Array
        The voting outcome, depending on the voting system:
        - "basic": index of the chosen candidate (int).
        - "ranked": ranking array of candidate indices (e.g., [2, 0, 1]).
        - "quadratic": array of quadratic vote allocations per candidate.
    """

    # Extract indices of continuous nodes matching preferences
    preferences_idx = [edges[idx].value_parents[0] for idx in input_idxs]

    # Extract current beliefs
    expected_mean = jnp.array(
        [node_trajectories[i]["expected_mean"][-1] for i in preferences_idx]
    )
    expected_precision = jnp.array(
        [node_trajectories[i]["expected_precision"][-1] for i in preferences_idx]
    )

    # Compute dissatisfaction with current state
    current_dissatisfaction = kl_divergence(
        expected_mean,
        expected_precision,
        attributes[-1]["preferences"]["mean"],
        attributes[-1]["preferences"]["precision"],
    )
    
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)

    # Evaluate each candidate
    candidate_preferences = []
    for candidate in candidates:
        candidate_mean_pref, candidate_precision_pref = candidate
        expected_dissatisfaction = kl_divergence(
            expected_mean,
            expected_precision,
            candidate_mean_pref,
            candidate_precision_pref
        )
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_preferences.append(
            total_current_dissatisfaction - total_expected_dissatisfaction
        )

    candidate_preferences = jnp.array(candidate_preferences)

    # Voting system switch ----
    # Basic voting: softmax sampling of a single candidate
    if voting_system == "basic":
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences)
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote , softmax_probs,node_trajectories
    
    # Basic voting with Theory of Mind: softmax sampling with weights
    elif voting_system == "basic (ToM)":
        if average_proportions_vector is None:
            raise ValueError("average_proportions_vector is required for 'basic (ToM)' voting system")
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences * average_proportions_vector)     # condition (basic weight) jax.nn.softmax(masked_preferences* poid for each candidate (0,1)) inféré que le candidat remporte l'election) 
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote , softmax_probs,node_trajectories
    
    # Ranked voting: full ranking of candidates
    elif voting_system == "ranked":
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        available = masked_preferences.copy()
        ranking = []
        for _ in range(masked_preferences.shape[0]):
            softmax_probs = jax.nn.softmax(available)
            choice = jax.random.categorical(key, jnp.log(softmax_probs))
            ranking.append(choice)
            available = available.at[choice].set(-jnp.inf)
        ranking = jnp.array(ranking)
        return ranking, softmax_probs,node_trajectories
    
    # Quadratic voting: quadratic vote allocation based on preference strength
    elif voting_system == "quadratic":
        return None

    else:
        raise ValueError(f"Unknown voting system: {voting_system}")
    
def generate_observations(
    n_nodes: int,
    n_steps: int,
    scenario: int = 1,
    shock_pattern: str = None,
    shock_time: int = None,
    recovery_time: int = None,
    trend_shape: str = "linear",
    dispersion: float = 1.0
) -> np.ndarray:
    """
    Generate observations for nodes based on specified scenarios and shock patterns.

    Parameters
    ----------
    n_nodes : int
        Number of nodes for which observations are generated.
    n_steps : int
        Number of time steps for simulations.
    scenario : int, optional (default=1)
        Scenario identifier (1 or 2).
    shock_pattern : str, optional
        Pattern of shock for scenario 2 (None, "phase", "sudden", or "trend").
    shock_time : int, optional
        Time step at which shock begins.
    recovery_time : int, optional
        Time step at which recovery begins.
    trend_shape : str, optional (default="linear")
        Shape of the trend. Supported: "linear".
    dispersion : float, optional (default=1.0)
        Controls the dispersion of noise added to observations.

    Returns
    -------
    np.ndarray
        A 2D array with observations for each node across all time steps.
    """
    np.random.seed(42)  # Fix seed for reproducibility
    node_observations = []
    phase1_params = (15, 1)
    phase2_params = (2, 2)
    phase3_params = phase1_params

    def generate_beta(params, size):
        a, b = params
        obs = np.random.beta(a, b, size=size)
        # Add Gaussian noise for dispersion
        obs += np.random.normal(0, 0.05 * dispersion, size=size)
        return np.clip(obs, 0, 1)

    for node in range(n_nodes):
        if scenario == 1:
            # Generate observations using phase 1 parameters for all time steps
            node_observations.append(generate_beta(phase1_params, n_steps))
        elif scenario == 2:
            # Set default shock and recovery times if not provided
            shock_time = shock_time or n_steps // 3
            recovery_time = recovery_time or 2 * n_steps // 3

            if shock_pattern in [None, "phase"]:
                phase1_end, phase2_end = shock_time, recovery_time
                obs = np.concatenate([
                    generate_beta(phase1_params, phase1_end),
                    generate_beta(phase2_params, phase2_end - phase1_end),
                    generate_beta(phase3_params, n_steps - phase2_end)
                ])
            elif shock_pattern == "sudden":
                obs = np.concatenate([
                    generate_beta(phase1_params, shock_time),
                    generate_beta(phase2_params, recovery_time - shock_time),
                    generate_beta(phase3_params, n_steps - recovery_time)
                ])
            elif shock_pattern == "trend":
                obs = np.zeros(n_steps)
                for t in range(recovery_time):
                    # Calculate weight based on trend shape
                    if trend_shape == "linear":
                        weight = (t / recovery_time)
                    else:
                        weight = (t / recovery_time) ** 2
                    # Interpolate between phase1 and phase2 parameters
                    alpha = phase1_params[0] * (1 - weight) + phase2_params[0] * weight
                    beta_param = phase1_params[1] * (1 - weight) + phase2_params[1] * weight
                    obs[t] = generate_beta((alpha, beta_param), 1)[0]

                for t in range(recovery_time, n_steps):
                    if trend_shape == "linear":
                        weight = (1 - (t - recovery_time) / (n_steps - recovery_time))
                    else:
                        weight = (1 - (t - recovery_time) / (n_steps - recovery_time)) ** 2
                    # Interpolate back between phase2 and phase1 parameters
                    alpha = phase2_params[0] * (1 - weight) + phase1_params[0] * weight
                    beta_param = phase2_params[1] * (1 - weight) + phase1_params[1] * weight
                    obs[t] = generate_beta((alpha, beta_param), 1)[0]
            else:
                raise ValueError("Invalid shock_pattern specified for scenario 2.")

            node_observations.append(obs)
        else:
            raise ValueError("Scenario must be 1 or 2.")

    # Stack node observations horizontally to form a 2D array
    return np.column_stack(node_observations)

def generate_candidates(n_candidates, n_preferences, manual_means=None, manual_precisions=None):
    """
    Generates a list of candidates, each with preferences.

    - If manual_means and manual_precisions are provided, they must be
      numpy arrays (or nested lists) of shape (n_candidates, n_preferences).
    - Otherwise, candidates are generated randomly:
        mus ~ Normal(loc=2, scale=1)
        pis ~ HalfNormal(scale=1)

    Parameters
    ----------
    n_candidates : int
        The number of candidates to generate.
    n_preferences : int
        The number of preferences for each candidate.
    manual_means : array-like, optional
        Shape (n_candidates, n_preferences). If provided, overrides random generation.
    manual_precisions : array-like, optional
        Shape (n_candidates, n_preferences). If provided, overrides random generation.

    Returns
    -------
    list of tuple of numpy.ndarray
        A list of candidates. Each candidate is represented as a tuple
        (mus, pis), where mus and pis are numpy arrays of length n_preferences.
    """
    if manual_means is not None and manual_precisions is not None:
        manual_means = np.array(manual_means)
        manual_precisions = np.array(manual_precisions)

        assert manual_means.shape == (n_candidates, n_preferences), \
            f"manual_means must be shape ({n_candidates}, {n_preferences})"
        assert manual_precisions.shape == (n_candidates, n_preferences), \
            f"manual_precisions must be shape ({n_candidates}, {n_preferences})"

        candidates = [(manual_means[i], manual_precisions[i]) for i in range(n_candidates)]

    else:
        mu_sigma = 1
        sigma_scale = 1

        candidates = []
        for _ in range(n_candidates):
            mus = norm.rvs(loc=2, scale=mu_sigma, size=n_preferences)
            pis = halfnorm.rvs(scale=sigma_scale, size=n_preferences)
            candidates.append((mus, pis))

    return candidates

def individual_vote(
    mus: np.ndarray,
    pis: np.ndarray,
    tonic_volatility: float,
    key,
    network,
    candidates,
    *,
    n_preferences: int,
    input_data,
    mask,
    voting_system,
    average_proportions_vector=None  # Only for "basic (ToM)"
):
    """
    Generate individual votes based on network attributes, preferences, and input data.

    Parameters
    ----------
    mus : np.ndarray or None
        Mean preferences for the agent. If None, new preferences are sampled.
    pis : np.ndarray or None
        Precision of preferences for the agent. If None, new precisions are sampled.
    tonic_volatility : float
        Volatility of tonic preferences for updating nodes.
    key : jax.random.PRNGKey
        Random key for JAX.
    network : object
        Contains network attributes, edges, and methods like `input_data`.
    candidates : list
        List of candidates (mean, precision) to vote on.
    n_preferences : int
        Number of preferences to generate if mus or pis is None.
    input_data : array-like
        Input data to update the network.
    mask : array-like
        Boolean mask of valid candidates.
    voting_system : str
        Voting system to use ("basic", "basic (ToM)", "ranked", "quadratic").
    average_proportions_vector : array-like, optional
        Weights for candidate probabilities in "basic (ToM)" voting system.

    Returns
    -------
    Tuple of (vote, softmax_probs, dissatisfaction_scores)
        Vote outcome, softmax probabilities for candidates, and dissatisfaction scores.
    """
    if mus is None or pis is None:
        # Sample preferences for the agent if not provided
        mus, pis = [], []
        for _ in range(n_preferences):
            mus.append(norm.rvs(2, 1))  # Sample from a normal distribution
            pis.append(halfnorm.rvs(0, 1))  # Sample from a half-normal distribution
        # Update network attributes with new preferences
        network.attributes[-1]["preferences"] = {
            "mean": np.array(mus),
            "precision": np.array(pis)
        }
    else:
        # Use provided preferences
        network.attributes[-1]["preferences"] = {
            "mean": mus,
            "precision": pis
        }

    # Get continuous nodes matching preferences
    preferences_idx = [
        network.edges[idx].value_parents[0] for idx in network.input_idxs
    ]

    # Update the tonic volatilities for this agent
    for idx in preferences_idx:
        network.attributes[idx]["tonic_volatility"] = tonic_volatility

    # Add observations from input data
    network.input_data(input_data=input_data)

    # Compute votes using the get_votes function
    vote, softmax_probs, node_trajectories = get_votes(
        key,
        network.attributes,
        network.edges,
        network.node_trajectories,
        network.input_idxs,
        candidates,
        mask,
        voting_system,
        average_proportions_vector=average_proportions_vector
    )

    # Compute dissatisfaction per candidate for this agent
    dissatisfaction_scores = total_dissatisfaction_per_candidate(
        node_trajectories=network.node_trajectories,
        input_idxs=network.input_idxs,
        candidates=candidates,
        attributes=network.attributes
    )

    return vote, softmax_probs, dissatisfaction_scores, node_trajectories

def total_dissatisfaction_per_candidate(
    node_trajectories: dict,
    input_idxs: tuple,
    candidates: list,
    attributes: list
) -> jnp.ndarray:
    """
    Compute the total dissatisfaction for each candidate based on KL divergence
    between the agent's current beliefs and candidate preferences.

    Parameters
    ----------
    node_trajectories : dict
        Dictionary mapping node indices to their belief trajectories, with keys:
        "expected_mean" and "expected_precision" (arrays of shape [time_steps]).
    input_idxs : tuple
        Indices of the nodes whose preferences are considered.
    candidates : list of tuple of ArrayLike
        Each candidate is a tuple of (mean_pref, precision_pref), each an array
        of the same length as `input_idxs`.
    attributes : list of dict
        Network attributes; the last element should contain the baseline preferences:
        attributes[-1]["preferences"]["mean"] and ["precision"].

    Returns
    -------
    jnp.ndarray
        Array of shape (n_candidates,) containing total dissatisfaction scores for
        each candidate. Higher values indicate candidates that better reduce
        dissatisfaction relative to the baseline.
    """
    # Extract current beliefs of the input nodes
    expected_mean = jnp.array([node_trajectories[i]["expected_mean"][-1] for i in input_idxs])
    expected_precision = jnp.array([node_trajectories[i]["expected_precision"][-1] for i in input_idxs])

    # Baseline dissatisfaction relative to current network preferences
    baseline_mean = jnp.array(attributes[-1]["preferences"]["mean"])
    baseline_precision = jnp.array(attributes[-1]["preferences"]["precision"])
    current_dissatisfaction = jnp.sum(
        kl_divergence(expected_mean, expected_precision, baseline_mean, baseline_precision)
    )

    # Function to compute dissatisfaction for a single candidate
    def candidate_dissatisfaction(candidate):
        mean_pref, precision_pref = candidate
        kl = kl_divergence(expected_mean, expected_precision, mean_pref, precision_pref)
        return current_dissatisfaction - jnp.sum(kl)

    # Vectorized computation over all candidates
    total_diss = jnp.array([candidate_dissatisfaction(c) for c in candidates])
    return total_diss

def init_preferences(n_agents, n_preferences, manual_means=None, manual_precisions=None):
    """
    Initialize preferences for agents either manually or randomly.

    Parameters
    ----------
    n_agents : int
        Number of agents for which to initialize preferences.
    n_preferences : int
        Number of preferences each agent has.
    manual_means : np.ndarray, optional
        Manually specified mean preferences for agents. Shape must be (n_agents, n_preferences).
    manual_precisions : np.ndarray, optional
        Manually specified precision preferences for agents. Shape must be (n_agents, n_preferences).

    Returns
    -------
    dict
        A dictionary with keys "mean" and "precision", each containing a numpy array of shape (n_agents, n_preferences).
    """
    if manual_means is not None and manual_precisions is not None:
        manual_means = np.array(manual_means)
        manual_precisions = np.array(manual_precisions)
        assert manual_means.shape == (n_agents, n_preferences), "manual_means must have shape (n_agents, n_preferences)"
        assert manual_precisions.shape == (n_agents, n_preferences), "manual_precisions must have shape (n_agents, n_preferences)"
        all_mus, all_pis = manual_means, manual_precisions
    else:
        all_mus, all_pis = [], []
        for _ in range(n_agents):
            # Generate means and precisions randomly
            mus = norm.rvs(4, 1, size=n_preferences)
            pis = halfnorm.rvs(loc=0, scale=0.5, size=n_preferences)
            all_mus.append(mus)
            all_pis.append(pis)
        all_mus = np.array(all_mus)
        all_pis = np.array(all_pis)

    return {"mean": all_mus, "precision": all_pis}
