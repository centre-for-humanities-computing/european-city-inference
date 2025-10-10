from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.typing import ArrayLike
from pyhgf.typing import Attributes, Edges
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
    voting_system: str = "Plurality Voting",
    average_proportions_vector=None,
    budget: float = 100.0,
) -> ArrayLike:
    """Get votes based on network attributes and input data.

    Parameters
    ----------
    key
        Random key for JAX.
    attributes
        Network-level attributes, including preference priors.
    edges
        Network structure mapping nodes to parents.
    node_trajectories
        Dictionary of node belief trajectories (expected_mean, expected_precision).
    input_idxs
        Indices of input nodes whose preferences are considered.
    candidates
        List of candidate preferences (mean, precision).
    mask
        Boolean mask of valid candidates.
    voting_system
        Voting system to use:
        - "Plurality Voting" (default): softmax sampling of a single candidate.
        - "Ranked Voting": full ranking of candidates (descending preference).
        - "Quadratic Voting": quadratic vote allocation based on preference strength.
        - "Plurality Voting (ToM)": basic voting with Theory of Mind weights.
        - "Ranking Voting (ToM)": ranked voting with Theory of Mind weights.
        - "Quadratic Voting (ToM)": quadratic voting with Theory of Mind weights.
    average_proportions_vector
        Weights for candidate probabilities in "ToM" voting systems.
    budget
        Budget for quadratic voting (default is 100.0).

    Returns
    -------
    Array
        The voting outcome, depending on the voting system:
        - "Plurality Voting": index of the chosen candidate (int).
        - "Ranked Voting": ranking array of candidate indices (e.g., [2, 0, 1]).
        - "Quadratic Voting": array of quadratic vote allocations per candidate.
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
            candidate_precision_pref,
        )
        total_expected_dissatisfaction = jnp.sum(expected_dissatisfaction)
        candidate_preferences.append(
            total_current_dissatisfaction - total_expected_dissatisfaction
        )

    candidate_preferences = jnp.array(candidate_preferences)

    # Voting system switch ----
    # Plurality voting: softmax sampling of a single candidate
    if voting_system == "Plurality Voting":
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences)
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote, softmax_probs, node_trajectories

    # Plurality voting with Theory of Mind: softmax sampling with weights
    elif voting_system == "Plurality Voting (ToM)":
        if average_proportions_vector is None:
            raise ValueError(
                "average_proportions_vector is required for 'basic (ToM)' voting system"
            )
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences * average_proportions_vector)
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote, softmax_probs, node_trajectories

    elif voting_system == "Ranking Voting":
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        available = masked_preferences.copy()
        ranking = []
        loop_key = key
        for _ in range(masked_preferences.shape[0]):
            softmax_probs = jax.nn.softmax(available)
            loop_key, subkey = jax.random.split(loop_key)
            choice = jax.random.categorical(subkey, jnp.log(softmax_probs))
            ranking.append(choice)
            available = available.at[choice].set(-jnp.inf)
        ranking = jnp.array(ranking)
        return ranking, softmax_probs, node_trajectories

    # Ranked voting with Theory of Mind
    elif voting_system == "Ranking Voting (ToM)":
        if average_proportions_vector is None:
            raise ValueError("average_proportions_vector is required for ToM.")
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        weighted_preferences = masked_preferences * average_proportions_vector
        available = weighted_preferences.copy()
        ranking = []
        loop_key = key
        for _ in range(masked_preferences.shape[0]):
            softmax_probs = jax.nn.softmax(available)
            loop_key, subkey = jax.random.split(loop_key)
            choice = jax.random.categorical(subkey, jnp.log(softmax_probs))
            ranking.append(choice)
            available = available.at[choice].set(-jnp.inf)
        ranking = jnp.array(ranking)
        return ranking, softmax_probs, node_trajectories

    elif voting_system == "Quadratic Voting (Budget)":
        masked_preferences = jnp.where(mask, candidate_preferences, 0.0)
        sorted_idx = jnp.argsort(masked_preferences)[::-1]
        allocations = jnp.zeros_like(masked_preferences)
        n_valid = jnp.sum(mask)

        # Assign 0.6 of the budget to the top candidate (if any)
        def assign_top(allocs):
            return allocs.at[sorted_idx[0]].set(0.6)

        allocations = jax.lax.cond(n_valid >= 1, assign_top, lambda x: x, allocations)

        # Assign 0.3 of the budget to the second candidate (if any)
        def assign_second(allocs):
            return allocs.at[sorted_idx[1]].set(0.3)

        allocations = jax.lax.cond(
            n_valid >= 2, assign_second, lambda x: x, allocations
        )

        # Assign 0.1 of the budget to the least preferred candidate (if at least 3)
        def assign_least(allocs):
            return allocs.at[sorted_idx[-1]].set(0.1)

        allocations = jax.lax.cond(n_valid >= 3, assign_least, lambda x: x, allocations)
        # Convert allocations into actual credits and quadratic votes
        credits_spent = allocations * budget
        quadratic_votes = jnp.sqrt(credits_spent)
        proportions = allocations
        return quadratic_votes, proportions, node_trajectories

    elif voting_system == "Quadratic Voting (Budget ToM)":
        if average_proportions_vector is None:
            raise ValueError("average_proportions_vector is required for ToM.")

        # Mask and weight preferences
        masked_preferences = jnp.where(mask, candidate_preferences, 0.0)
        weighted_preferences = masked_preferences * average_proportions_vector

        # Sort candidates by weighted preference
        sorted_idx = jnp.argsort(weighted_preferences)[::-1]
        allocations = jnp.zeros_like(weighted_preferences)
        n_valid = jnp.sum(mask)

        # Assign 0.6 of the budget to the top candidate (if any)
        def assign_top(allocs):
            return allocs.at[sorted_idx[0]].set(0.6)

        allocations = jax.lax.cond(n_valid >= 1, assign_top, lambda x: x, allocations)

        # Assign 0.3 of the budget to the second candidate (if any)
        def assign_second(allocs):
            return allocs.at[sorted_idx[1]].set(0.3)

        allocations = jax.lax.cond(
            n_valid >= 2, assign_second, lambda x: x, allocations
        )

        # Assign 0.1 of the budget to the least preferred candidate (if at least 3)
        def assign_least(allocs):
            return allocs.at[sorted_idx[-1]].set(0.1)

        allocations = jax.lax.cond(n_valid >= 3, assign_least, lambda x: x, allocations)

        # Convert allocations into actual credits and quadratic votes
        credits_spent = allocations * budget
        quadratic_votes = jnp.sqrt(credits_spent)

        proportions = allocations
        return quadratic_votes, proportions, node_trajectories

    # Quadratic voting: quadratic vote allocation based on preference strength
    elif voting_system == "Quadratic Voting":
        masked_preferences = jnp.where(mask, candidate_preferences, 0.0)
        positive_preferences = jnp.maximum(masked_preferences, 0.0)
        quadratic_votes = jnp.sqrt(positive_preferences)
        total_votes = jnp.sum(quadratic_votes)
        proportions = jnp.nan_to_num(quadratic_votes / total_votes)
        return quadratic_votes, proportions, node_trajectories

    # Quadratic voting with Theory of Mind
    elif voting_system == "Quadratic Voting (ToM)":
        if average_proportions_vector is None:
            raise ValueError("average_proportions_vector is required for ToM.")
        masked_preferences = jnp.where(mask, candidate_preferences, 0.0)
        weighted_preferences = masked_preferences * average_proportions_vector
        positive_preferences = jnp.maximum(weighted_preferences, 0.0)
        quadratic_votes = jnp.sqrt(positive_preferences)
        total_votes = jnp.sum(quadratic_votes)
        proportions = jnp.nan_to_num(quadratic_votes / total_votes)
        return quadratic_votes, proportions, node_trajectories

    else:
        raise ValueError(f"Unknown voting system: {voting_system}")


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
    average_proportions_vector=None,
    budget: float = 100.0,
):
    """Generate individual votes, preferences, and input data.

    Parameters
    ----------
    mus
        Mean preferences for the agent. If None, new preferences are sampled.
    pis
        Precision of preferences for the agent. If None, new precisions are sampled.
    tonic_volatilityt
        Volatility of tonic preferences for updating nodes.
    key
        Random key for JAX.
    network
        Contains network attributes, edges, and methods like `input_data`.
    candidates
        List of candidates (mean, precision) to vote on.
    n_preferences
        Number of preferences to generate if mus or pis is None.
    input_data
        Input data to update the network.
    mask
        Boolean mask of valid candidates.
    voting_system
        Voting system to use ("basic", "basic (ToM)", "ranked", "quadratic").
    average_proportions_vector
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
            "precision": np.array(pis),
        }
    else:
        # Use provided preferences
        network.attributes[-1]["preferences"] = {"mean": mus, "precision": pis}

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
        average_proportions_vector=average_proportions_vector,
        budget=budget,
    )

    # Compute dissatisfaction per candidate for this agent
    dissatisfaction_scores = total_dissatisfaction_per_candidate(
        node_trajectories=network.node_trajectories,
        input_idxs=network.input_idxs,
        candidates=candidates,
        attributes=network.attributes,
    )

    return vote, softmax_probs, dissatisfaction_scores, node_trajectories


def total_dissatisfaction_per_candidate(
    node_trajectories: dict, input_idxs: tuple, candidates: list, attributes: list
) -> jnp.ndarray:
    """Compute the total dissatisfaction for each candidate.

        This is based on the KL divergence between the agent's current beliefs and
        the candidate's preferences.

    Parameters
    ----------
    node_trajectories
        Dictionary mapping node indices to their belief trajectories, with keys:
        "expected_mean" and "expected_precision" (arrays of shape [time_steps]).
    input_idxs
        Indices of the nodes whose preferences are considered.
    candidates
        Each candidate is a tuple of (mean_pref, precision_pref), each an array
        of the same length as `input_idxs`.
    attributes
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
    expected_mean = jnp.array(
        [node_trajectories[i]["expected_mean"][-1] for i in input_idxs]
    )
    expected_precision = jnp.array(
        [node_trajectories[i]["expected_precision"][-1] for i in input_idxs]
    )

    # Baseline dissatisfaction relative to current network preferences
    baseline_mean = jnp.array(attributes[-1]["preferences"]["mean"])
    baseline_precision = jnp.array(attributes[-1]["preferences"]["precision"])
    current_dissatisfaction = jnp.sum(
        kl_divergence(
            expected_mean, expected_precision, baseline_mean, baseline_precision
        )
    )

    # Function to compute dissatisfaction for a single candidate
    def candidate_dissatisfaction(candidate):
        mean_pref, precision_pref = candidate
        kl = kl_divergence(expected_mean, expected_precision, mean_pref, precision_pref)
        return current_dissatisfaction - jnp.sum(kl)

    # Vectorized computation over all candidates
    total_diss = jnp.array([candidate_dissatisfaction(c) for c in candidates])
    return total_diss


def init_preferences(
    n_agents, n_preferences, manual_means=None, manual_precisions=None
):
    """Initialize preferences for agents either manually or randomly.

    Parameters
    ----------
    n_agents
        Number of agents for which to initialize preferences.
    n_preferences
        Number of preferences each agent has.
    manual_means
        Manually specified mean preferences for agents.
    manual_precisions
        Manually specified precision preferences for agents.

    Returns
    -------
    dict
        A dictionary with keys "mean" and "precision".
    """
    if manual_means is not None and manual_precisions is not None:
        manual_means = np.array(manual_means)
        manual_precisions = np.array(manual_precisions)
        if manual_means.shape != (n_agents, n_preferences):
            raise ValueError("manual_means must have shape (n_agents, n_preferences)")

        if manual_precisions.shape != (n_agents, n_preferences):
            raise ValueError(
                "manual_precisions must have shape (n_agents, n_preferences)"
            )

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
