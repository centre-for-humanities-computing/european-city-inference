import jax.numpy as jnp
from abc import ABC, abstractmethod
import numpy as np
from jax import vmap
from jax.tree_util import Partial
from pyhgf.model import Network
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde,halfnorm
import colorsys
import matplotlib.gridspec as gridspec
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from pyhgf.typing import Attributes, Edges
from jax.typing import ArrayLike
from functools import partial
from collections import Counter
from typing import Optional
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class Agent(ABC):
    """An abstract base class for all agents in the simulation.

    This class provides the basic structure for any agent, ensuring that
    each has a unique integer ID and a `step` method for its actions.

    Parameters
    ----------
    agent_id : int
        A unique identifier for the agent.

    Attributes
    ----------
    id : int
        The agent's unique identifier.

    """
    def __init__(self, agent_id: int):
        self.id = agent_id

    @abstractmethod
    def step(self, environment: Any) -> None:
        """Defines the agent's action during a single simulation step.

        This method must be implemented by all subclasses. It contains the
        logic for what the agent does during its activation.

        Parameters
        ----------
        environment : Any
            The environment in which the agent exists, providing access to
            global state and other agents if needed.

        """
        pass

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the agent."""
        return f"{self.__class__.__name__}(id={self.id})"

class Voter(Agent):
    """Represents a voter agent in the simulation.

    Voters have a set of preferences and a volatility level, which are used
    by the `Environment` to compute their voting decisions. The `last_*`
    attributes are used to store the results of these computations for each step.

    Parameters
    ----------
    agent_id : int
        The unique identifier for the voter.
    preferences : dict
        A dictionary containing the voter's preferences, typically including
        'mean' and 'precision' vectors.
    tonic_volatility : float
        A parameter representing the voter's baseline level of choice volatility.

    Attributes
    ----------
    preferences : dict
        The voter's intrinsic preferences.
    tonic_volatility : float
        The voter's baseline choice volatility.
    last_vote : Optional[int]
        The ID of the candidate the voter chose in the last step.
    last_softmax_probs : Optional[dict]
        A dictionary mapping candidate IDs to the voter's choice probabilities
        in the last step.
    last_dissatisfactions : Optional[dict]
        A dictionary mapping candidate IDs to the voter's dissatisfaction
        score for each candidate in the last step.
    traj : Any
        Stores trajectory data from the last simulation step, if any.
    observation : Any
        Stores observation data from the last simulation step, if any.

    """
    def __init__(self, agent_id: int, preferences: Dict, tonic_volatility: float):
        """Initializes a Voter agent."""
        super().__init__(agent_id)
        self.preferences = preferences
        self.tonic_volatility = tonic_volatility
        self.last_vote: Optional[int] = None
        self.last_softmax_probs: Optional[Dict[int, float]] = None
        self.last_dissatisfactions: Optional[Dict[int, float]] = None
        self.traj: Optional[Any] = None
        self.observation: Optional[Any] = None

    def step(self, environment: Any) -> None:
        """Performs the voter's action for a step.

        Notes
        -----
        In this model, the core voting logic is handled externally by the
        vectorized JAX functions in the `Environment`. This method is a
        placeholder for any additional, non-vectorized actions a voter
        might take.

        """
        pass

class Candidate(Agent):
    """Represents a candidate agent who can be elected.

    Candidates have a policy platform that voters evaluate.

    Parameters
    ----------
    agent_id : int
        The unique identifier for the candidate.
    policy_platform : dict
        A dictionary representing the candidate's policy platform, typically
        including 'mean' and 'precision' vectors.

    Attributes
    ----------
    policy : dict
        The candidate's policy platform.
    vote_count : int
        A simple counter for votes, primarily for debugging. The official
        tally is managed by the `VotingSystem`.

    """
    def __init__(self, agent_id: int, policy_platform: Dict):
        """Initializes a Candidate agent."""
        super().__init__(agent_id)
        self.policy = policy_platform
        self.vote_count = 0

    def step(self, environment: Any) -> None:
        """Performs the candidate's action for a step.

        Notes
        -----
        This method could be used to implement dynamic behaviors, such as
        a candidate changing their policy platform in response to voter
        opinions (i.e., campaigning).

        """
        pass
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
    if voting_system == "Plurality Voting":
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences)
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote , softmax_probs,node_trajectories
    
    # Basic voting with Theory of Mind: softmax sampling with weights
    elif voting_system == "Plurality Voting (ToM)":
        if average_proportions_vector is None:
            raise ValueError("average_proportions_vector is required for 'basic (ToM)' voting system")
        masked_preferences = jnp.where(mask, candidate_preferences, -jnp.inf)
        softmax_probs = jax.nn.softmax(masked_preferences * average_proportions_vector)     # condition (basic weight) jax.nn.softmax(masked_preferences* poid for each candidate (0,1)) inféré que le candidat remporte l'election) 
        vote = jax.random.categorical(key, jnp.log(softmax_probs))
        return vote , softmax_probs,node_trajectories
    
    # Ranked voting: full ranking of candidates
    elif voting_system == "Ranking Voting":
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
    elif voting_system == "Quadratic Voting":
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

class VotingSystem(ABC):
    """An interface for vote-counting systems.

    Notes
    -----
    This is an abstract base class (ABC) that defines the essential methods
    and properties any concrete voting system must implement.

    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the voting system."""
        pass

    @abstractmethod
    def counting_votes(self, voters: list[Voter], candidates: list[Candidate]) -> int:
        """Count votes and determine the winning candidate.

        This method must also store the detailed results of the count in an
        internal attribute for later inspection.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects participating in the election.
        candidates : list[Candidate]
            A list of Candidate objects running in the election.

        Returns
        -------
        int
            The ID of the winning candidate. Returns -1 in the case of a tie
            or if no votes are cast.

        """
        pass
    
class PluralityVoting(VotingSystem):
    """A vote-counting system based on the plurality rule.

    Attributes
    ----------
    last_results : dict[int, int]
        A dictionary storing the results of the last vote count, mapping
        candidate IDs to their vote totals.

    Notes
    -----
    The rule is simple: the candidate with the most first-preference votes wins.
    This is also known as "first-past-the-post".

    """
    def __init__(self):
        """Initializes the PluralityVoting system."""
        self.last_results: dict[int, int] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
        return "Plurality Voting"

    def counting_votes(self, voters: list[Voter], candidates: list[Candidate]) -> int:
        """Count votes based on the plurality rule.

        The method counts the number of times each candidate's ID appears
        in the voters' `last_vote` attribute.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects. It is assumed that `voter.last_vote`
            contains the ID of the chosen candidate.
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        int
            The ID of the winning candidate. Returns -1 if there is a tie for
            first place or if no votes were cast.

        """
        vote_indices = [v.last_vote for v in voters if v.last_vote is not None]

        if not vote_indices:
            self.last_results = {c.id: 0 for c in candidates}
            return -1

        # **CORRECTION : Convertir les indices en véritables IDs de candidats**
        candidate_ids = [c.id for c in candidates]
        actual_vote_ids = [candidate_ids[idx] for idx in vote_indices]
        
        # Initialiser les résultats pour tous les candidats
        self.last_results = {c.id: 0 for c in candidates}

        # Compter les occurrences de chaque ID de vote
        counts = Counter(actual_vote_ids)
        self.last_results.update(counts)

        # La logique pour trouver le gagnant reste la même
        if not counts:
             return -1
             
        most_common = counts.most_common(2)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return -1  # Tie

        winner_id = most_common[0][0]
        return winner_id

class RankingVoting(VotingSystem):
    """A system where voters rank candidates, and a winner is chosen via Borda count.

    Attributes
    ----------
    last_results : dict[int, int]
        Stores the Borda point totals from the last election, mapping
        candidate IDs to their final scores.

    Notes
    -----
    In the Borda count method, each voter provides a ranking of candidates.
    Points are awarded based on rank. For N candidates, a first-place rank
    gets N-1 points, a second-place rank gets N-2 points, and so on, down to
    0 points for the last-place candidate. The candidate with the highest
    total score wins.

    This implementation uses a pre-computed ranking stored in the `last_vote`
    attribute of the Voter object.

    """
    def __init__(self):
        """Initializes the RankingVoting system."""
        self.last_results: dict[int, int] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
        return "Ranking Voting"

    def counting_votes(self, voters: list[Voter], candidates: list[Candidate]) -> int:
        """Calculates the Borda count winner from voter rankings.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects. Each voter is expected to have a
            `last_vote` attribute containing a pre-computed array of
            candidate IDs, ordered by preference.
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        int
            The ID of the winning candidate. Returns -1 if there is a tie for
            first place or if no valid rankings are provided.

        """
        num_candidates = len(candidates)
        if num_candidates == 0:
            self.last_results = {}
            return -1

        points = {c.id: 0 for c in candidates}
        borda_points = jnp.arange(num_candidates - 1, -1, -1)

        for voter in voters:
            ranked_indices = getattr(voter, 'last_vote', None)
            if ranked_indices is None or not hasattr(ranked_indices, '__iter__'):
                continue

            # **CORRECTION : Itérer sur les indices et les mapper aux IDs des candidats**
            for rank, candidate_idx in enumerate(ranked_indices):
                if rank < len(borda_points):
                    # Récupérer le vrai ID du candidat à partir de son index
                    candidate_id = candidates[int(candidate_idx)].id
                    points[candidate_id] += borda_points[rank]

        self.last_results = points
        if sum(points.values()) == 0:
            return -1

        max_score = -1
        winner_id = -1
        is_tie = False
        
        # Trouver le score maximum et vérifier les égalités
        sorted_scores = sorted(points.items(), key=lambda item: item[1], reverse=True)
        
        if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
             is_tie = True
        
        if not is_tie and sorted_scores:
             winner_id = sorted_scores[0][0]

        return winner_id if not is_tie else -1

class QuadraticVoting(VotingSystem):
    """A system where voters allocate credits to express preference intensity.

    Attributes
    ----------
    VOTE_CREDITS_BUDGET : int
        The total number of credits each voter can allocate among candidates.
    last_results : dict[int, float]
        Stores the quadratic vote totals from the last election, mapping
        candidate IDs to their final scores.

    Notes
    -----
    In Quadratic Voting, each voter has a budget of "vote credits." They can
    allocate these credits to candidates to show the intensity of their preference.
    The number of official votes a candidate receives from a voter is the 
    **square root** of the credits allocated. 
    
    This system allows for nuanced preference expression while curbing the
    influence of overly passionate minorities. This implementation uses a
    `last_softmax_probs` attribute on the Voter object to determine how the
    credit budget is distributed. 

    """
    VOTE_CREDITS_BUDGET = 100

    def __init__(self):
        """Initializes the QuadraticVoting system."""
        self.last_results: dict[int, float] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
        return "Quadratic Voting"

    def counting_votes(self, voters: list[Voter], candidates: list[Candidate]) -> int:
        """Calculates the winner using the Quadratic Voting method.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects. Each voter is expected to have a
            `last_softmax_probs` attribute: a dictionary mapping candidate IDs
            to a probability-like score (summing to 1.0).
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        int
            The ID of the winning candidate. Returns -1 if there is a tie for
            first place or if no credits are allocated.

        """
        if not candidates:
            self.last_results = {}
            return -1

        total_quadratic_votes = {c.id: 0.0 for c in candidates}

        for voter in voters:
            probabilities_by_index = getattr(voter, 'last_softmax_probs', None)
            if not isinstance(probabilities_by_index, dict):
                continue

            # **CORRECTION : Mapper l'index de la probabilité à l'ID du candidat**
            for candidate_idx, prob in probabilities_by_index.items():
                if candidate_idx < len(candidates):
                    # Récupérer le vrai ID du candidat
                    candidate_id = candidates[candidate_idx].id
                    credits_allocated = prob * self.VOTE_CREDITS_BUDGET
                    quadratic_votes = jnp.sqrt(credits_allocated)
                    total_quadratic_votes[candidate_id] += float(quadratic_votes)

        self.last_results = total_quadratic_votes
        if sum(total_quadratic_votes.values()) == 0:
            return -1

        # Trouver le gagnant (logique similaire à ci-dessus)
        sorted_scores = sorted(total_quadratic_votes.items(), key=lambda item: item[1], reverse=True)
        
        if not sorted_scores:
            return -1

        if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
            return -1 # C'est une égalité

        return sorted_scores[0][0]

class Scheduler:
    """A scheduler to manage the activation order of agents.

    This scheduler activates agents in a pre-determined sequence. It is a
    "simultaneous" or "synchronous" scheduler, meaning all agents are
    activated once per step, in the order they are provided.

    Attributes
    ----------
    agents : list[Agent]
        The list of all agent objects managed by the scheduler.
    step_count : int
        The number of steps the scheduler has executed.

    """
    def __init__(self, agents: list[Agent]):
        """Initializes the scheduler.

        Parameters
        ----------
        agents : list[Agent]
            A list of agents to be activated in each step. The order of
            activation is determined by the order of this list.

        """
        self.agents = agents
        self.step_count = 0

    def step(self, environment) -> None:
        """Activates all agents for one simulation step.

        This method iterates through the list of agents and calls the `step`
        method on each one, passing the environment. After all agents have
        been activated, the internal step counter is incremented.

        Parameters
        ----------
        environment : Any
            The simulation environment, which is passed to each agent's
            `step` method.

        """
        for agent in self.agents:
            agent.step(environment)
        self.step_count += 1

class DataCollector:
    """Collects and stores detailed data from the simulation.

    This class is responsible for recording the state of the simulation at
    each step and providing methods to export the collected data into
    structured pandas DataFrames for analysis.

    Attributes
    ----------
    records : list[dict]
        A list of dictionaries, where each dictionary holds the collected
        data for a single simulation step.

    """
    def __init__(self):
        """Initializes the DataCollector with an empty records list."""
        self.records = []

    def collect(self, environment: 'Environment') -> None:
        """Records the state of the model at the end of a step.

        This method captures key metrics from the environment, including
        the current step number, the winning candidate's ID, the voting system
        in use, and detailed scores and proportions for each candidate.

        Parameters
        ----------
        environment : Environment
            The simulation environment object from which to collect data. It is
            expected to have `scheduler`, `winner_id`, and `voting_system`
            attributes.

        """
        step_info = {
            'step': environment.scheduler.step_count,
            'winner_id': environment.winner_id,
            'voting_system': environment.voting_system.name
        }

        results = environment.voting_system.last_results
        if not results:
            self.records.append(step_info)
            return

        # Store raw scores for each candidate
        raw_scores = {f"candidate_{cid}_score": score for cid, score in results.items()}
        step_info.update(raw_scores)

        # Calculate and store score proportions
        total_score = sum(results.values())
        if total_score > 0:
            proportions = {f"candidate_{cid}_prop": score / total_score for cid, score in results.items()}
        else:
            proportions = {f"candidate_{cid}_prop": 0.0 for cid in results.keys()}
        step_info.update(proportions)

        self.records.append(step_info)

    def get_dataframe(self) -> pd.DataFrame:
        """Exports the collected data to a wide-format pandas DataFrame.

        In the wide format, each row represents a single simulation step, and
        each candidate has its own columns for score and proportion.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the collected simulation data. Missing
            values are filled with 0.

        """
        return pd.DataFrame(self.records).fillna(0)

    def get_long_dataframe(self) -> pd.DataFrame:
        """Exports data to a long-format pandas DataFrame.

        The long format is often more convenient for plotting and statistical
        analysis, as it structures the data with one observation per row
        (e.g., one row per candidate per step).

        

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the simulation data, unpivoted into a
            long format with columns like 'candidate_id', 'score', and
            'proportion'.

        """
        wide_df = self.get_dataframe()
        if wide_df.empty:
            return pd.DataFrame()

        id_vars = ['step', 'winner_id', 'voting_system']
        prop_vars = [col for col in wide_df.columns if '_prop' in col]
        score_vars = [col for col in wide_df.columns if '_score' in col]

        # Ensure we have data to melt
        if not prop_vars or not score_vars:
            # If there's no score/prop data, return the base info
            return wide_df[id_vars].copy()

        # Unpivot the proportion columns
        long_prop = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=prop_vars,
            var_name='candidate_info',
            value_name='proportion'
        )
        long_prop['candidate_id'] = long_prop['candidate_info'].str.extract(r'(\d+)').astype(int)

        # Unpivot the score columns
        long_score = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=score_vars,
            var_name='candidate_info',
            value_name='score'
        )
        long_score['candidate_id'] = long_score['candidate_info'].str.extract(r'(\d+)').astype(int)

        # Merge the two long DataFrames
        long_df = pd.merge(
            long_prop,
            long_score,
            on=['step', 'winner_id', 'voting_system', 'candidate_id']
        )

        return long_df.drop(columns=['candidate_info_x', 'candidate_info_y'])
    
class Environment:
    """Orchestrates an agent-based simulation of a voting process.

    This class is the main controller for the simulation. It is responsible
    for creating and managing agents (Voters and Candidates), setting up the
    simulation components, and running the step-by-step execution of the model.
    It heavily utilizes the JAX library to perform efficient, vectorized
    computations for the voting process.

    Parameters
    ----------
    num_voters : int
        The number of voter agents to create.
    num_candidates : int
        The number of candidate agents to create.
    num_preferences : int
        The dimensionality of the preference/policy space.
    voting_system : VotingSystem
        An instantiated voting system object that will be used to count votes.
    scenario : int, optional
        An identifier for the scenario to run, used for data generation.
        Default is 1.

    Attributes
    ----------
    key : jax.random.PRNGKey
        The master JAX random key for ensuring reproducibility.
    voters : list[Voter]
        The list of all Voter agents in the simulation.
    candidates : list[Candidate]
        The list of all Candidate agents in the simulation.
    scheduler : Scheduler
        The scheduler object that manages agent activation.
    datacollector : DataCollector
        The object responsible for collecting data at each step.
    vmap_get_votes_fn : callable
        A pre-compiled, vectorized JAX function for efficiently computing
        the votes of all agents simultaneously.
    winner_id : int or None
        The ID of the winning candidate from the most recent step.

    """
    def __init__(
        self,
        num_voters: int,
        num_candidates: int,
        num_preferences: int,
        voting_system: 'VotingSystem',
        scenario: int = 1
    ):
        """Initializes the simulation environment."""
        self.key = jax.random.PRNGKey(42)
        self.num_preferences = num_preferences
        self.next_agent_id = 0

        # --- Create Agents ---
        self.voters: List[Voter] = []
        self.candidates: List[Candidate] = []
        self._create_voters(num_voters)
        self._create_candidates(num_candidates)
        self.agents = self.voters + self.candidates

        # --- JAX Pre-computation and Setup ---
        # This section prepares and pre-compiles the core JAX function
        # to avoid compilation overhead during the simulation run.
        print("Pre-compiling JAX function...")
        self.input_data = generate_observations(n_nodes=self.num_preferences, n_steps=500, scenario=scenario)
        network_template = Network()
        network_template.add_nodes(kind="binary-state", n_nodes=self.num_preferences)
        for i in range(self.num_preferences):
            network_template.add_nodes(value_children=i)

        candidate_list = [(c.policy['mean'], c.policy['precision']) for c in self.candidates]
        
        get_votes_fn = partial(
            individual_vote,
            network=network_template,
            candidates=candidate_list,
            n_preferences=self.num_preferences,
            input_data=self.input_data,
            mask=np.ones(len(candidate_list)),
            voting_system=voting_system.name
        )
        self.vmap_get_votes_fn = vmap(get_votes_fn)
        print("Compilation complete.")

        # --- Simulation Components ---
        self.voting_system = voting_system
        self.scheduler = Scheduler(self.agents)
        self.datacollector = DataCollector()
        self.winner_id = None

    def _get_new_agent_id(self) -> int:
        """Generates and returns a unique ID for a new agent."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

    def _create_voters(self, num_voters: int) -> None:
        """Factory method to create Voters with random preferences."""
        for _ in range(num_voters):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            mean = jax.random.uniform(subkey1, shape=(self.num_preferences,))
            precision = jax.random.uniform(subkey2, shape=(self.num_preferences,)) + 1.0
            preferences = {'mean': mean, 'precision': precision}
            tonic_volatility = np.random.normal(-3.0, 1.0)
            voter = Voter(
                agent_id=self._get_new_agent_id(),
                preferences=preferences,
                tonic_volatility=tonic_volatility
            )
            self.voters.append(voter)

    def _create_candidates(self, num_candidates: int) -> None:
        """Factory method to create Candidates with random policy platforms."""
        for _ in range(num_candidates):
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            mean = jax.random.uniform(subkey1, shape=(self.num_preferences,))
            precision = jax.random.uniform(subkey2, shape=(self.num_preferences,)) * 5.0 + 1.0
            policy = {'mean': mean, 'precision': precision}
            candidate = Candidate(agent_id=self._get_new_agent_id(), policy_platform=policy)
            self.candidates.append(candidate)

    def _gather_agent_data(self) -> tuple:
        """Gathers data from all Voter objects into JAX arrays for vectorized processing."""
        all_mus = jnp.array([v.preferences['mean'] for v in self.voters])
        all_pis = jnp.array([v.preferences['precision'] for v in self.voters])
        all_volatilities = jnp.array([v.tonic_volatility for v in self.voters])
        return all_mus, all_pis, all_volatilities

    def _scatter_results(self, results: tuple) -> None:
        """Distributes results from the vectorized computation back to individual Voter objects.

        Parameters
        ----------
        results : tuple
            A tuple containing the outputs from the JAX computation, such as
            votes, probabilities, dissatisfactions, and trajectories.

        """
        votes, softmax_probs, dissatisfactions, node_trajectories = results
        is_ranking_vote = (self.voting_system.name == "Ranking Voting")
        for i, voter in enumerate(self.voters):
            voter.last_vote = votes[i] if is_ranking_vote else int(votes[i])
            voter.last_softmax_probs = {cid: prob for cid, prob in enumerate(softmax_probs[i])}
            voter.last_dissatisfactions = {cid: diss for cid, diss in enumerate(dissatisfactions[i])}
            voter.traj = node_trajectories  # Store trajectories

    def step(self) -> None:
        """Executes a single step of the simulation.

        The method follows a clear sequence:
        1. GATHER data from all agents into JAX arrays.
        2. PROCESS the data in a single, vectorized JAX call.
        3. SCATTER the results back to the individual agents.
        4. Activate agents via the scheduler.
        5. Count the votes to determine a winner.

        """
        # 1. GATHER
        all_mus, all_pis, all_volatilities = self._gather_agent_data()

        # 2. PROCESS
        self.key, *agent_keys = jax.random.split(self.key, len(self.voters) + 1)
        results = self.vmap_get_votes_fn(all_mus, all_pis, all_volatilities, jnp.array(agent_keys))

        # 3. SCATTER
        self._scatter_results(results)
        
        # 4. Agent and System Updates
        self.scheduler.step(self)
        self.winner_id = self.voting_system.counting_votes(self.voters, self.candidates)
        print(f"Winner is Candidate ID: {self.winner_id}.")

    def run(self, num_steps: int) -> None:
        """Runs the simulation for a given number of steps.

        Parameters
        ----------
        num_steps : int
            The total number of steps to run the simulation for.

        """
        print("Starting simulation...")
        for i in range(num_steps):
            print(f"--- Step {i+1}/{num_steps} ---")
            self.step()
            self.datacollector.collect(self)
        print("Simulation finished.")        

class SimulationVisualizer:
    """A class dedicated to plotting and visualizing simulation results.

    This class provides a suite of methods to generate insightful plots
    from the data produced by the simulation environment and its datacollector.

    Parameters
    ----------
    environment : Environment
        The main simulation environment instance.
    datacollector : DataCollector
        The data collector instance containing the simulation records.

    Attributes
    ----------
    env : Environment
        A reference to the simulation environment.
    collector : DataCollector
        A reference to the data collector.
    input_data : np.ndarray
        A reference to the input data used for the simulation.

    """
    def __init__(self, environment: 'Environment', datacollector: 'DataCollector'):
        """Initializes the visualizer and sets a custom plot theme."""
        self.env = environment
        self.collector = datacollector
        self.input_data = environment.input_data
        sns.set_theme(style="whitegrid", rc={
            "axes.facecolor": "#f9f9f9",
            "figure.facecolor": "#f9f9f9",
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica"
        })

    def plot_preference_distributions(self, num_voters_to_show: int = 10) -> None:
        """Visualizes the preference distributions of candidates and voters.

        This plot displays the probability density function (PDF) for each
        preference topic. Candidate preferences are shown as distinct, colored,
        filled areas. A sample of voter preferences is overlaid as a
        semi-transparent black distribution to show their alignment with the
        candidates.

        Parameters
        ----------
        num_voters_to_show : int, optional
            The number of voters to include in the visualization, by default 10.

        """
        voters = self.env.voters
        candidates = self.env.candidates
        n_preferences = self.env.num_preferences
        x_vals = np.linspace(-3, 4, 400)
        rows = []

        # Prepare candidate data
        for c in candidates:
            mus, precisions = c.policy['mean'], c.policy['precision']
            for pref in range(n_preferences):
                pdf = norm.pdf(x_vals, loc=mus[pref], scale=1/np.sqrt(precisions[pref]))
                rows.extend([{"group": "Candidate", "id": f"C{c.id}", "preference": f"Topic {pref+1}", "x": x, "pdf": y} for x, y in zip(x_vals, pdf)])

        # Prepare voter data
        for v in voters[:num_voters_to_show]:
            mus, precisions = v.preferences['mean'], v.preferences['precision']
            for pref in range(n_preferences):
                pdf = norm.pdf(x_vals, loc=mus[pref], scale=1/np.sqrt(precisions[pref]))
                rows.extend([{"group": "Voter", "id": f"V{v.id}", "preference": f"Topic {pref+1}", "x": x, "pdf": y} for x, y in zip(x_vals, pdf)])

        df = pd.DataFrame(rows)
        if df.empty:
            print("No data available to plot preference distributions.")
            return

        # Plotting setup
        preferences = sorted(df["preference"].unique())
        candidate_ids = sorted([c for c in df["id"].unique() if c.startswith("C")])
        voter_ids_to_plot = sorted([v for v in df["id"].unique() if v.startswith("V")])
        n_candidates = len(candidate_ids)
        colors = [colorsys.hls_to_rgb(i / n_candidates, 0.8, 0.6) for i in range(n_candidates)]

        fig, axes = plt.subplots(len(preferences), 1, figsize=(12, 5 * len(preferences)), sharex=True)
        axes = [axes] if len(preferences) == 1 else axes

        for ax, pref in zip(axes, preferences):
            sub_df = df[df["preference"] == pref]
            for j, cand_id in enumerate(candidate_ids):
                cand_df = sub_df[sub_df["id"] == cand_id]
                ax.fill_between(cand_df["x"], cand_df["pdf"], color=colors[j], alpha=0.8, label=cand_id)

            for i, voter_id in enumerate(voter_ids_to_plot):
                agent_df = sub_df[sub_df["id"] == voter_id]
                label = "Voters" if i == 0 else ""
                ax.fill_between(agent_df["x"], agent_df["pdf"], color="black", alpha=0.03, label=label)

            ax.set_title(pref)
            ax.set_ylabel("Density")
            ax.legend(title="Legend")
            ax.set_ylim(bottom=0)

        axes[-1].set_xlabel("Preference Value")
        plt.tight_layout()
        plt.show()

    def plot_belief_trajectory(self, voter: 'Voter') -> None:
        """Plots a voter's belief trajectory and the density of observations.

        This composite plot shows two things:
        1. The main chart displays the voter's belief (expected mean) over time,
           with a 95% confidence interval, against the actual observations.
        2. A side panel shows the normalized probability density of the
           observations, providing a summary of the data distribution.

        Parameters
        ----------
        voter : Voter
            The Voter agent whose belief trajectory will be plotted.

        """
        if not hasattr(voter, 'traj') or not voter.traj:
            print(f"Voter {voter.id} has no trajectory data to plot.")
            return

        # --- 1. Data Preparation ---
        traj = voter.traj[0]
        mean, std_dev = traj["expected_mean"][voter.id], traj["precision"][voter.id]
        observations = self.input_data[:, 0]

        # --- 2. Custom Layout Creation ---
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 5, figure=fig)
        ax_main = fig.add_subplot(gs[0, 0:4])
        ax_density = fig.add_subplot(gs[0, 4], sharey=ax_main)

        # --- 3. Main Trajectory Plot ---
        ax_main.scatter(range(len(observations)), observations, s=10, alpha=1, label="Observations", color="black", zorder=2)
        ax_main.plot(mean, label="Voter's Belief (Expected Mean)", color="crimson", linewidth=2.5, zorder=3)
        upper_bound = mean + 1.96 * std_dev
        lower_bound = mean - 1.96 * std_dev
        ax_main.fill_between(range(len(mean)), lower_bound, upper_bound, color="crimson", alpha=0.2, label="95% Confidence Interval", zorder=1)
        ax_main.set_title(f"Belief Trajectory for Voter {voter.id} and Observation Density")
        ax_main.set_xlabel("Time Step")
        ax_main.set_ylabel("Preference Value")
        ax_main.legend()
        ax_main.grid(True, linestyle='--', alpha=0.6)

        # --- 4. Side Density Plot ---
        kde = gaussian_kde(observations)
        y_grid = np.linspace(observations.min(), observations.max(), 500)
        pdf_values = kde(y_grid)
        pdf_scaled = (pdf_values - pdf_values.min()) / (pdf_values.max() - pdf_values.min())
        ax_density.fill_betweenx(y_grid, pdf_scaled, color='lightgrey', alpha=0.8, edgecolor='grey')
        ax_density.set_xlabel('Normalized Density')
        ax_density.grid(True, which='both', linestyle='--', linewidth=0.3)
        ax_density.set_xlim(0, 1.1)
        plt.setp(ax_density.get_yticklabels(), visible=False)

        # --- 5. Final Display ---
        plt.tight_layout()
        plt.show()

    def plot_simulation_results_distribution(self, results_df: Optional[pd.DataFrame] = None, plot_kind: str = 'histogram') -> None:
        """Visualizes the distribution of vote proportions across all simulations.

        Parameters
        ----------
        results_df : pd.DataFrame, optional
            A pre-generated long-format DataFrame of simulation results.
            If None, it will be fetched from the datacollector. Defaults to None.
        plot_kind : str, optional
            The type of plot to generate. Options are:
            - 'distribution': A boxplot combined with a stripplot to show
              summary statistics and individual data points.
            - 'histogram': Overlaid density histograms for each candidate.
            Defaults to 'histogram'.

        """
        if results_df is None:
            results_df = self.collector.get_long_dataframe()
            print("IDs uniques des candidats trouvés :", results_df['candidate_id'].unique())

        if results_df.empty:
            print("DataFrame is empty. Cannot generate plot.")
            return

        n_sims = results_df['step'].nunique()
        print(f"Generating results distribution plot (type: {plot_kind}) for {n_sims} steps...")

        candidate_ids = sorted(results_df['candidate_id'].unique())
        n_candidates = len(candidate_ids)
        palette = sns.color_palette("viridis", n_colors=n_candidates)
        fig, ax = plt.subplots(figsize=(12, 7))
        title = f"Distribution of Vote Proportions Across {n_sims} Steps"

        if plot_kind == 'distribution':
            sns.stripplot(data=results_df, x='proportion', y='candidate_id', orient='h',
                        color=".25", jitter=True, alpha=0.5, size=4, ax=ax)
            ax.set_ylabel("Candidate ID")
            ax.set_xlabel("Proportion of Votes")
            ax.set_title(title)
            ax.set_xlim(0, max(1.0, results_df['proportion'].max() * 1.05))
        elif plot_kind == 'histogram':
            for i, cid in enumerate(candidate_ids):
                proportions = results_df.loc[results_df["candidate_id"] == cid, "proportion"]
                ax.hist(proportions, bins=25, density=True, alpha=0.6,
                        color=palette[i], edgecolor="black",
                        label=f"Candidate {cid}", histtype="stepfilled")
            ax.set_xlabel("Proportion of Votes")
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend(frameon=True, title="Candidate ID")
            ax.set_xlim(0, 1)
        else:
            print(f"Error: Unknown plot_kind '{plot_kind}'. Please use 'distribution' or 'histogram'.")
            return

        plt.tight_layout()
        plt.show()