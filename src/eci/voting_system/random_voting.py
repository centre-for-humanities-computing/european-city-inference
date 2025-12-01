import jax
import jax.numpy as jnp
from pyhgf.model import Network

from eci.core_logic import kl_divergence


def random_vote(env, key, *args, **kwargs) -> dict:
    """Each agent in env votes randomly for one of the available options."""
    options = jnp.array([c.id for c in env.candidates])
    for agent in env.agents:
        agent.last_vote = int(jax.random.choice(key, options))

    result = {
        "votes": {agent.id: agent.last_vote for agent in env.agents},
        "counts": {
            int(opt): sum(agent.last_vote == int(opt) for agent in env.agents)
            for opt in options
        },
    }
    return result


def template_function(env, *args, **kwargs) -> dict:
    """Template function for a voting system."""
    for agent in env.agents:
        network_template = Network()
        network_template.add_nodes(kind="binary-state", n_nodes=env.num_preferences)
        for i in range(env.num_preferences):
            network_template.add_nodes(value_children=i)
        network_template.input_data(input_data=env.input_data())

    for agent in env.agents:
        agent.compute_dissatisfaction()

    for agent in env.agents:
        agent.select_candidate()

    result = {
        "votes": {agent.id: agent.last_vote for agent in env.agents},
        "summary": env.summarize_results()
        if hasattr(env, "summarize_results")
        else None,
    }
    return result


# observation function
def network_building(env):
    """Build the network template for the voting system."""
    network_template = Network()
    network_template.add_nodes(kind="binary-state", n_nodes=env.num_preferences)
    for i in range(env.num_preferences):
        network_template.add_nodes(value_children=i)
    return network_template


# observation function
def observation_function(env, network):
    """Feed input data to the network."""
    return network.input_data(input_data=env.input_data())


def extract_node_beliefs(edges, input_idxs, node_trajectories):
    """Extract expected mean and precision for the given input indices."""
    preferences_idx = [edges[idx].value_parents[0] for idx in input_idxs]
    expected_mean = jnp.array(
        [node_trajectories[i]["expected_mean"][-1] for i in preferences_idx]
    )
    expected_precision = jnp.array(
        [node_trajectories[i]["expected_precision"][-1] for i in preferences_idx]
    )
    return expected_mean, expected_precision


def compute_current_dissatisfaction(expected_mean, expected_precision, attributes):
    """Compute total current dissatisfaction relative to the baseline preferences."""
    current_dissatisfaction = kl_divergence(
        expected_mean,
        expected_precision,
        attributes[-1]["preferences"]["mean"],
        attributes[-1]["preferences"]["precision"],
    )
    return jnp.sum(current_dissatisfaction)


def eval_candidate(
    expected_mean, expected_precision, total_current_dissatisfaction, candidate
):
    """Compute relative improvement in dissatisfaction for a single candidate."""
    mean_pref, prec_pref = candidate
    expected_dissatisfaction = kl_divergence(
        expected_mean, expected_precision, mean_pref, prec_pref
    )
    return total_current_dissatisfaction - jnp.sum(expected_dissatisfaction)


def compute_candidate_preferences(
    expected_mean, expected_precision, attributes, candidates
):
    """Compute relative preference scores for each candidate."""
    total_current_dissatisfaction = compute_current_dissatisfaction(
        expected_mean, expected_precision, attributes
    )
    candidate_preferences = jnp.array(
        [
            eval_candidate(
                expected_mean, expected_precision, total_current_dissatisfaction, c
            )
            for c in candidates
        ]
    )
    return candidate_preferences
