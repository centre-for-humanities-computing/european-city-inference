from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
from jax import vmap
from jax.tree_util import Partial
from pyhgf.model import Network

from eci.population import Population
from eci.world import World


@dataclass(frozen=True)
class HGFConfig:
    """Parameters for the underlying HGF network.

    Attributes
    ----------
    precision_state :
        Initial precision for each continuous state node.
    update_type :
        The pyhgf network update rule. Defaults to ``"unbounded"`` to
        match the current behaviour.
    """

    precision_state: float = 100.0
    update_type: str = "unbounded"

    @classmethod
    def from_env_config(cls, env_config) -> "HGFConfig":
        """Build an HGFConfig from a flat :class:`EnvConfig`."""
        return cls(precision_state=env_config.precision_state)


# ---------------------------------------------------------------------------
# PerceptualModel (stateful, NOT a pytree)
# ---------------------------------------------------------------------------
class PerceptualModel:
    """Stateful wrapper around a configured :class:`pyhgf.model.Network`.

    Parameters
    ----------
    n_preferences :
        Number of preference dimensions the HGF tracks.
    config :
        HGF-specific parameters; see :class:`HGFConfig`.

    Attributes
    ----------
    network : pyhgf.model.Network
        The underlying HGF network. Mutated during inference (each
        ``run`` call re-writes the agent-specific attributes), which is
        why this class is not a pytree.
    """

    def __init__(self, n_preferences: int, config: HGFConfig = HGFConfig()):
        """Build the model and its HGF network (see class docstring for parameters)."""
        self.n_preferences = n_preferences
        self.config = config
        self.network = self._build_network()

    def _build_network(self) -> Network:
        """Build a fresh HGF network with the configured hierarchy."""
        network = Network(update_type=self.config.update_type)
        network.add_nodes(
            kind="continuous-state",
            n_nodes=self.n_preferences,
            precision=self.config.precision_state,
            expected_precision=self.config.precision_state,
        )
        # Each input node gets a value parent and a volatility parent.
        for i in range(self.n_preferences):
            network.add_nodes(value_children=i)
            network.add_nodes(volatility_children=i)
        return network

    @classmethod
    def from_env_config(cls, env_config) -> "PerceptualModel":
        """Build a PerceptualModel from a flat :class:`EnvConfig`."""
        return cls(
            n_preferences=env_config.num_preferences,
            config=HGFConfig.from_env_config(env_config),
        )

    def _run_one_agent(
        self,
        mu: jax.Array,
        pi: jax.Array,
        tonic_volatility: jax.Array,
        observations: jax.Array,
        network: Network,
    ) -> Tuple[Any, Any]:
        """Run HGF inference for a single agent. Mutates ``network``."""
        # Inject this agent's preferences on the (private) preferences node.
        network.attributes[-1]["preferences"] = {"mean": mu, "precision": pi}
        # Override tonic_volatility on every input node and its value parent.
        for p, idx in enumerate(network.input_idxs):
            network.attributes[idx]["tonic_volatility"] = tonic_volatility
            value_parent_idx = self.n_preferences + 2 * p
            network.attributes[value_parent_idx]["tonic_volatility"] = tonic_volatility
        # Feed the observation stream and run.
        network.input_data(input_data=observations)
        return network.last_attributes, network.node_trajectories

    def run(self, population: Population, world: World) -> Dict[str, Any]:
        """Run HGF inference for every voter, vmapped over the population.

        Returns a dict with:
            - ``last_attributes``: pytree, per-voter final HGF state
            - ``node_trajectories``: pytree, per-voter full trajectory
            - ``preferences_idx``: list of int, network input node indices

        These three fields together let downstream code reconstruct each
        voter's posterior belief at every time step.
        """
        vmap_single = Partial(
            self._run_one_agent,
            observations=world.observations,
            network=self.network,
        )
        last_attributes, node_trajectories = vmap(vmap_single)(
            population.voter_means,
            population.voter_precisions,
            population.voter_volatilities,
        )
        return {
            "last_attributes": last_attributes,
            "node_trajectories": node_trajectories,
            "preferences_idx": list(self.network.input_idxs),
        }
