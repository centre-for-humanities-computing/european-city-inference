# agents.py

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import pyhgf.model as pyhgf_model

# En supposant que l'état du réseau pyhgf est lui-même un pytree ou peut être aplati
# Si ce n'est pas le cas, il faudra peut-être l'adapter manuellement.
# On part du principe que pyhgf est bien conçu pour être compatible avec JAX.

@register_pytree_node_class
class Electeur:
    def __init__(self, key, preferences, pyhgf_network_state):
        self.key = key
        self.preferences = preferences  # Un tableau JAX (e.g., jnp.array([mu, sigma, ...]))
        self.pyhgf_network_state = pyhgf_network_state # L'état du réseau pyhgf

    def tree_flatten(self):
        # On définit les "feuilles" (données à traiter par JAX) et les "auxiliaires"
        children = (self.preferences, self.pyhgf_network_state)
        aux_data = (self.key,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        key = aux_data[0]
        preferences, pyhgf_network_state = children
        return cls(key, preferences, pyhgf_network_state)

@register_pytree_node_class
class Candidat:
    def __init__(self, identifiant, preferences):
        self.identifiant = identifiant
        self.preferences = preferences # JNP array de (mu, sigma)

    def tree_flatten(self):
        children = (self.preferences,)
        aux_data = (self.identifiant,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        identifiant = aux_data[0]
        preferences = children[0]
        return cls(identifiant, preferences)