# strategies/decision/hgf_strategy.py
import jax
import jax.numpy as jnp
from jax import vmap

from .base_decision import DecisionStrategy
from .kl_divergence import calculate_kl_divergence

# This is now a standalone, pure JAX function
def calculate_vote_from_params(key, mu_belief, prec_belief, mean_pref, precision_pref, candidates_params):
    """
    Calculates a single vote based purely on JAX-compatible arrays.
    
    Args:
        candidates_params: A tuple of (candidate_means, candidate_precisions)
    """
    candidate_means, candidate_precisions = candidates_params
    
    current_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref)
    
    def get_utility(c_mean, c_prec):
        expected_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, c_mean, c_prec)
        return current_dissatisfaction - expected_dissatisfaction
    
    # Use vmap internally to calculate utility for all candidates efficiently
    utilities = vmap(get_utility)(candidate_means, candidate_precisions)
    
    softmax_probs = jax.nn.softmax(utilities)
    return jax.random.categorical(key, jnp.log(softmax_probs))


class HGFDecisionStrategy(DecisionStrategy):
    """The 'decide' method is kept for potential non-vmap use, but is not part of the core fix."""
    def decide(self, agent, environment, candidates: list, key) -> int:
        # This logic is now handled externally before the vmap call.
        print("Warning: HGFDecisionStrategy.decide() was called directly.")
        return 0