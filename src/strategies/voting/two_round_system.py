# strategies/voting/two_round_system.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from .base_voting import VotingSystem
from ..decision.hgf_strategy import calculate_vote_from_params # Import the new pure function

class TwoRoundSystem(VotingSystem):
    """Implémentation d'un système de vote majoritaire à deux tours."""

    def run_election(self, agents: list, environment, key) -> dict:
        # --- STEP 1: DATA EXTRACTION (Python World) ---
        # Extract all agent parameters into numpy arrays BEFORE JAX sees them.
        all_mu_beliefs, all_prec_beliefs, all_mean_prefs, all_precision_prefs = [], [], [], []
        for agent in agents:
            mu_b, prec_b, mean_p, prec_p = agent.get_decision_parameters()
            all_mu_beliefs.append(mu_b)
            all_prec_beliefs.append(prec_b)
            all_mean_prefs.append(mean_p)
            all_precision_prefs.append(prec_p)

        # Convert lists of arrays into single large JAX arrays.
        # Shape will be (n_agents, n_preferences)
        agents_mu_beliefs = jnp.array(all_mu_beliefs)
        agents_prec_beliefs = jnp.array(all_prec_beliefs)
        agents_mean_prefs = jnp.array(all_mean_prefs)
        agents_precision_prefs = jnp.array(all_precision_prefs)

        # --- STEP 2: VMAP THE PURE JAX FUNCTION ---
        # `in_axes` maps over the first dimension of our new JAX arrays.
        # `candidates_params` is broadcasted (`None`).
        vmap_vote = vmap(calculate_vote_from_params, in_axes=(0, 0, 0, 0, 0, None))
        
        keys = jax.random.split(key, len(agents))

        # --- STEP 3: RUN THE ELECTION ---
        # Round 1
        r1_candidate_params = self._get_candidate_params(environment.candidates)
        votes_r1 = vmap_vote(keys, agents_mu_beliefs, agents_prec_beliefs, agents_mean_prefs, agents_precision_prefs, r1_candidate_params)
        
        counts_r1 = np.bincount(np.array(votes_r1), minlength=len(environment.candidates))
        proportions_r1 = counts_r1 / np.sum(counts_r1) if np.sum(counts_r1) > 0 else np.zeros_like(counts_r1)
        
        if len(counts_r1) < 2 or np.max(proportions_r1) > 0.5:
            winner_idx = np.argmax(counts_r1)
            return {"winner_index": winner_idx, "round_1_proportions": proportions_r1, "round_2_proportions": None, "finalists_indices": None}

        top2_indices = np.argsort(counts_r1)[-2:][::-1]
        top2_candidates = [environment.candidates[i] for i in top2_indices]
        
        # Round 2
        r2_candidate_params = self._get_candidate_params(top2_candidates)
        votes_r2 = vmap_vote(keys, agents_mu_beliefs, agents_prec_beliefs, agents_mean_prefs, agents_precision_prefs, r2_candidate_params)
        
        counts_r2 = np.bincount(np.array(votes_r2), minlength=len(top2_candidates))
        proportions_r2 = counts_r2 / np.sum(counts_r2) if np.sum(counts_r2) > 0 else np.zeros_like(counts_r2)
        
        winner_local_idx = np.argmax(counts_r2)
        winner_original_idx = top2_indices[winner_local_idx]

        return {"winner_index": winner_original_idx, "round_1_proportions": proportions_r1, "round_2_proportions": proportions_r2, "finalists_indices": top2_indices}

    @staticmethod
    def _get_candidate_params(candidates: list):
        """Helper to convert list of candidate tuples to JAX arrays."""
        c_means = jnp.array([[p[0] for p in c] for c in candidates])
        c_precisions = jnp.array([[1.0 / (p[1]**2) for p in c] for c in candidates])
        return c_means, c_precisions