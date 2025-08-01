# main.py

import jax
import jax.numpy as jnp
import pyhgf.model as pyhgf_model
from agents import Electeur, Candidat
from simulation_logic import get_vote_for_agent
# ... autres imports ...

def main():
    # Paramètres de simulation
    n_candidates = 6
    n_preferences = 3
    n_agents = 10
    simulations = 10
    
    # Générer les données d'entrée
    input_data = jnp.array(generate_observations(n_nodes=n_preferences, n_steps=1000, scenario=1))
    
    # Clé aléatoire pour JAX
    key = jax.random.PRNGKey(0)

    # Création des candidats (JNP arrays)
    # L'initialisation doit être JAX-compatible si possible
    key, subkey = jax.random.split(key)
    def init_candidate(k):
        # Utiliser jax.random pour l'aléatoire
        mu_sigma = jax.random.normal(k, shape=(n_preferences, 2))
        return Candidat(identifiant=jnp.array(0), preferences=mu_sigma)
    
    # On crée un tableau de candidats
    candidats_keys = jax.random.split(subkey, n_candidates)
    candidats = jax.vmap(init_candidate)(candidats_keys)

    # Création des agents (Électeurs)
    key, subkey = jax.random.split(key)
    def init_electeur(k):
        # Créer l'état initial du réseau pyhgf
        network_state = pyhgf_model.Network().get_state() # Supposons que pyhgf le permet
        preferences = jax.random.normal(k, shape=(n_preferences, 2))
        return Electeur(k, preferences, network_state)
        
    electeurs_keys = jax.random.split(subkey, n_agents)
    electeurs = jax.vmap(init_electeur)(electeurs_keys)

    # On pré-compile la fonction de vote vectorisée
    vmap_get_vote = jax.vmap(get_vote_for_agent, in_axes=(0, 0, None, None), out_axes=(0, 0))

    for sim_round in range(simulations):
        print(f"Simulation ronde {sim_round}")
        
        # Le code de ton notebook est basé sur plusieurs rounds de vote
        # On peut adapter cela ici.
        
        # --- PREMIER TOUR ---
        key, subkey = jax.random.split(key)
        electeurs, votes_1st_round = vmap_get_vote(jax.random.split(subkey, n_agents), electeurs, candidats, input_data)
        
        counts = jnp.bincount(votes_1st_round, length=n_candidates)
        # ... logique pour le top 2 ...

        # --- DEUXIÈME TOUR ---
        # Si un deuxième tour est nécessaire
        if len(jnp.where(counts > 0)[0]) >= 2:
            top2_indices = jnp.argsort(counts)[-2:][::-1]
            top_two_candidats = jax.tree_util.tree_map(lambda x: x[top2_indices], candidats)

            key, subkey = jax.random.split(key)
            electeurs, votes_2nd_round = vmap_get_vote(jax.random.split(subkey, n_agents), electeurs, top_two_candidats, input_data)

            counts_2nd = jnp.bincount(votes_2nd_round, length=2)
            # ... stockage des résultats ...
        
if __name__ == "__main__":
    main()