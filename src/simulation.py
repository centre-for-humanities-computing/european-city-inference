# simulation_logic.py

import jax
import jax.numpy as jnp
import pyhgf
from agents import Electeur, Candidat
from pyhgf_utils import hgf_update_state # Voir la section suivante

def calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref):
    # La fonction est déjà JAX-compatible, c'est bien.
    # ... (le code que tu as déjà) ...
    var_belief = jnp.where(prec_belief > 0, 1 / prec_belief, jnp.inf)
    var_pref = jnp.where(precision_pref > 0, 1 / precision_pref, jnp.inf)

    kl = jnp.log(jnp.sqrt(var_pref) / jnp.sqrt(var_belief)) + \
         (var_belief + (mu_belief - mean_pref) ** 2) / (2 * var_pref) - 0.5
    return kl

# @jax.jit pour compiler cette fonction une seule fois
@jax.jit
def get_vote_for_agent(rng_key, electeur: Electeur, candidats: Candidat, input_data):
    """
    Fonction pure pour obtenir le vote d'un seul agent.
    Elle prend l'agent en entrée et retourne le résultat de son vote.
    """
    # 1. Mettre à jour l'état HGF avec les nouvelles observations
    # On suppose que pyhgf a une fonction JAX-compatible pour la mise à jour
    new_pyhgf_network_state = hgf_update_state(electeur.pyhgf_network_state, input_data)

    # 2. Extraire les croyances du nouvel état du réseau
    # L'accès aux attributs doit être adapté pour être JAX-compatible
    # Par exemple, si l'état est un dictionnaire, utiliser un get ou un jnp.array
    # Ici, nous supposons que pyhgf_network_state a une structure fixe
    mu_belief = jnp.array([
        new_pyhgf_network_state[i]["expected_mean"][-1]
        for i in range(len(electeur.preferences))
    ])
    prec_belief = jnp.array([
        new_pyhgf_network_state[i]["expected_precision"][-1]
        for i in range(len(electeur.preferences))
    ])

    # 3. Calculer la "dissatisfaction" actuelle
    mean_pref = electeur.preferences[:, 0]
    precision_pref = electeur.preferences[:, 1]
    current_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, mean_pref, precision_pref)
    total_current_dissatisfaction = jnp.sum(current_dissatisfaction)
    
    # 4. Calculer la "dissatisfaction" pour chaque candidat (vectorisation manuelle)
    # On utilise vmap ici car on a un tableau de candidats
    def dissatisfaction_for_candidate(candidate_preferences):
        candidate_mean_pref = candidate_preferences[:, 0]
        candidate_precision_pref = candidate_preferences[:, 1]
        expected_dissatisfaction = calculate_kl_divergence(mu_belief, prec_belief, candidate_mean_pref, candidate_precision_pref)
        return jnp.sum(expected_dissatisfaction)

    total_expected_dissatisfactions = jax.vmap(dissatisfaction_for_candidate)(candidats.preferences)

    candidate_preferences_scores = total_current_dissatisfaction - total_expected_dissatisfactions
    
    # 5. Obtenir la décision de vote
    softmax_probs = jax.nn.softmax(candidate_preferences_scores)
    vote_decision = jax.random.categorical(rng_key, jnp.log(softmax_probs))

    # Retourner le nouvel état de l'électeur et le vote
    # On crée une nouvelle instance d'Electeur avec l'état HGF mis à jour
    # Les préférences ne changent pas pour le moment
    new_electeur = Electeur(electeur.key, electeur.preferences, new_pyhgf_network_state)
    return new_electeur, vote_decision

@jax.jit
def decision_modele_base(electeur, candidats):
    # Logique coûteuse : simuler les conséquences du vote
    # ... utiliser le réseau pyhgf pour des projections ...
    return jnp.argmax(simulated_outcome_values)

@jax.jit
def decision_modele_libre(electeur, candidats):
    # Logique simple : se baser sur la dernière expérience réussie
    # ... utiliser l'historique ou une préférence fixe ...
    return electeur.last_successful_vote

@jax.jit
def get_vote_for_agent(rng_key, electeur: Electeur, candidats: Candidat, input_data):
    # ... (mise à jour de l'état HGF) ...

    # Choisir la fonction de décision en fonction de l'état
    chosen_decision_fn = jax.lax.cond(
        electeur.decision_mode == 1,
        lambda e, c: decision_modele_base(e, c),
        lambda e, c: decision_modele_libre(e, c),
        electeur, candidats
    )
    vote_decision = chosen_decision_fn
    # ... (le reste de la fonction) ...
    return new_electeur, vote_decision