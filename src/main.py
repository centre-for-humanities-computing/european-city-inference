from agents.voter import Voter
from agents.candidate import Candidate
from environment.environment import Environment
from mediator.election_manager import ElectionManager
from voting_systems.majority_voting import MajorityVoting
from voting_systems.quadratic_voting import QuadraticVoting
import config

def main():
    """Point d'entrée principal de la simulation."""
    # 1. Initialisation de l'environnement et des agents
    env = Environment(num_dimensions=config.NUM_DIMENSIONS)
    
    voters = [Voter(id=i, num_dimensions=config.NUM_DIMENSIONS) for i in range(config.NUM_VOTERS)]
    candidates = [Candidate(id=i, num_dimensions=config.NUM_DIMENSIONS) for i in range(config.NUM_CANDIDATES)]
    
    # 2. Enregistrement des électeurs comme observateurs de l'environnement
    for voter in voters:
        env.register(voter)
        
    # 3. Sélection du système de vote via la stratégie
    if config.VOTING_SYSTEM == 'majority':
        voting_strategy = MajorityVoting()
    elif config.VOTING_SYSTEM == 'quadratic':
        voting_strategy = QuadraticVoting()
    else:
        raise ValueError("Système de vote non reconnu.")

    # 4. Initialisation du médiateur
    election_manager = ElectionManager(voters, candidates, voting_strategy)
    
    # 5. Lancement de la simulation
    print("Début de la simulation électorale...")
    election_manager.run_election()
    
    # 6. Affichage des résultats
    winner = election_manager.get_winner()
    print(f"Le vainqueur de l'élection est : {winner}")
    
    # (Optionnel) Lancer la visualisation
    # plotter.plot_results(...)

if __name__ == "__main__":
    main()
