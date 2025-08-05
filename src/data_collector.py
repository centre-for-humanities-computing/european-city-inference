import pandas as pd

class DataCollector:
    """Collecte les données de chaque simulation."""
    def __init__(self):
        self.results = []

    def record_simulation(self, sim_number: int, election_results: dict):
        """Enregistre le résultat d'une simulation."""
        record = {
            "simulation_number": sim_number,
            "winner_index": election_results['winner_index'],
            "finalists": election_results.get('finalists_indices')
        }
        for i, prop in enumerate(election_results['round_1_proportions']):
            record[f'r1_prop_cand_{i}'] = prop
            
        if election_results['round_2_proportions'] is not None:
            finalist_indices = election_results['finalists_indices']
            for i, prop in enumerate(election_results['round_2_proportions']):
                original_idx = finalist_indices[i]
                record[f'r2_prop_cand_{original_idx}'] = prop

        self.results.append(record)

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne les résultats collectés sous forme de DataFrame Pandas."""
        return pd.DataFrame(self.results).fillna(0)