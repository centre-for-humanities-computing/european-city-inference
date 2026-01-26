import re
from typing import Any, Counter, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm


class SimulationAdapter:
    """Serves as a bridge between the Environment and SimulationVisualizer."""

    @staticmethod
    def prepare_preference_data(env) -> pd.DataFrame:
        """Prepare preference distribution data for visualization."""
        rows = []
        x_vals = np.linspace(-3, 4, 400)

        # Extract Candidates
        for c in env.candidates:
            for p_idx, (m, p) in enumerate(
                zip(c.policy["mean"], c.policy["precision"])
            ):
                pdf = norm.pdf(x_vals, loc=m, scale=p)
                rows.extend(
                    [
                        {
                            "group": "Candidate",
                            "id": f"C{c.id}",
                            "preference": f"{p_idx}",
                            "x": x,
                            "pdf": y,
                        }
                        for x, y in zip(x_vals, pdf)
                    ]
                )

        # Extract Voters
        for v in env.voters:
            for p_idx, (m, p) in enumerate(
                zip(v.preferences["mean"], v.preferences["precision"])
            ):
                pdf = norm.pdf(x_vals, loc=m, scale=p)
                rows.extend(
                    [
                        {
                            "group": "Voter",
                            "id": f"V{v.id}",
                            "preference": f"{p_idx}",
                            "x": x,
                            "pdf": y,
                        }
                        for x, y in zip(x_vals, pdf)
                    ]
                )

        return pd.DataFrame(rows)

    @staticmethod
    def extract_vote_counts(env_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert JAX/Pandas DataFrame into a list of proportions dynamically."""
        records = []

        def clean_votes(votes_obj) -> List[int]:
            if votes_obj is None:
                return []
            return np.array(votes_obj).flatten().tolist()

        vote_cols = [c for c in env_df.columns if "vote" in c.lower()]

        for _, row in env_df.iterrows():
            sim_id = row.get("simulation_id", row.name)

            for col_name in vote_cols:
                digits = re.findall(r"\d+", col_name)

                if digits:
                    round_id = int(digits[-1])
                else:
                    round_id = col_name

                raw_votes = row[col_name]
                votes = clean_votes(raw_votes)
                total = len(votes)

                if total > 0:
                    counts = Counter(votes)
                    for cand_id, count in counts.items():
                        records.append(
                            {
                                "simulation_id": sim_id,
                                "round": round_id,
                                "candidate_id": int(cand_id),
                                "proportion": count / total,
                                "total_votes": total,
                            }
                        )

        return records

    @staticmethod
    def get_voter_trajectory_data(env, voter_id: int, pref_idx: int = 0):
        """Retrieve specific arrays for a single voter's belief trajectory."""
        voter = next(v for v in env.voters if v.id == voter_id)
        return {
            "means": voter.trajectory["expected_mean"][voter.id],
            "precisions": voter.trajectory["precision"][voter.id],
            "observations": env.input_data[:, pref_idx],
            "preference_params": (
                voter.preferences["mean"][pref_idx],
                voter.preferences["precision"][pref_idx],
            ),
            "title_suffix": f"for Voter {voter_id}",
        }

    @staticmethod
    def process_simulation_results(simulations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw simulation results into a DataFrame suitable for analysis."""
        all_votes = []
        for sim in simulations:
            all_votes.extend(sim.get("vote_round_1", []))
            all_votes.extend(sim.get("vote_final_round_2", []))

        # Ensure we cover all candidates that appeared
        all_candidates = np.unique(all_votes)

        data_list = []
        for i, sim_data in enumerate(simulations):
            winner = sim_data.get("final_winner")

            for round_name, round_key in [
                ("Tour 1", "vote_round_1"),
                ("Tour 2", "vote_round_2"),
            ]:
                votes = np.array(sim_data.get(round_key, []))

                # Robustly calculate proportions
                if len(votes) == 0:
                    continue

                unique, counts = np.unique(votes, return_counts=True)
                counts_map = dict(zip(unique, counts))
                total_votes = len(votes)

                for cand in all_candidates:
                    prop = counts_map.get(cand, 0) / total_votes
                    data_list.append(
                        {
                            "simulation": i,
                            "candidat": cand,
                            "proportion": prop,
                            "round": round_name,
                            "a_gagne_final": (cand == winner),
                        }
                    )

        return pd.DataFrame(data_list)
