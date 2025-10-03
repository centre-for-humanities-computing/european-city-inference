from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from eci.agents import Candidate, Voter


class VotingSystem(ABC):
    """An interface for vote-counting systems.

    Notes
    -----
    An abstract base class (ABC) that defines the essential methods
    and properties any concrete voting system must implement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the voting system."""
        pass

    @abstractmethod
    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> Tuple[int, Dict[Any, Any]]:
        """Count votes and determine the winning candidate.

        Store the detailed results of the count in an
        internal attribute for later inspection.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects participating in the election.
        candidates : list[Candidate]
            A list of Candidate objects running in the election.

        Returns
        -------
        tuple[int, dict[Any, Any]]
            A tuple containing:
            - The ID of the winning candidate (-1 for a tie).
            - A dictionary with the final vote counts/scores.
        """
        pass


class PluralityVoting(VotingSystem):
    """A vote-counting system based on the plurality rule.

    Attributes
    ----------
    last_results : dict[int, int]
        A dictionary storing the results of the last vote count, mapping
        candidate IDs to their vote totals.

    Notes
    -----
    The rule is simple: the candidate with the most first-preference votes wins.
    """

    def __init__(self, use_theory_of_mind: bool = False):
        """Initialize the PluralityVoting system.

        Parameters
        ----------
        use_theory_of_mind : bool, optional
            If True, enables Theory of Mind modifications to the system's
            behavior and name (default is False).
        """
        self.last_results: dict[int, int] = {}
        self._use_tom = use_theory_of_mind  # Stores the state

    @property
    def name(self) -> str:
        """The name of the system, modified if Theory of Mind is active."""
        if self._use_tom:
            return "Plurality Voting (ToM)"
        return "Plurality Voting"

    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> tuple[int, dict[int, int]]:
        """Count votes based on the plurality rule.

        The method counts the number of times each candidate's ID appears
        in the voters' `last_vote` attribute.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects. It is assumed that `voter.last_vote`
            contains the ID of the chosen candidate.
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        tuple[int, dict[int, int]]
            A tuple containing:
            - The ID of the winning candidate (-1 for a tie or no votes).
            - A dictionary with the final vote counts for each candidate.
        """
        vote_indices = [v.last_vote for v in voters if v.last_vote is not None]

        if not vote_indices:
            self.last_results = {c.id: 0 for c in candidates}
            return -1, self.last_results

        candidate_ids = [c.id for c in candidates]
        actual_vote_ids = [
            candidate_ids[int(idx)] for idx in vote_indices if isinstance(idx, int)
        ]

        self.last_results = {c.id: 0 for c in candidates}

        counts = Counter(actual_vote_ids)
        self.last_results.update(counts)

        if not counts:
            return -1, self.last_results

        most_common = counts.most_common(2)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return -1, self.last_results  # Tie

        winner_id = most_common[0][0]
        return winner_id, self.last_results


class RankingVoting(VotingSystem):
    """A system where voters rank candidates, scored using Borda count.

    Attributes
    ----------
    last_results : dict[int, int]
        Stores the Borda point totals from the last election, mapping
        candidate IDs to their final scores.

    Notes
    -----
    In the Borda count method, each voter provides a ranking of candidates.
    Points are awarded based on rank. For N candidates, a first-place rank
    gets N-1 points, a second-place rank gets N-2 points, and so on, down to
    0 points for the last-place candidate. The candidate with the highest
    total score wins.
    """

    def __init__(self, use_theory_of_mind: bool = False):
        """Initialize the RankingVoting system.

        Parameters
        ----------
        use_theory_of_mind : bool, optional
            If True, enables Theory of Mind modifications to the system's
            behavior and name (default is False).
        """
        self.last_results: dict[int, int] = {}
        self._use_tom = use_theory_of_mind

    @property
    def name(self) -> str:
        """The name of the system, modified if Theory of Mind is active."""
        if self._use_tom:
            return "Ranking Voting (ToM)"
        return "Ranking Voting"

    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> tuple[int, dict[int, int]]:
        """Count votes using the Borda count method.

        Parameters
        ----------
        voters : list[Voter]
            A list of voters, where each voter's `last_vote` is an ordered
            iterable of candidate indices representing their ranking.
        candidates : list[Candidate]
            A list of Candidate objects participating in the election.

        Returns
        -------
        tuple[int, dict[int, int]]
            A tuple containing:
            - The ID of the winning candidate, or -1 in case of a tie or if
            no valid votes are cast.
            - A dictionary mapping candidate IDs to their final Borda scores.
        """
        num_candidates = len(candidates)
        if num_candidates == 0:
            self.last_results = {}
            return -1, {}

        # Initialize scores for each candidate to 0.
        points = {c.id: 0 for c in candidates}
        borda_points = jnp.arange(num_candidates - 1, -1, -1)

        # Tabulate votes
        for voter in voters:
            ranked_indices = getattr(voter, "last_vote", None)
            if ranked_indices is None or not hasattr(ranked_indices, "__iter__"):
                continue

            for rank, candidate_idx in enumerate(ranked_indices):
                if rank < len(borda_points) and int(candidate_idx) < len(candidates):
                    candidate_id = candidates[int(candidate_idx)].id
                    points[candidate_id] += borda_points[rank]

        self.last_results = {cid: int(score) for cid, score in points.items()}

        if not self.last_results or sum(self.last_results.values()) == 0:
            return -1, self.last_results

        # Sort candidates by score
        sorted_scores = sorted(
            self.last_results.items(), key=lambda item: item[1], reverse=True
        )
        max_score = sorted_scores[0][1]
        winners = [cid for cid, score in sorted_scores if score == max_score]

        # Tie → -1
        if len(winners) > 1:
            return -1, self.last_results

        winner_id = winners[0]
        return winner_id, self.last_results


class QuadraticVoting(VotingSystem):
    """Votes are summed from voters' quadratic allocations.

    Attributes
    ----------
    last_results : dict[int, float]
        Stores the summed quadratic vote totals from the last election.
    """

    def __init__(self, use_theory_of_mind: bool = False):
        """Initialize the QuadraticVoting system.

        Parameters
        ----------
        use_theory_of_mind : bool, optional
            If True, enables Theory of Mind modifications to the system's
            behavior and name (default is False).
        """
        self.last_results: dict[int, float] = {}
        self._use_tom = use_theory_of_mind

    @property
    def name(self) -> str:
        """The name of the system, modified if Theory of Mind is active."""
        if self._use_tom:
            return "Quadratic Voting (ToM)"
        return "Quadratic Voting"

    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> tuple[int, dict[int, float]]:
        """Count votes based on summing each voter's quadratic allocations.

        Parameters
        ----------
        voters : list[Voter]
            Voters, where `voter.last_vote` is an array of votes allocated
            to candidates.
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        tuple[int, dict[int, float]]
            A tuple containing:
            - The ID of the winning candidate (-1 for a tie or no votes).
            - A dictionary with the final summed votes for each candidate.
        """
        if not candidates:
            self.last_results = {}
            return -1, self.last_results

        total_quadratic_votes = {c.id: 0.0 for c in candidates}

        for voter in voters:
            # Assumes voter.last_vote contains the array of quadratic votes.
            votes_by_index = getattr(voter, "last_vote", None)
            if votes_by_index is None or not hasattr(votes_by_index, "__iter__"):
                continue

            for candidate_idx, num_votes in enumerate(votes_by_index):
                if candidate_idx < len(candidates):
                    candidate_id = candidates[candidate_idx].id
                    total_quadratic_votes[candidate_id] += float(num_votes)

        self.last_results = total_quadratic_votes
        if sum(self.last_results.values()) == 0:
            return -1, self.last_results

        sorted_scores = sorted(
            self.last_results.items(), key=lambda item: item[1], reverse=True
        )
        if not sorted_scores:
            return -1, self.last_results
        # Tie check for floating-point numbers
        if (
            len(sorted_scores) > 1
            and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 1e-9
        ):
            return -1, self.last_results

        winner_id = sorted_scores[0][0]
        return winner_id, self.last_results


class QuadraticVotingBudget(VotingSystem):
    """A system where votes are summed from voters' budget-constrained.

    Attributes
    ----------
    last_results : dict[int, float]
        Stores the summed quadratic vote totals from the last election.
    """

    def __init__(self, use_theory_of_mind: bool = False):
        self.last_results: dict[int, float] = {}
        self._use_tom = use_theory_of_mind

    @property
    def name(self) -> str:
        """The name of the system."""
        if self._use_tom:
            return "Quadratic Voting (Budget ToM)"
        return "Quadratic Voting (Budget)"

    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> tuple[int, dict[int, float]]:
        """QuadraticVoting counting method."""
        if not candidates:
            self.last_results = {}
            return -1, self.last_results

        total_quadratic_votes = {c.id: 0.0 for c in candidates}

        for voter in voters:
            votes_by_index = getattr(voter, "last_vote", None)
            if votes_by_index is None or not hasattr(votes_by_index, "__iter__"):
                continue

            for candidate_idx, num_votes in enumerate(votes_by_index):
                if candidate_idx < len(candidates):
                    candidate_id = candidates[candidate_idx].id
                    total_quadratic_votes[candidate_id] += float(num_votes)

        self.last_results = total_quadratic_votes
        if sum(self.last_results.values()) == 0:
            return -1, self.last_results

        sorted_scores = sorted(
            self.last_results.items(), key=lambda item: item[1], reverse=True
        )
        if not sorted_scores:
            return -1, self.last_results

        if (
            len(sorted_scores) > 1
            and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 1e-9
        ):
            return -1, self.last_results

        winner_id = sorted_scores[0][0]
        return winner_id, self.last_results
