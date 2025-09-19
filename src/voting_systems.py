from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from agents import Candidate, Voter


class VotingSystem(ABC):
    """An interface for vote-counting systems.

    Notes
    -----
    This is an abstract base class (ABC) that defines the essential methods
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

        This method must also store the detailed results of the count in an
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
    This is also known as "first-past-the-post".
    """

    def __init__(self):
        """Initialize the PluralityVoting system."""
        self.last_results: dict[int, int] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
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
        actual_vote_ids = [candidate_ids[idx] for idx in vote_indices]

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
    """A system where voters rank candidates.

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

    This implementation uses a pre-computed ranking stored in the `last_vote`
    attribute of the Voter object.
    """

    def __init__(self):
        """Initialize the RankingVoting system."""
        self.last_results: dict[int, int] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
        return "Ranking Voting"


def counting_votes(
    self, voters: list[Voter], candidates: list[Candidate]
) -> tuple[int, dict[int, int]]:
    """Calculate the Borda count winner from voter rankings."""
    # Get the total number of candidates.
    num_candidates = len(candidates)
    # Handle the edge case where there are no candidates.
    if num_candidates == 0:
        self.last_results = {}
        return -1, {}

    # Initialize a dictionary to store the score for each candidate, starting at 0.
    points = {c.id: 0 for c in candidates}
    # Create an array of Borda points.
    borda_points = jnp.arange(num_candidates - 1, -1, -1)

    # Iterate through each voter to tabulate their ranked votes.
    for voter in voters:
        # Safely get the voter's ranked list of candidate indices.
        ranked_indices = getattr(voter, "last_vote", None)
        # Skip voters who have not submitted a valid, iterable vote.
        if ranked_indices is None or not hasattr(ranked_indices, "__iter__"):
            continue

        # Award points based on the rank of each candidate in the voter's list.
        for rank, candidate_idx in enumerate(ranked_indices):
            # Safety check to ensure the rank.
            if rank < len(borda_points) and int(candidate_idx) < len(candidates):
                # Get the candidate's unique ID from their index in the list.
                candidate_id = candidates[int(candidate_idx)].id
                # Add the corresponding Borda points to the candidate's total score.
                points[candidate_id] += borda_points[rank]

    # Convert scores to standard Python integers for compatibility.
    final_points = {cid: int(score) for cid, score in points.items()}
    # Store the final scores in the instance variable.
    self.last_results = final_points
    # -------------------------

    # If no votes were cast or all scores are zero, return no winner.
    if not final_points or sum(final_points.values()) == 0:
        return -1, self.last_results

    # Sort candidates by their final scores in descending order to find the winner.
    sorted_scores = sorted(final_points.items(), key=lambda item: item[1], reverse=True)

    # Assume no winner by default.
    winner_id = -1
    # Check for a tie between the top two candidates.
    if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
        # In case of a tie, winner_id remains -1.
        pass
    # If there's a clear winner (and at least one candidate was scored).
    elif sorted_scores:
        # The winner is the first candidate in the sorted list.
        winner_id = sorted_scores[0][0]

    # Return the winner's ID and the dictionary of all final scores.
    return winner_id, self.last_results


class QuadraticVoting(VotingSystem):
    """A system where voters allocate credits to express preference intensity.

    Attributes
    ----------
    VOTE_CREDITS_BUDGET : int
        The total number of credits each voter can allocate among candidates.
    last_results : dict[int, float]
        Stores the quadratic vote totals from the last election, mapping
        candidate IDs to their final scores.

    Notes
    -----
    In Quadratic Voting, each voter has a budget of "vote credits." They can
    allocate these credits to candidates to show the intensity of their preference.
    The number of official votes a candidate receives from a voter is the
    **square root** of the credits allocated.

    This system allows for nuanced preference expression while curbing the
    influence of overly passionate minorities. This implementation uses a
    `last_softmax_probs` attribute on the Voter object to determine how the
    credit budget is distributed.
    """

    VOTE_CREDITS_BUDGET = 100

    def __init__(self):
        """Initialize the QuadraticVoting system."""
        self.last_results: dict[int, float] = {}

    @property
    def name(self) -> str:
        """str: The name of the voting system."""
        return "Quadratic Voting"

    def counting_votes(
        self, voters: list[Voter], candidates: list[Candidate]
    ) -> tuple[int, dict[int, float]]:
        """Calculate the winner using the Quadratic Voting method.

        Parameters
        ----------
        voters : list[Voter]
            A list of Voter objects. Each voter is expected to have a
            `last_softmax_probs` attribute: a dictionary mapping candidate IDs
            to a probability-like score (summing to 1.0).
        candidates : list[Candidate]
            A list of all candidates in the election.

        Returns
        -------
        int
            The ID of the winning candidate. Returns -1 if there is a tie for
            first place or if no credits are allocated.
        """
        if not candidates:
            self.last_results = {}
            return -1, self.last_results

        total_quadratic_votes = {c.id: 0.0 for c in candidates}

        for voter in voters:
            probabilities_by_index = getattr(voter, "last_softmax_probs", None)
            if not isinstance(probabilities_by_index, dict):
                continue

            # **CORRECTION : Mapper l'index de la probabilité à l'ID du candidat**
            for candidate_idx, prob in probabilities_by_index.items():
                if candidate_idx < len(candidates):
                    # Récupérer le vrai ID du candidat
                    candidate_id = candidates[candidate_idx].id
                    credits_allocated = prob * self.VOTE_CREDITS_BUDGET
                    quadratic_votes = jnp.sqrt(credits_allocated)
                    total_quadratic_votes[candidate_id] += float(quadratic_votes)

        self.last_results = total_quadratic_votes
        if sum(total_quadratic_votes.values()) == 0:
            return -1, self.last_results

        # Trouver le gagnant (logique similaire à ci-dessus)
        sorted_scores = sorted(
            total_quadratic_votes.items(), key=lambda item: item[1], reverse=True
        )

        if not sorted_scores:
            return -1, self.last_results

        if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
            return -1, self.last_results

        # Get the winner ID from the sorted list
        winner_id = sorted_scores[0][0]
        # Return the ID and the results dictionary as a tuple
        return winner_id, self.last_results
