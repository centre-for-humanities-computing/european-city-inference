from voting_systems import (
    PluralityVoting,
    QuadraticVoting,
    QuadraticVotingBudget,
    RankingVoting,
)


# --- Dummy agents ---
class DummyCandidate:
    """Minimal mock candidate with an ID for testing."""

    def __init__(self, cid):
        """Initialize candidate with given ID."""
        self.id = cid


class DummyVoter:
    """Minimal mock voter with a last_vote attribute for testing."""

    def __init__(self, vote=None):
        """Initialize voter with an optional vote."""
        self.last_vote = vote


# ------------------------------
# Tests PluralityVoting
# ------------------------------
def test_plurality_voting_normal():
    """Test plurality voting."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [DummyVoter(0), DummyVoter(0), DummyVoter(1)]
    system = PluralityVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == 0
    assert results == {0: 2, 1: 1}


def test_plurality_voting_no_votes():
    """Test plurality voting when all voters abstain returns no winner."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [DummyVoter(None), DummyVoter(None)]
    system = PluralityVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == -1
    assert results == {0: 0, 1: 0}


def test_plurality_voting_tie():
    """Test plurality voting returns tie when votes are equal."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [DummyVoter(0), DummyVoter(1)]
    system = PluralityVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == -1
    assert results == {0: 1, 1: 1}


def test_plurality_voting_tom_name():
    """Test plurality voting system name includes ToM when enabled."""
    system = PluralityVoting(use_theory_of_mind=True)
    assert "ToM" in system.name


# ------------------------------
# Tests RankingVoting
# ------------------------------
def test_ranking_voting_borda_count():
    """Test ranking voting with Borda count produces valid results."""
    candidates = [DummyCandidate(0), DummyCandidate(1), DummyCandidate(2)]
    voters = [
        DummyVoter([0, 1, 2]),
        DummyVoter([1, 0, 2]),
    ]
    system = RankingVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert isinstance(winner, int)
    assert set(results.keys()) == {0, 1, 2}
    if winner == -1:
        assert results[0] == results[1] == max(results.values())
    else:
        assert results[winner] == max(results.values())


def test_ranking_voting_tie():
    """Test ranking voting."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [
        DummyVoter([0, 1]),
        DummyVoter([1, 0]),
    ]
    system = RankingVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == -1


def test_ranking_voting_no_candidates():
    """Test ranking voting returns no winner when no candidates exist."""
    voters = [DummyVoter([0])]
    system = RankingVoting()
    winner, results = system.counting_votes(voters, [])
    assert winner == -1
    assert results == {}


def test_ranking_voting_tom_name():
    """Test ranking voting system name includes ToM when enabled."""
    system = RankingVoting(use_theory_of_mind=True)
    assert "ToM" in system.name


# ------------------------------
# Tests QuadraticVoting
# ------------------------------
def test_quadratic_voting_counts():
    """Test quadratic voting counts votes with squared costs."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [
        DummyVoter([1.0, 2.0]),
        DummyVoter([0.0, 3.0]),
    ]
    system = QuadraticVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert results[1] > results[0]
    assert winner == 1


def test_quadratic_voting_tie():
    """Test quadratic voting returns tie when votes are symmetric."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [
        DummyVoter([1.0, 0.0]),
        DummyVoter([0.0, 1.0]),
    ]
    system = QuadraticVoting()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == -1


def test_quadratic_voting_no_candidates():
    """Test quadratic voting returns no winner when candidate list is empty."""
    system = QuadraticVoting()
    winner, results = system.counting_votes([], [])
    assert winner == -1
    assert results == {}


def test_quadratic_voting_tom_name():
    """Test quadratic voting system name includes ToM when enabled."""
    system = QuadraticVoting(use_theory_of_mind=True)
    assert "ToM" in system.name


# ------------------------------
# Tests QuadraticVotingBudget
# ------------------------------
def test_quadratic_voting_budget_counts():
    """Test quadratic voting with budget produces valid results."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [
        DummyVoter([2.0, 0.0]),
        DummyVoter([0.0, 5.0]),
    ]
    system = QuadraticVotingBudget()
    winner, results = system.counting_votes(voters, candidates)
    assert winner in results
    assert results[winner] == max(results.values())


def test_quadratic_voting_budget_tie():
    """Test quadratic voting with budget returns tie when votes are equal."""
    candidates = [DummyCandidate(0), DummyCandidate(1)]
    voters = [
        DummyVoter([1.0, 0.0]),
        DummyVoter([0.0, 1.0]),
    ]
    system = QuadraticVotingBudget()
    winner, results = system.counting_votes(voters, candidates)
    assert winner == -1


def test_quadratic_voting_budget_no_candidates():
    """Test quadratic voting with budget returns no winner when empty input."""
    system = QuadraticVotingBudget()
    winner, results = system.counting_votes([], [])
    assert winner == -1
    assert results == {}


def test_quadratic_voting_budget_tom_name():
    """Test quadratic voting with budget."""
    system = QuadraticVotingBudget(use_theory_of_mind=True)
    assert "ToM" in system.name
