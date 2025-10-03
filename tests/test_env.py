import jax.numpy as jnp

from analysis import DataCollector
from environment import Environment, Scheduler


class DummyVotingSystem:
    """Minimal mock voting system for testing purposes."""

    def __init__(self, name="Plurality Voting"):
        """Initialize the dummy voting system with a given name."""
        self.name = name

    def counting_votes(self, voters, candidates):
        """Return the first candidate as winner."""
        # Ensures correct IDs are returned to avoid KeyError
        return candidates[0].id, {
            c.id: (len(voters) if i == 0 else 0) for i, c in enumerate(candidates)
        }


def test_scheduler_step_increments():
    """Test that scheduler step increments count and calls agent step."""

    class DummyAgent:
        """Minimal mock agent to check scheduler calls step."""

        def __init__(self):
            """Initialize dummy agent with a called flag."""
            self.called = False

        def step(self, env):
            """Set flag to True when step is executed."""
            self.called = True

    agent = DummyAgent()
    scheduler = Scheduler([agent])
    scheduler.step(environment={})
    assert agent.called
    assert scheduler.step_count == 1


def test_environment_creation():
    """Test that environment creates voters and candidates."""
    env = Environment(
        num_voters=3,
        num_candidates=2,
        num_preferences=2,
        voting_system=DummyVotingSystem(),
    )
    assert len(env.voters) == 3
    assert len(env.candidates) == 2
    assert isinstance(env.datacollector, DataCollector)


def test_get_new_agent_id():
    """Test that new agent IDs are incremented correctly."""
    env = Environment(1, 1, 1, DummyVotingSystem())
    first_id = env._get_new_agent_id()
    second_id = env._get_new_agent_id()
    assert second_id == first_id + 1


def test_update_public_poll():
    """Test that public poll is updated with normalized probabilities."""
    env = Environment(2, 2, 1, DummyVotingSystem(), use_theory_of_mind=True)
    env._update_public_poll({env.candidates[0].id: 1, env.candidates[1].id: 3})
    assert jnp.allclose(env.public_poll, jnp.array([0.25, 0.75]))


def test_gather_agent_data_shapes():
    """Test that gathered agent data arrays have expected shapes."""
    env = Environment(2, 1, 2, DummyVotingSystem())
    mus, pis, vols, budgets = env._gather_agent_data()
    assert mus.shape[0] == 2
    assert pis.shape[0] == 2
    assert vols.shape[0] == 2
    assert budgets.shape[0] == 2


def test_environment_step_and_run():
    """Test that environment step updates state and run collects data."""
    env = Environment(3, 2, 1, DummyVotingSystem())
    env.step()
    assert env.winner_id is not None
    assert isinstance(env.last_round1_results, dict)

    env.run(2)  # run 2 steps
    df = env.datacollector.get_dataframe()
    assert not df.empty
    assert "winner_id" in df.columns
