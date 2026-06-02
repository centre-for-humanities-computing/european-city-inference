"""Smoke tests for the eci.data skeleton.

Until concrete implementations land in v0.2, these tests just verify
that the package layout is importable and the stubs raise
``NotImplementedError`` (so calling them by accident fails loudly).
"""

import pytest

from eci.data import loaders, schemas, transformers
from eci.data.schemas import (
    BallotCast,
    Experiment,
    Participant,
)


def test_subpackage_imports():
    """Importing eci.data and its submodules works."""
    assert schemas is not None
    assert loaders is not None
    assert transformers is not None


def test_experiment_dataclass_round_trip():
    """Schemas are usable as ordinary dataclasses."""
    p = Participant(participant_id="p1")
    b = BallotCast(participant_id="p1", election_id="e1", candidate_id="c1")
    exp = Experiment(
        experiment_id="e1",
        schema_version=1,
        participants=[p],
        ballots=[b],
    )
    assert exp.participants[0].participant_id == "p1"
    assert exp.ballots[0].weight == 1.0


def test_stubs_raise_not_implemented():
    """Stubbed loaders fail loudly so they can't be called by accident."""
    with pytest.raises(NotImplementedError):
        loaders.load_from_csv("dummy.csv")
    with pytest.raises(NotImplementedError):
        loaders.load_from_url("https://example.com/dump.csv")
    with pytest.raises(NotImplementedError):
        transformers.experiment_to_voting_data(
            Experiment(
                experiment_id="e1",
                schema_version=1,
                participants=[],
                ballots=[],
            )
        )
