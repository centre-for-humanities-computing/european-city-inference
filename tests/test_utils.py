import numpy as np
import pytest

from utils import generate_candidates, generate_observations


def test_generate_observations_shape_scenario1():
    """Test that scenario 1 returns a 2D array with correct dimensions."""
    obs = generate_observations(n_nodes=3, n_steps=50, scenario=1)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (50, 3)
    assert np.all((0 <= obs) & (obs <= 1))


def test_generate_observations_shape_scenario2_phase():
    """Test scenario 2 with shock_pattern='phase' returns correct shape."""
    obs = generate_observations(
        n_nodes=2, n_steps=60, scenario=2, shock_pattern="phase"
    )
    assert obs.shape == (60, 2)
    assert np.all((0 <= obs) & (obs <= 1))


def test_generate_observations_shape_scenario2_sudden():
    """Test scenario 2 with shock_pattern='sudden' returns correct shape."""
    obs = generate_observations(
        n_nodes=1, n_steps=40, scenario=2, shock_pattern="sudden"
    )
    assert obs.shape == (40, 1)


def test_generate_observations_shape_scenario2_trend_linear():
    """Test scenario 2 with shock_pattern='trend' and trend_shape='linear'."""
    obs = generate_observations(
        n_nodes=1,
        n_steps=50,
        scenario=2,
        shock_pattern="trend",
        trend_shape="linear",
    )
    assert obs.shape == (50, 1)


def test_generate_observations_invalid_scenario():
    """Test that invalid scenario raises ValueError."""
    with pytest.raises(ValueError):
        generate_observations(n_nodes=1, n_steps=10, scenario=99)


def test_generate_observations_invalid_pattern():
    """Test that invalid shock_pattern raises ValueError."""
    with pytest.raises(ValueError):
        generate_observations(
            n_nodes=1, n_steps=10, scenario=2, shock_pattern="invalid"
        )


def test_generate_candidates_random_shape():
    """Test random candidate generation returns correct structure."""
    cands = generate_candidates(3, 2)
    assert len(cands) == 3
    for mus, pis in cands:
        assert mus.shape == (2,)
        assert pis.shape == (2,)
        assert np.all(pis >= 0)


def test_generate_candidates_manual_ok():
    """Test manual candidate generation with valid input shapes."""
    means = np.ones((2, 3))
    precisions = np.full((2, 3), 2.0)
    cands = generate_candidates(2, 3, manual_means=means, manual_precisions=precisions)
    assert len(cands) == 2
    for mus, pis in cands:
        assert np.all(mus == 1.0)
        assert np.all(pis == 2.0)


def test_generate_candidates_manual_invalid_shape_means():
    """Test that invalid manual means shape raises ValueError."""
    means = np.ones((2, 2))
    precisions = np.ones((2, 3))
    with pytest.raises(ValueError):
        generate_candidates(2, 3, manual_means=means, manual_precisions=precisions)


def test_generate_candidates_manual_invalid_shape_precisions():
    """Test that invalid manual precisions shape raises ValueError."""
    means = np.ones((2, 3))
    precisions = np.ones((2, 2))
    with pytest.raises(ValueError):
        generate_candidates(2, 3, manual_means=means, manual_precisions=precisions)
