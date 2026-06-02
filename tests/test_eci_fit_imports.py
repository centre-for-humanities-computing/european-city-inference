"""Smoke tests for the eci.fit skeleton."""

import pytest

from eci.fit import diagnostics, models, priors


def test_subpackage_imports():
    """Importing eci.fit and its submodules works."""
    assert priors is not None
    assert models is not None
    assert diagnostics is not None


def test_stubs_raise_not_implemented():
    """Stubbed functions fail loudly so they can't be called by accident."""
    with pytest.raises(NotImplementedError):
        priors.default_tau_p_prior()
    with pytest.raises(NotImplementedError):
        models.build_model({}, observed_votes=None)
    with pytest.raises(NotImplementedError):
        diagnostics.convergence_report(idata=None)
