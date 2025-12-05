import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

from eci.utils import generate_candidates, generate_observations, kl_divergence


class TestKLDivergence:
    """Unit tests for the kl_divergence function."""

    def test_identity(self):
        """Test: KL(P || P) must be 0."""
        mean = jnp.array([1.0, 2.0])
        prec = jnp.array([1.0, 0.5])
        kl = kl_divergence(mean, prec, mean, prec)
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_theoretical_value(self):
        """Test: Comparison with a theoretical value."""
        kl = kl_divergence(
            mean_belief=0.0, precision_belief=1.0, mean_pref=1.0, precision_pref=1.0
        )
        np.testing.assert_allclose(kl, 0.5, atol=1e-6)

    def test_non_negative(self):
        """Test: KL must always be >= 0."""
        rng = np.random.default_rng(42)
        m1 = jnp.array(rng.normal(size=10))
        m2 = jnp.array(rng.normal(size=10))
        p1 = jnp.array(np.abs(rng.normal(size=10))) + 0.1
        p2 = jnp.array(np.abs(rng.normal(size=10))) + 0.1
        kl = kl_divergence(m1, p1, m2, p2)
        assert jnp.all(kl >= -1e-7)

    def test_asymmetry(self):
        """Test: KL(A || B) != KL(B || A) generally."""
        args_a = (0.0, 1.0)
        args_b = (1.0, 0.5)
        kl_ab = kl_divergence(*args_a, *args_b)
        kl_ba = kl_divergence(*args_b, *args_a)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(kl_ab, kl_ba)

    def test_broadcasting(self):
        """Test: Correct handling of scalars vs arrays (Broadcasting)."""
        mean_belief = 0.0
        prec_belief = 1.0
        mean_pref = jnp.array([0.0, 1.0, 2.0])
        prec_pref = jnp.array([1.0, 1.0, 1.0])
        kl = kl_divergence(mean_belief, prec_belief, mean_pref, prec_pref)
        assert kl.shape == (3,)
        assert kl[0] == 0.0

    def test_jit_compatibility(self):
        """Test: That the function is currently JIT-compatible."""
        jitted_kl = jit(kl_divergence)
        jitted_kl(
            jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.0]), jnp.array([1.0])
        )


class TestGenerateObservations:
    """Unit tests for observation simulation (generate_observations)."""

    def test_output_shape_scenario_1(self):
        """Test: Output shape (n_steps, n_nodes) for Scenario 1."""
        n_nodes, n_steps = 5, 100
        obs = generate_observations(n_nodes=n_nodes, n_steps=n_steps, scenario=1)
        assert obs.shape == (n_steps, n_nodes)

    def test_output_shape_scenario_2(self):
        """Test: Output shape for Scenario 2."""
        n_nodes, n_steps = 3, 50
        obs = generate_observations(
            n_nodes=n_nodes, n_steps=n_steps, scenario=2, shock_pattern="phase"
        )
        assert obs.shape == (n_steps, n_nodes)

    def test_value_bounds(self):
        """Test: Values bounded between [0, 1] (clipping)."""
        obs = generate_observations(n_nodes=2, n_steps=50, dispersion=2.0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_reproducibility(self):
        """Test: Random seed must guarantee identical results."""
        kwargs = {"n_nodes": 4, "n_steps": 20, "scenario": 2, "shock_pattern": "trend"}
        obs1 = generate_observations(**kwargs)
        obs2 = generate_observations(**kwargs)
        np.testing.assert_array_equal(obs1, obs2)

    def test_trend_logic_execution(self):
        """Test: The complex 'trend' loop executes without error."""
        obs = generate_observations(
            n_nodes=1,
            n_steps=30,
            scenario=2,
            shock_pattern="trend",
            trend_shape="linear",
        )
        assert obs.shape == (30, 1)

    def test_custom_timings(self):
        """Test: Usage of custom shock timings."""
        obs = generate_observations(
            n_nodes=1,
            n_steps=100,
            scenario=2,
            shock_pattern="phase",
            shock_time=20,
            recovery_time=80,
        )
        assert obs.shape == (100, 1)

    def test_error_invalid_scenario(self):
        """Test: Error on invalid scenario ID."""
        with pytest.raises(ValueError, match="Scenario must be 1 or 2"):
            generate_observations(n_nodes=1, n_steps=10, scenario=3)

    def test_error_invalid_pattern(self):
        """Test: Error on invalid shock pattern."""
        with pytest.raises(ValueError, match="Invalid shock_pattern"):
            generate_observations(
                n_nodes=1, n_steps=10, scenario=2, shock_pattern="unknown"
            )

    @pytest.mark.parametrize("pattern", ["phase", "sudden", "trend"])
    def test_generate_scenario_patterns(self, pattern):
        """Test: Generating all valid shock patterns for Scenario 2."""
        n_steps = 100
        s_time = 30
        r_time = 70

        result = generate_observations(
            scenario=2,
            n_nodes=1,
            n_steps=n_steps,
            shock_pattern=pattern,
            shock_time=s_time,
            recovery_time=r_time,
        )

        assert len(result) == n_steps


class TestGenerateCandidates:
    """Unit tests for candidate generation (generate_candidates)."""

    def test_random_structure(self):
        """Test: Default structure (list of tuples)."""
        n_c, n_p = 5, 3
        candidates = generate_candidates(n_candidates=n_c, n_preferences=n_p)

        assert len(candidates) == n_c
        assert isinstance(candidates[0], tuple)
        assert len(candidates[0]) == 2
        assert candidates[0][0].shape == (n_p,)
        assert candidates[0][1].shape == (n_p,)

    def test_validity_values(self):
        """Test: Finite values and positive precisions."""
        candidates = generate_candidates(n_candidates=5, n_preferences=2)
        for means, precs in candidates:
            assert np.all(precs >= 0)
            assert np.all(np.isfinite(means))

    def test_manual_injection(self):
        """Test: Manual data injection (Must be exact)."""
        n_c, n_p = 2, 2
        in_means = np.array([[1.0, 2.0], [3.0, 4.0]])
        in_precs = np.array([[0.1, 0.2], [0.3, 0.4]])
        candidates = generate_candidates(
            n_c, n_p, manual_means=in_means, manual_precisions=in_precs
        )
        np.testing.assert_array_equal(candidates[0][0], in_means[0])
        np.testing.assert_array_equal(candidates[1][1], in_precs[1])

    def test_error_shape_mismatch(self):
        """Test: Errors if manual inputs have the wrong shape."""
        n_c, n_p = 2, 2
        bad_shape = np.zeros((5, 5))
        good_shape = np.zeros((2, 2))

        with pytest.raises(ValueError, match="means must have shape"):
            generate_candidates(
                n_c, n_p, manual_means=bad_shape, manual_precisions=good_shape
            )
        with pytest.raises(ValueError, match="precisions must have shape"):
            generate_candidates(
                n_c, n_p, manual_means=good_shape, manual_precisions=bad_shape
            )

    def test_fallback_logic(self):
        """Test: If only one manual array is provided, fallback to random."""
        candidates = generate_candidates(
            2, 2, manual_means=np.zeros((2, 2)), manual_precisions=None
        )
        assert not np.array_equal(candidates[0][0], np.zeros(2))
