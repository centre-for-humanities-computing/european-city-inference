"""Tests for the Expected Free Energy decision model.

Covers the three pieces that make up the active-inference vote:
:func:`_fuse_belief_candidate` (the Bayesian update that turns a candidate's
platform into an expected post-election world), :func:`_get_expected_free_energy`
(the risk + ambiguity score of that world), and :func:`response_function_efe`
(the precision-weighted softmax over the negative scores).
"""

import jax
import jax.numpy as jnp
import pytest

from eci.decision import _get_expected_free_energy, response_function_efe
from eci.decision.utilities import _fuse_belief_candidate
from eci.utils import cross_entropy, kl_divergence
from eci.voting import _vote_plurality


@pytest.fixture
def key():
    """Random key for testing."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def four_candidate_data():
    """One voter at -4.5 wanting +6; four candidates ordered worst → best."""
    return {
        "beliefs": {"mean": jnp.array([[-4.5]]), "precision": jnp.array([[1.0]])},
        "preferences": {"mean": jnp.array([[6.0]]), "precision": jnp.array([[1.0]])},
        "candidates": {
            "mean": jnp.array([[-4.5], [-1.5], [1.5], [4.5]]),
            "precision": jnp.ones((4, 1)),
        },
    }


def _efe(data, **kw):
    return _get_expected_free_energy(
        data["beliefs"]["mean"],
        data["beliefs"]["precision"],
        data["preferences"]["mean"],
        data["preferences"]["precision"],
        data["candidates"]["mean"],
        data["candidates"]["precision"],
        **kw,
    )


class TestFuseBeliefCandidate:
    """The Bayesian update that turns a platform into an expected future world."""

    def test_precisions_add_and_mean_is_precision_weighted(self):
        """Conjugate Gaussian update: precisions add, means average by precision."""
        b_mean = jnp.array([[0.0], [2.0]])
        b_prec = jnp.array([[1.0], [3.0]])
        c_mean = jnp.array([[4.0], [-2.0]])
        c_prec = jnp.array([[1.0], [1.0]])

        mean, prec = _fuse_belief_candidate(b_mean, b_prec, c_mean, c_prec)

        assert mean.shape == (2, 2, 1)
        assert prec.shape == (2, 2, 1)
        # agent 0 (belief 0, precision 1) vs candidate 0 (mean 4, precision 1)
        assert float(prec[0, 0, 0]) == pytest.approx(2.0)
        assert float(mean[0, 0, 0]) == pytest.approx(2.0)  # halfway
        # agent 1 (belief 2, precision 3) vs candidate 1 (mean -2, precision 1)
        assert float(prec[1, 1, 0]) == pytest.approx(4.0)
        assert float(mean[1, 1, 0]) == pytest.approx((2.0 * 3 + -2.0 * 1) / 4)

    def test_fused_world_is_sharper_than_the_belief(self):
        """Combining two sources can only reduce uncertainty."""
        b_prec = jnp.array([[1.0, 2.0]])
        _, prec = _fuse_belief_candidate(
            jnp.zeros((1, 2)), b_prec, jnp.zeros((3, 2)), jnp.full((3, 2), 0.5)
        )
        assert bool(jnp.all(prec > b_prec[:, None, :]))

    def test_vague_candidate_barely_moves_the_world(self):
        """A low-precision platform leaves the belief roughly where it was."""
        b_mean = jnp.array([[0.0]])
        mean, _ = _fuse_belief_candidate(
            b_mean, jnp.array([[10.0]]), jnp.array([[5.0]]), jnp.array([[1e-4]])
        )
        assert float(mean[0, 0, 0]) == pytest.approx(0.0, abs=1e-3)


class TestExpectedFreeEnergy:
    """The risk + ambiguity score of that expected future world."""

    def test_equals_risk_plus_ambiguity(self, four_candidate_data):
        """G = KL(q_future || P) + H(q_future), the cross-entropy decomposition."""
        d = four_candidate_data
        future_mean, future_prec = _fuse_belief_candidate(
            d["beliefs"]["mean"],
            d["beliefs"]["precision"],
            d["candidates"]["mean"],
            d["candidates"]["precision"],
        )
        p_mean = d["preferences"]["mean"][:, None, :]
        p_prec = d["preferences"]["precision"][:, None, :]

        risk = jnp.sum(kl_divergence(future_mean, future_prec, p_mean, p_prec), axis=-1)
        ambiguity = jnp.sum(0.5 * jnp.log(2 * jnp.pi * jnp.e / future_prec), axis=-1)

        assert jnp.allclose(_efe(d), risk + ambiguity, atol=1e-5)

    def test_matches_cross_entropy_of_the_fused_world(self, four_candidate_data):
        """G is exactly H(q_future, P), summed over preference dimensions."""
        d = four_candidate_data
        future_mean, future_prec = _fuse_belief_candidate(
            d["beliefs"]["mean"],
            d["beliefs"]["precision"],
            d["candidates"]["mean"],
            d["candidates"]["precision"],
        )
        expected = jnp.sum(
            cross_entropy(
                future_mean,
                future_prec,
                d["preferences"]["mean"][:, None, :],
                d["preferences"]["precision"][:, None, :],
            ),
            axis=-1,
        )
        assert jnp.allclose(_efe(d), expected, atol=1e-6)

    def test_shape_and_ranking(self, four_candidate_data):
        """Shape is (n_agents, n_candidates) and the closest platform scores best."""
        free_energy = _efe(four_candidate_data)
        assert free_energy.shape == (1, 4)
        # Preference is +6; candidates run -4.5 -> +4.5, so G must decrease.
        assert bool(jnp.all(jnp.diff(free_energy[0]) < 0))
        assert int(jnp.argmin(free_energy[0])) == 3

    def test_sums_over_preference_dimensions(self):
        """A multi-dimensional agent's G is the sum of its per-dimension scores."""
        two_dim = {
            "mean": jnp.array([[0.0, 0.0]]),
            "precision": jnp.array([[1.0, 1.0]]),
        }
        both = _get_expected_free_energy(
            two_dim["mean"],
            two_dim["precision"],
            jnp.array([[1.0, 3.0]]),
            jnp.ones((1, 2)),
            jnp.array([[0.5, 0.5]]),
            jnp.ones((1, 2)),
        )
        per_dim = [
            _get_expected_free_energy(
                two_dim["mean"][:, i : i + 1],
                two_dim["precision"][:, i : i + 1],
                jnp.array([[1.0, 3.0]])[:, i : i + 1],
                jnp.ones((1, 1)),
                jnp.array([[0.5, 0.5]])[:, i : i + 1],
                jnp.ones((1, 1)),
            )
            for i in range(2)
        ]
        assert float(both[0, 0]) == pytest.approx(
            float(per_dim[0][0, 0]) + float(per_dim[1][0, 0]), rel=1e-5
        )


class TestResponseFunctionEFE:
    """The precision-weighted softmax vote over the negative EFE scores."""

    def test_return_shapes_and_distribution(self, four_candidate_data, key):
        """Shapes follow the ResponseFunction contract; softmax is a distribution."""
        vote, softmax_probs, utilities, next_key = response_function_efe(
            four_candidate_data, key
        )
        assert vote.shape == (1,)
        assert softmax_probs.shape == (1, 4)
        assert utilities.shape == (1, 4)
        assert jnp.allclose(jnp.sum(softmax_probs, axis=1), 1.0, atol=1e-5)
        assert not jnp.array_equal(jnp.asarray(next_key), jnp.asarray(key))

    def test_utilities_are_negative_gamma_scaled_efe(self, four_candidate_data, key):
        """candidate_utilities are exactly ``-gamma * G(c)``."""
        _, _, utilities, _ = response_function_efe(four_candidate_data, key, gamma=2.5)
        assert jnp.allclose(utilities, -2.5 * _efe(four_candidate_data), atol=1e-5)

    def test_favours_the_lowest_efe_candidate(self, four_candidate_data, key):
        """The modal vote is the candidate minimising Expected Free Energy."""
        _, softmax_probs, _, _ = response_function_efe(four_candidate_data, key)
        assert int(jnp.argmax(softmax_probs[0])) == int(
            jnp.argmin(_efe(four_candidate_data)[0])
        )

    def test_gamma_controls_decisiveness(self, four_candidate_data, key):
        """Decisiveness rises monotonically with gamma; gamma = 0 is uniform."""
        peaks = [
            float(jnp.max(response_function_efe(four_candidate_data, key, gamma=g)[1]))
            for g in (0.0, 0.05, 0.5, 5.0)
        ]
        assert peaks == sorted(peaks)
        # gamma = 0 zeroes every utility, so the vote is exactly uniform.
        assert peaks[0] == pytest.approx(0.25)
        # large gamma commits almost entirely to the lowest-EFE candidate
        assert peaks[-1] > 0.99

    def test_belief_changes_the_ranking(self, four_candidate_data, key):
        """The belief enters through the fusion, so it can move the favourite.

        This is what separates EFE from the direct response functions, where the
        belief only scales decisiveness. A belief pinned hard at -4.5 drags every
        fused world leftward, so the platform nearest the *preference* is no
        longer automatically the one nearest the resulting world.
        """
        far = _efe(four_candidate_data)
        stubborn = {
            **four_candidate_data,
            "beliefs": {
                "mean": jnp.array([[-4.5]]),
                "precision": jnp.array([[50.0]]),
            },
        }
        assert not jnp.allclose(far, _efe(stubborn))
        # A near-immovable belief flattens the differences between platforms.
        assert float(jnp.ptp(_efe(stubborn)[0])) < float(jnp.ptp(far[0]))

    def test_mask_excludes_candidates(self, four_candidate_data, key):
        """A False entry in the mask gets -inf utility and zero probability."""
        mask = jnp.array([True, True, True, False])
        vote, softmax_probs, utilities, _ = response_function_efe(
            four_candidate_data, key, mask
        )
        assert bool(jnp.isneginf(utilities[0, 3]))
        assert float(softmax_probs[0, 3]) == pytest.approx(0.0)
        assert int(vote[0]) != 3

    def test_plugs_into_plurality(self, four_candidate_data, key):
        """Drop-in for the ResponseFunction protocol used by the voting rules."""
        result = _vote_plurality(four_candidate_data, response_function_efe, key)
        assert result["votes_per_candidate"].shape == (4,)
        assert int(result["winner"]) == 3
