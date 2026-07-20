from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from eci.decision.types import ElectionData


def kl_divergence(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> ArrayLike:
    r"""KL divergence between two univariate Gaussians, given by precisions.

    Parameters
    ----------
    mean_belief, precision_belief :
        Parameters of the belief distribution :math:`q`.
    mean_pref, precision_pref :
        Parameters of the preference distribution :math:`p`.

    Returns
    -------
    Element-wise KL :math:`\mathrm{KL}(q \| p)`. Broadcasting follows
    NumPy / JAX rules.
    """
    mean_belief = jnp.asarray(mean_belief)
    precision_belief = jnp.asarray(precision_belief)
    mean_pref = jnp.asarray(mean_pref)
    precision_pref = jnp.asarray(precision_pref)
    return 0.5 * (
        jnp.log(precision_belief / precision_pref)
        + (precision_pref / precision_belief)
        + (precision_pref * (mean_belief - mean_pref) ** 2)
        - 1.0
    )


def cross_entropy(
    mean_belief: ArrayLike,
    precision_belief: ArrayLike,
    mean_pref: ArrayLike,
    precision_pref: ArrayLike,
) -> ArrayLike:
    r"""Cross-entropy :math:`H(q, p)` between two univariate Gaussians (precisions).

    Same argument order as :func:`kl_divergence`: ``q`` is
    ``(mean_belief, precision_belief)`` and ``p`` is
    ``(mean_pref, precision_pref)``.

    Equals :math:`\mathrm{KL}(q \| p) + H(q)`, i.e. the KL term plus the entropy
    of ``q``. Compared to KL, the extra entropy term means a diffuse (low
    precision) ``q`` is penalised even when its mean matches ``p``.

    Returns
    -------
    Element-wise cross-entropy. Broadcasting follows NumPy / JAX rules.
    """
    mean_belief = jnp.asarray(mean_belief)
    precision_belief = jnp.asarray(precision_belief)
    mean_pref = jnp.asarray(mean_pref)
    precision_pref = jnp.asarray(precision_pref)
    return 0.5 * (
        jnp.log(2.0 * jnp.pi)
        - jnp.log(precision_pref)
        + (precision_pref / precision_belief)
        + (precision_pref * (mean_belief - mean_pref) ** 2)
    )


def get_voter_trajectory_data(env, voter_id: int, pref_idx: int = 0):
    """Retrieve arrays for plotting one voter's belief trajectory.

    Parameters
    ----------
    env :
        The simulation environment containing agents and candidates.
    voter_id :
        The ID of the voter to retrieve data for.
    pref_idx :
        Preference-dimension index to extract.
    """
    voter = next(
        candidate_voter
        for candidate_voter in env.voters
        if candidate_voter.id == voter_id
    )
    preference_node_index = env.preferences_idx[pref_idx]
    trajectory = voter.trajectory[preference_node_index]
    return {
        "expected_mean": trajectory["expected_mean"],
        "expected_precision": trajectory["expected_precision"],
        "observations": env.input_data[:, pref_idx],
        "preference_params": (
            voter.preferences["mean"][pref_idx],
            voter.preferences["precision"][pref_idx],
        ),
        "title_suffix": f"for Voter {voter_id}",
    }


def _extract_env_data_vectorized(env) -> ElectionData:
    """Extract per-agent belief / preference / candidate arrays from an env.

    Returns the canonical ``data`` dict that every voting rule and
    response function consumes:

    ``{"beliefs": {"mean", "precision"},
       "preferences": {"mean", "precision"},
       "candidates":  {"mean", "precision"}}``
    """
    preference_node_indices = env.preferences_idx
    candidate_policy_means = jnp.stack(
        [candidate.policy["mean"].ravel() for candidate in env.candidates]
    )
    candidate_policy_precisions = jnp.stack(
        [candidate.policy["precision"].ravel() for candidate in env.candidates]
    )
    belief_means = jnp.stack(
        [
            env.last_attributes[node_index]["expected_mean"]
            for node_index in preference_node_indices
        ],
        axis=-1,
    )
    belief_precisions = jnp.stack(
        [
            env.last_attributes[node_index]["expected_precision"]
            for node_index in preference_node_indices
        ],
        axis=-1,
    )
    preference_indices = jnp.array(preference_node_indices)
    agent_preference_means = env.last_attributes[-1]["preferences"]["mean"][
        :, preference_indices
    ]
    agent_preference_precisions = env.last_attributes[-1]["preferences"]["precision"][
        :, preference_indices
    ]
    return {
        "beliefs": {
            "mean": belief_means,
            "precision": belief_precisions,
        },
        "preferences": {
            "mean": agent_preference_means,
            "precision": agent_preference_precisions,
        },
        "candidates": {
            "mean": candidate_policy_means,
            "precision": candidate_policy_precisions,
        },
    }
