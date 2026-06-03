import jax.numpy as jnp
from jax.typing import ArrayLike


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
    voter = next(v for v in env.voters if v.id == voter_id)
    return {
        "expected_mean": voter.trajectory[0]["expected_mean"],
        "expected_precision": voter.trajectory[0]["expected_precision"],
        "observations": env.input_data[:, pref_idx],
        "preference_params": (
            voter.preferences["mean"][pref_idx],
            voter.preferences["precision"][pref_idx],
        ),
        "title_suffix": f"for Voter {voter_id}",
    }


def _extract_env_data_vectorized(env):
    """Extract per-agent belief / preference / candidate arrays from an env.

    Returns the canonical ``data`` dict that every voting rule and
    response function consumes:

    ``{"beliefs": {"mean", "precision"},
       "preferences": {"mean", "precision"},
       "candidates":  {"mean", "precision"}}``
    """
    pref_idx_list = env.preferences_idx
    policy_means = jnp.stack([c.policy["mean"].ravel() for c in env.candidates])
    policy_precs = jnp.stack([c.policy["precision"].ravel() for c in env.candidates])
    means_belief = jnp.stack(
        [env.last_attributes[i]["expected_mean"] for i in pref_idx_list], axis=-1
    )
    precs_belief = jnp.stack(
        [env.last_attributes[i]["expected_precision"] for i in pref_idx_list], axis=-1
    )
    p_idx_jax = jnp.array(pref_idx_list)
    agent_pref_means = env.last_attributes[-1]["preferences"]["mean"][:, p_idx_jax]
    agent_pref_precs = env.last_attributes[-1]["preferences"]["precision"][:, p_idx_jax]
    return {
        "beliefs": {"mean": means_belief, "precision": precs_belief},
        "preferences": {"mean": agent_pref_means, "precision": agent_pref_precs},
        "candidates": {"mean": policy_means, "precision": policy_precs},
    }
