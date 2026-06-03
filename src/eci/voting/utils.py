import jax
import jax.numpy as jnp


def _find_top_k_winners(
    votes_array: jnp.ndarray, num_candidates: int, k: int
) -> jnp.ndarray:
    """Return the indices of the k candidates with the most votes.

    Ties are broken by ``jax.lax.top_k`` (lowest index wins on tie).
    """
    counts = jnp.bincount(votes_array.astype(jnp.int32), length=num_candidates)
    _, winners = jax.lax.top_k(counts, k=k)
    return winners


def _find_winner(votes_array: jnp.ndarray, num_candidates: int) -> jnp.ndarray:
    """Return the index of the single candidate with the most votes.

    Convenience for the common ``k=1`` case of :func:`_find_top_k_winners`.
    """
    return _find_top_k_winners(votes_array, num_candidates, k=1)[0]
