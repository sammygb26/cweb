import jax
import jax.numpy as jnp

@jax.jit
def cross_entropy(p : jax.Array, q : jax.Array) -> jax.Array:
    """
    Calculates the cross entropy of q from p.

    Args:
        p (jax.Array) : A tensor of target probability distributions (...,d).
        q (jax.Array) : A tensor of approximate probability distributions (...,d).

    Returns:
        jax.Array : A tensor of cross-entropies (...).
    """
    return -jnp.sum(p * jnp.log(q), axis=-1)