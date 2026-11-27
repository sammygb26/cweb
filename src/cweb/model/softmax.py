
import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_vectors

@jax.jit
def softmax(input : jax.Array) -> jax.Array:
    """
    Softmax transforms a tensor of atrbitrary vectiors to a tensor of vectors representing discrete probability distributions.

    Args:
        input (jax.Array) : The tensor of vectors to be transformed (...,d).

    Returns:
        jax.Array : The transformed tensor of vectors (...,d).
    """
    validate_tensor_of_vectors(input) 

    exp = jnp.exp(input)
    return exp / jnp.sum(exp, axis=-1)[...,None]