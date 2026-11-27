from jax.typing import ArrayLike

import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_vectors

def get_random_batch(key : ArrayLike, batch_size : int, data : jax.Array, step : int = 0) -> jax.Array:
    """
    Gets a random batch drawing with replacement from some data.

    Args:
        key (ArrayLike)  : The key to be used for random sampling.
        batch_size (int) : The size of the batch (b) to be sampled.
        data (jax.Array) : The data to be sampled (N,...).
        step (int)       : The step of the batch (to be folded into the key).

    Returns:
        jax.Array : The sampled data (b,...).
    """
    validate_tensor_of_vectors(data) 

    step_key = jax.random.fold_in(key, step)
    indices = jax.random.choice(step_key, jnp.arange(data.shape[0]), (batch_size,))

    return data[indices,...]