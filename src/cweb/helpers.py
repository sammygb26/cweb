import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax import ShapeDtypeStruct

def random_phases(key : ArrayLike, shape : ShapeDtypeStruct):
    return jax.random.uniform(key, shape) * 2 * jnp.pi
