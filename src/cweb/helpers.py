import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax import ShapeDtypeStruct

def random_phases(key : ArrayLike, shape : ShapeDtypeStruct):
    return jax.random.uniform(key, shape) * 2 * jnp.pi

def trees_all_close(tree1, tree2, *args, **kwargs):
    close_tree = jax.tree.map(lambda a, b: jnp.allclose(a, b, *args, **kwargs), tree1, tree2)
    return jax.tree.all(close_tree)