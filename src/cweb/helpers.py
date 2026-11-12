import jax
import jax.numpy as jnp
import zlib

from jax.typing import ArrayLike
from jax import ShapeDtypeStruct

def random_phases(key : ArrayLike, shape : ShapeDtypeStruct):
    return jax.random.uniform(key, shape) * 2 * jnp.pi

def fold_in_path(key : ArrayLike, path : any):
    path_str = str(any)
    crc = zlib.crc32(bytes(path_str, encoding='utf-8'))
    return jax.random.fold_in(key, crc)


def randomize_tree_phases(key, tree : dict):
    def random_phases_with_path(path : str, array : jnp.array):
        subkey = fold_in_path(key, path)
        return random_phases(subkey, array.shape)

    return jax.tree.map_with_path(random_phases_with_path, tree)


def trees_all_close(tree1 : dict, tree2 : dict, *args, **kwargs):
    close_tree = jax.tree.map(lambda a, b: jnp.allclose(a, b, *args, **kwargs), tree1, tree2)
    return jax.tree.all(close_tree)