from cweb.utils import *
from cweb.helpers import random_phases, trees_all_close

import jax
import jax.numpy as jnp

import os

from test_helpers.filesystem import folder

def test_data_logger_creates_folder(folder):
    assert not os.path.exists(folder)
    DataLogger(folder)
    assert os.path.exists(folder)

def test_data_logger_works_with_premade_folder(folder):
    os.makedirs(folder)
    DataLogger(folder)

def test_data_logger_saves_jax_tensor(folder):
    tensor = jax.random.uniform(jax.random.key(0), (4, 6, 7))

    dl = DataLogger(folder)

    dl.save('some_tensor.ts', tensor)
    loaded_tensor = dl.load('some_tensor.ts')

    assert jnp.allclose(tensor, loaded_tensor)

def test_data_logger_save_jax_pytree(folder):
    key = jax.random.key(0)

    pytree = {
        'something' : random_phases(key, (1, 2, 3)),
        'nothing' : [],
        'something_deep' : {
            'a' : random_phases(key, (1, 2)),
            'b' : random_phases(key, (2, 1))
        }
    }

    dl = DataLogger(folder)

    dl.save('some_pytree.pt', pytree)
    loaded_pytree = dl.load('some_pytree.pt')


    assert trees_all_close(pytree, loaded_pytree)
