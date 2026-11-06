from cweb.model import *
from cweb.helpers import random_phases

import jax
import jax.numpy as jnp

def test_clements_parsed_for_compatible_sublayer_dimensions():
    for m in [1, 3, 5, 8]:
        trainable_layer_params = make_clements_primary_column_params(random_phases(jax.random.key(0), (m)))
        data_layer_params      = make_clements_primary_column_params(random_phases(jax.random.key(0), (10, 3, m)))

        params = make_clements_params([
            trainable_layer_params,
            data_layer_params
        ])

        clements_unitary = parse_clements(params)
        clements_eye = clements_unitary @ jnp.matrix_transpose(clements_unitary.conj())
        assert jnp.allclose(jnp.eye(m)[None,None], clements_eye, atol=1e-6)
