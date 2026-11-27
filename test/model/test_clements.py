from cweb.model import *
from cweb.helpers import random_phases

import jax
import jax.numpy as jnp

def test_clements_parsed_for_compatible_sublayer_dimensions():
    for m in [1, 3, 5, 8]:
        # Random columns and diag with extra dimensions for a data layer.
        di =                random_phases(jax.random.key(0), (m       ))
        tc = ClementsColumn(random_phases(jax.random.key(1), (m       )), is_primary=True)
        dc = ClementsColumn(random_phases(jax.random.key(2), (10, 3, m)), is_primary=True)

        # Unitary is created without error from above setup
        c = Clements(di, [tc, dc])
        c_u = c.get_unitary()
        c_eye = c_u @ jnp.matrix_transpose(c_u.conj())

        # Check it is unitary
        assert jnp.allclose(jnp.eye(m)[None,None], c_eye, atol=1e-6)
