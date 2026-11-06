from cweb.model import *

import jax
import jax.numpy as jnp

def test_two_mode_clements_primary_is_equal_to_mzi():
    m = 2

    phases = jax.random.uniform(jax.random.key(0), (3, 4, m)) * 2 * jnp.pi

    mzi_params      = make_mach_zender_interferometer_params(phases=phases)
    clements_params = make_clements_primary_column_params(phases=phases)

    mzis             = parse_mach_zender_interferometer(mzi_params)
    clements_columns = parse_clements_column(clements_params)

    assert jnp.allclose(mzis, clements_columns)

def test_two_mode_clements_always_has_first_top_left_is_one():
    m = 2

    phases = jax.random.uniform(jax.random.key(0), (3, 4, m)) * 2 * jnp.pi

    clements_params = make_clements_secondary_column_params(phases=phases)
    clements_columns = parse_clements_column(clements_params)

    top_left = clements_columns[...,0,0]

    assert jnp.allclose(top_left, 1)

def test_clements_columns_unitary():
    for m in [2, 3, 4, 8]:
        print(f'modes={m}')

        phases = jax.random.uniform(jax.random.key(0), (m)) * 2 * jnp.pi

        eye = jnp.eye(m)

        # primary case
        primary_params   = make_clements_primary_column_params  (phases=phases)
        primary_column = parse_clements_column(primary_params)
        primary_eye = primary_column @ primary_column.conj().T
        assert jnp.allclose(primary_eye, eye, atol=1e-7)

        #secondary case
        secondary_params = make_clements_secondary_column_params(phases=phases)
        secondary_column = parse_clements_column(secondary_params)
        secondary_eye = secondary_column @ secondary_column.conj().T
        assert jnp.allclose(secondary_eye, eye, atol=1e-7)


