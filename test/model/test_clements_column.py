from cweb.model import *
from cweb.helpers import random_phases

import jax
import jax.numpy as jnp

def test_two_mode_clements_primary_is_equal_to_mzi():
    m = 2

    # Random phases
    phases = random_phases(jax.random.key(0), (3, 4, m))

    # MZI and primary clements column made with same phases
    mzi = MachZenderInterferometer(phases=phases)
    cc  =           ClementsColumn(phases=phases, is_primary=True)

    # Are equal
    assert jnp.allclose(mzi.get_unitary(), cc.get_unitary())

def test_clements_secondary_always_has_first_top_left_as_one():
    for m in [2, 3, 4, 5]:
        # Random phases
        phases = random_phases(jax.random.key(0), (3, 2, m))

        cc = ClementsColumn(phases=phases, is_primary=False)
        top_left = cc.get_unitary()[...,0,0]

        assert jnp.allclose(top_left, 1.0)

def test_clements_columns_unitary():
    for m in [2, 3, 4, 8]:
        print(f'modes={m}')

        phases = random_phases(jax.random.key(0), (m))
        eye = jnp.eye(m)

        # primary case
        pc = ClementsColumn(phases=phases, is_primary=True)
        pu = pc.get_unitary()
        p_eye = pu @ pu.conj().T
        assert jnp.allclose(p_eye, eye, atol=1e-7)

        #secondary case
        sc = ClementsColumn(phases=phases, is_primary=False)
        su = sc.get_unitary()
        s_eye = su @ su.conj().T
        assert jnp.allclose(s_eye, eye, atol=1e-7)


