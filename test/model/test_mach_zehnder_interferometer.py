from cweb.model import MachZenderInterferometer
from cweb.helpers import random_phases

import jax
import jax.numpy as jnp

def test_mzi_with_external_phases_only_proportional_to_x_gate_and_phases_shifter():
    thetas = jnp.arange(0, 2 * jnp.pi, 0.1)
    phases = jnp.array([[theta, 0] for theta in thetas])
    mzi = MachZenderInterferometer(phases)

    expected = jnp.array([1j * jnp.array([[0, 1], [jnp.exp(1j * theta), 0]]) for theta in thetas])
    actual   = mzi.get_unitary()

    print(expected)
    print(actual)

    assert jnp.allclose(expected, actual, atol=1e-8)

def test_mzi_is_unitary():
    b = 10

    mzi = MachZenderInterferometer(random_phases(jax.random.key(0), (b,2)))

    expected = jnp.array([jnp.eye(2) for _ in range(b)])


    mzi_unitaries = mzi.get_unitary()
    actual = mzi_unitaries @ jnp.matrix_transpose(mzi_unitaries.conj())

    print(expected)
    print(actual)

    assert jnp.allclose(expected, actual, atol=1e-7)
