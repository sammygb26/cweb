from cweb.model import parse_mach_zender_interferometer, make_mach_zender_interferometer_params

import jax
import jax.numpy as jnp

def test_mzi_with_external_phases_only_proportional_to_x_gate_and_phases_shifter():
    thetas = jnp.arange(0, 2 * jnp.pi, 0.1)
    phases = jnp.array([[theta, 0] for theta in thetas])
    params = make_mach_zender_interferometer_params(phases=phases)

    expected = jnp.array([1j * jnp.array([[0, 1], [jnp.exp(1j * theta), 0]]) for theta in thetas])
    actual   = parse_mach_zender_interferometer(params)

    print(expected)
    print(actual)

    assert jnp.allclose(expected, actual, atol=1e-8)

def test_mzi_is_unitary():
    b = 10

    phases = jax.random.uniform(jax.random.key(0), (b, 2)) * 2 * jnp.pi
    params = make_mach_zender_interferometer_params(phases=phases)

    expected = jnp.array([jnp.eye(2) for _ in range(b)])


    mzis = parse_mach_zender_interferometer(params=params)
    actual = mzis @ jnp.matrix_transpose(mzis.conj())

    print(expected)
    print(actual)

    assert jnp.allclose(expected, actual, atol=1e-7)
