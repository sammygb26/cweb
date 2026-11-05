from cweb.model import parse_mach_zender_interferometer, make_mach_zender_interferometer_params

import jax
import jax.numpy as jnp

def test_mzi_with_equal_phases_is_proportional_to_x_gate():
    thetas = jnp.arange(0, 2 * jnp.pi, 0.1)
    phases = jnp.array([[theta, theta] for theta in thetas])
    params = make_mach_zender_interferometer_params(phases=phases)

    x_gate = jnp.array([[0, 1], [1, 0]])

    expected = jnp.array([1j * x_gate * jnp.exp(1j * theta) for theta in thetas])
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
