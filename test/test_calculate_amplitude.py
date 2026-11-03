
import jax
import jax.numpy as jnp

from cweb.sample import calculate_amplitude
from cweb.permanent import calculate_permanent

def test_calculated_amplitudes_for_eye_are_eye():
    m = 10

    unitary = jnp.eye(m)

    for i in range(m):
        for j in range(m):
            fock_in  = jnp.array([i])
            fock_out = jnp.array([j])

            amplitude = calculate_amplitude(unitary, fock_in, fock_out)

            assert jnp.allclose(amplitude, 1.0 if i == j else 0.0)

def test_calculated_amplitudes_for_random_matrix_are_matrix_elements():
    m = 2

    real = jax.random.uniform(jax.random.key(0), (3, 4, m, m))
    imag = jax.random.uniform(jax.random.key(1), (3, 4, m, m))

    mat = real + 1j * imag

    for i in range(m):
        for j in range(m):
            fock_in  = jnp.array([i])
            fock_out = jnp.array([j])

            amplitude = calculate_amplitude(mat, fock_in, fock_out)

            assert jnp.allclose(amplitude, mat[...,j, i])

def test_calculated_amplitudes_for_uniform_in_and_out_are_matrix_permanent():
    m = 3

    real = jax.random.uniform(jax.random.key(0), (3, 4, m, m))
    imag = jax.random.uniform(jax.random.key(1), (3, 4, m, m))

    mat = real + 1j * imag
    fock = jnp.arange(m)

    perm = calculate_permanent(mat)
    amplitude = calculate_amplitude(mat, fock, fock)

    assert jnp.allclose(perm, amplitude)