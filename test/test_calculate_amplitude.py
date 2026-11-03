
import jax.numpy as jnp

from cweb.sample import calculate_amplitude

def test_calculated_amplitudes_for_eye_are_eye():
    m = 10

    unitary = jnp.eye(m)

    for i in range(m):
        for j in range(m):
            fock_in  = jnp.array([i])
            fock_out = jnp.array([j])

            amplitude = calculate_amplitude(unitary, fock_in, fock_out)

            print(f'<{j}|I|{i}> = {amplitude}') 

            assert jnp.allclose(amplitude, 1.0 if i == j else 0.0)

test_calculated_amplitudes_for_eye_are_eye()