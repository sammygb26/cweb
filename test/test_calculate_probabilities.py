
import jax
import jax.numpy as jnp

from cweb.sample import calculate_probabilities
from cweb.states import FixedPhotonCountMixedFockState

def test_calculate_probabilities_on_random_unitary_tensor_returns_output_of_correct_shape():
    unitaries = jnp.array(
        [jax.random.orthogonal(jax.random.key(i), 7) for i in range(10)], 
        dtype=jnp.complex64
    )

    fock_in = jnp.zeros(3)
    fock_outs = jnp.zeros((8, 3))

    probabilities = calculate_probabilities(unitaries, fock_in, fock_outs)

    observed_shape = probabilities.shape
    expected_shape = (10, 8)

    print(f'observed_shape={observed_shape}')
    print(f'expected_shape={expected_shape}')

    assert probabilities.shape == (10, 8)

def test_calculate_probabilities_on_complete_state_summs_to_1():
    m = 8
    n = 7

    unitary = jax.random.orthogonal(jax.random.key(0), m)
    fock_in = jnp.arange(n)

    complete_out = FixedPhotonCountMixedFockState.complete(m, n)

    probabilities = calculate_probabilities(unitary, fock_in, complete_out.fock_states)

    assert jnp.allclose(sum(probabilities), 1.0)



    