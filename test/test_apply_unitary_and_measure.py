import jax
import jax.numpy as jnp

from cweb.states import FixedPhotonCountMixedFockState
from cweb.sample import apply_unitary_and_measure

def test_apply_unitary_and_measure_on_eye_maintains_probabilities():
    m = 2
    n = 3

    state_in = FixedPhotonCountMixedFockState.complete(m, n)

    ps = jax.random.uniform(jax.random.key(0), state_in.get_state_count())
    state_in = state_in.add_probabilities(ps / sum(ps))

    eye = jnp.eye(m)

    state_out = apply_unitary_and_measure(eye, state_in) 

    assert jnp.allclose(state_in.probabilities, state_out.probabilities)

def test_apply_unitary_and_measure_maintains_probabilities_after_multiple_rounds():
    m = 2
    n = 3

    # Randomly generate state with random probabilities
    state = FixedPhotonCountMixedFockState.complete(m, n)
    ps = jax.random.uniform(jax.random.key(0), state.get_state_count())
    state = state.add_probabilities(ps / sum(ps))

    # Randomly generate unitary
    unitary = jax.random.orthogonal(jax.random.key(0), n=m, dtype=jnp.complex64)

    # Before and after  many rounds of sampling probabilities should be maintained
    assert jnp.allclose(jnp.sum(state.probabilities), 1.0)

    for _ in range(8):
        state = apply_unitary_and_measure(unitary, state) 

    assert jnp.allclose(jnp.sum(state.probabilities), 1.0)

def test_apply_unitary_tensor_and_measure_maintains_probabilities_after_multiple_rounds():
    m = 2
    n = 3

    # Randomly generate state with random probabilities
    state = FixedPhotonCountMixedFockState.complete(m, n)
    ps = jax.random.uniform(jax.random.key(0), state.get_state_count())
    state = state.add_probabilities(ps / sum(ps))

    # Randomly generate unitary
    unitary = jnp.array([
        jax.random.orthogonal(jax.random.key(i), n=m, dtype=jnp.complex64)
        for i in range(4)
    ])

    # Before and after  many rounds of sampling probabilities should be maintained
    print('starting state', state)
    assert jnp.allclose(jnp.sum(state.probabilities), 1.0)

    for _ in range(8):
        state = apply_unitary_and_measure(unitary, state) 

    print('final state', state)
    print(jnp.sum(state.probabilities, axis=-1))

    assert jnp.allclose(jnp.sum(state.probabilities, axis=-1), 1.0)