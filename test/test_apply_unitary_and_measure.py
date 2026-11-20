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