from cweb.states import MixedFockState, FixedPhotonCountMixedFockState

import jax.numpy as jnp

def test_can_create_mixed_fock_state_manually():
    n_and_state_counts = [(5, 6), (4, 3), (2, 7)]

    MixedFockState({n : 
        FixedPhotonCountMixedFockState(
                jnp.zeros((state_count, n)),
                jnp.zeros((state_count))
            ) for n, state_count in n_and_state_counts
        })