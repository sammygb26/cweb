import jax
import jax.numpy as jnp

from functools import partial

from collections.abc import Callable

from cweb.validation import validate_vector
from cweb.validation import validate_matrix

from .fixed_photon_count_mixed_fock_state import *

MIXED_FOCK_STATE = 'mixed_fock_state'
STATES = 'states'

class MixedFockState:
    def __init__(self, fixed_photon_count_mixed_states : list[FixedPhotonCountMixedFockState]):
        self.dict = {
            MIXED_FOCK_STATE : [],
            STATES : {}
        }

        for fixed_photon_count_mixed_state in fixed_photon_count_mixed_states:
            n = fixed_photon_count_mixed_state.n
            assert n not in self.dict[STATES]
            self.dict[STATES][n] = fixed_photon_count_mixed_state

    @classmethod
    def from_dict(cls, mixed_fock_state : MixedFockState): 
        return MixedFockState(mixed_fock_state[STATES])


@jax.jit
def validate_dict(mixed_fock_state_dict : dict) -> list[int]:
    assert MIXED_FOCK_STATE in mixed_fock_state_dict
    assert STATES           in mixed_fock_state_dict
    return list(mixed_fock_state_dict[STATES].keys())

@partial(jax.jit, static_argnames=['fn'])
def apply_photon_preserving_operation_to_mixed_fock_state(
        fn : Callable[[FixedPhotonCountMixedFockState], FixedPhotonCountMixedFockState],
        mixed_fock_state : dict
    ) -> dict:

    photon_numbers = validate_dict(mixed_fock_state)

    @jax.jit
    def scan_f(carry, n):
        mixed_fock_state = carry
        mixed_fock_state[n] = fn(mixed_fock_state[n])
        return carry, None

    mixed_fock_state, = jax.lax.scan(mixed_fock_state, photon_numbers)

    

    