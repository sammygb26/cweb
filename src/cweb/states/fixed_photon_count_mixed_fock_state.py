import functools
import itertools
import dataclasses

import jax
import jax.numpy as jnp


from cweb.validation import validate_matrix
from cweb.validation import validate_tensor_of_vectors

FIXED_PHOTON_COUNT_MIXED_STATE = 'fixed_photon_count_mixed_state'
FOCK_STATES                    = 'fock_state'
PROBABILITIES                  = 'probabilites'

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FixedPhotonCountMixedFockState:
    fock_states   : jnp.array
    probabilities : jnp.array

    #def __init__(
    #        self,
    #        fock_states : jnp.array,
    #        probabilities : jnp.array):

    #    self.dict = {
    #        FIXED_PHOTON_COUNT_MIXED_STATE : [],
    #        FOCK_STATES                    : fock_states,
    #        PROBABILITIES                  : probabilities
    #    }

    #    state_count, n = validate_dict(self.dict)

    #    self.state_count = state_count
    #    self.n = n

    #@classmethod
    #def from_dict(cls, fixed_photon_count_mixed_state : dict):

    #    validate_dict(fixed_photon_count_mixed_state)

    #    return FixedPhotonCountMixedFockState(
    #        fock_states  =fixed_photon_count_mixed_state[FOCK_STATES  ],
    #        probabilities=fixed_photon_count_mixed_state[PROBABILITIES]
    #    )

    @classmethod
    def complete(cls, m : int, n : int) -> FixedPhotonCountMixedFockState:
        fock_states = get_complete_combinations(m, n)
        return FixedPhotonCountMixedFockState(fock_states, jnp.zeros((fock_states.shape[0])))

    #def copy(self) -> FixedPhotonCountMixedFockState:
    #    return FixedPhotonCountMixedFockState.from_dict(self.dict)

    def add_probabilities(
        self,
        probabilities : jnp.array
    ) -> FixedPhotonCountMixedFockState:
        state_count = validate_tensor_of_vectors(probabilities)
        assert self.get_state_count() == state_count

        return FixedPhotonCountMixedFockState(self.fock_states, self.probabilities + probabilities)

    def get_state_count(self):
        return self.probabilities.shape[-1]

    def get_n(self):
        return self.fock_states.shape[-1]


def validate_dict(
        fixed_photon_count_mixed_state : dict
    ) -> tuple[int, int]:

    assert FIXED_PHOTON_COUNT_MIXED_STATE in fixed_photon_count_mixed_state, 'Invalid fixed photon count mixed state does not contain main tag.'
    assert FOCK_STATES                    in fixed_photon_count_mixed_state, 'Invalid fixed photon count mixed state does not contain fock states.'
    assert PROBABILITIES                  in fixed_photon_count_mixed_state, 'Invalid fixed photon count mixed state does not contain probabilities.'

    fock_states   = fixed_photon_count_mixed_state[FOCK_STATES  ]
    probabilities = fixed_photon_count_mixed_state[PROBABILITIES]

    state_count, n = validate_matrix(fock_states)
    p_state_count  = validate_tensor_of_vectors(probabilities)

    assert p_state_count == state_count, f'Invalid state, probabilities (...{p_state_count}) do not match states count ({state_count},{n}).'

    return state_count, n


@functools.cache
def get_complete_combinations(m : int, n : int) -> jnp.array:
    combs = itertools.combinations_with_replacement(range(m), n)
    return jnp.array(list(combs))