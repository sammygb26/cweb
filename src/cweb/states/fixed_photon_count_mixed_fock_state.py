import functools
import itertools
import dataclasses

import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_vectors

FIXED_PHOTON_COUNT_MIXED_STATE = 'fixed_photon_count_mixed_state'
FOCK_STATES                    = 'fock_state'
PROBABILITIES                  = 'probabilites'

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FixedPhotonCountMixedFockState:
    fock_states   : jnp.array
    probabilities : jnp.array

    @classmethod
    def complete(cls, m : int, n : int) -> FixedPhotonCountMixedFockState:
        fock_states = get_complete_combinations(m, n)
        return FixedPhotonCountMixedFockState(fock_states, jnp.zeros((fock_states.shape[0])))

    def add_probabilities(
        self,
        probabilities : jnp.array
    ) -> FixedPhotonCountMixedFockState:
        assert self.get_state_count() == validate_tensor_of_vectors(probabilities), 'Attempt to add incompatible probabilities to mixed state.'
        return FixedPhotonCountMixedFockState(self.fock_states, self.probabilities + probabilities)

    def get_state_count(self):
        return self.probabilities.shape[-1]

    def get_n(self):
        return self.fock_states.shape[-1]

@functools.cache
def get_complete_combinations(m : int, n : int) -> jnp.array:
    combs = itertools.combinations_with_replacement(range(m), n)
    return jnp.array(list(combs))