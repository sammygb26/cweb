import dataclasses
import jax

from collections.abc import Callable
from .fixed_photon_count_mixed_fock_state import FixedPhotonCountMixedFockState

MIXED_FOCK_STATE = 'mixed_fock_state'
STATES = 'states'

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MixedFockState:
    states : dict[int, FixedPhotonCountMixedFockState]

    def apply_photon_preserving_operation(
        self,
        fn : Callable[[FixedPhotonCountMixedFockState], FixedPhotonCountMixedFockState]
        ) -> MixedFockState:

        @jax.jit
        def scan_fn(output_states, state):
            output_states[state.get_n()] = fn(state)
            return output_states, None

        output_states, _ = jax.lax.scan(scan_fn, {}, self.states.values())

        return MixedFockState(output_states)
