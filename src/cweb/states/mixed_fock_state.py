from __future__ import annotations

import dataclasses
import jax

from collections.abc import Callable
from .fixed_photon_count_mixed_fock_state import FixedPhotonCountMixedFockState

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MixedFockState:
    """
    Representation of a mixed fock state where states have a variety of photon numbers.

    Attributes:
        states (dict[int, FixedPhotonCountMixedFockState]) : A dictionary of states and probabilities sorted by photon number.
    """
    states : dict[int, FixedPhotonCountMixedFockState]

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        for n, state in self.states.items():
            assert state.get_n() == n, 'Attempt to initialised MixedFockState with incorrect photon numbers.'
        
    def apply_photon_preserving_operation(
            self,
            fn : Callable[[FixedPhotonCountMixedFockState], FixedPhotonCountMixedFockState]
        ) -> MixedFockState:
        """
        Applies a photon preserving operations (FixedPhotonCountMixedFockState -> FixedPhotonCountMixedFockState) returning
        the resultant mixed state.

        Args:
            fn (Callable[[FixedPhotonCountMixedFockState], FixedPhotonCountMixedFockState]) : A function which transformes a fixed photon mixed fock state.

        Returns:
            MixedFockState : The state which results after the transformation.
        """

        @jax.jit
        def scan_fn(output_states, state):
            output_states[state.get_n()] = fn(state)
            return output_states, None

        output_states, _ = jax.lax.scan(scan_fn, {}, self.states.values())

        return MixedFockState(output_states)
