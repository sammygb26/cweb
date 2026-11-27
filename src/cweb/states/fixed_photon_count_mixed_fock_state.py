from __future__ import annotations

import functools
import itertools

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_vectors
from cweb.validation import validate_matrix

@jax.tree_util.register_dataclass
@dataclass
class FixedPhotonCountMixedFockState:
    """
    Represents a mixed fock state where the photon number is known.

    Attributes:
        fock_states (jax.Array)   : An matrix (s,n) where each row reprisents a mode index representation of some particular fock state.
        probabilities (jax.Array) : A tensor (...,s) of probabilities for each state.
    """
    fock_states   : jax.Array 
    probabilities : jax.Array

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        state_count, _ = validate_matrix(self.fock_states)
        assert state_count == validate_tensor_of_vectors(self.probabilities)

    @classmethod
    def complete(
            cls,
            m : int, 
            n : int,
            probabilities : jax.Array = None
        ) -> FixedPhotonCountMixedFockState:
        """
        Creates a complete mixed state meaning all possible fock states (for a given photon number) are represented.

        Args:
            m (int)                 : The number of modes.
            n (int)                 : The number of phtoons.
            fock_states (jax.Array) : The probabilities for the different states (0 if unset).

        Returns:
            FixedPhotonCountMixedFockState : The reprisentation of the mixed state with all probabilities set to 0.
        """
        fock_states = get_complete_combinations(m, n)

        # Fill with zeros if unset.
        if probabilities is None:
            probabilities = jnp.zeros((fock_states.shape[0]))

        return FixedPhotonCountMixedFockState(
            fock_states=fock_states,
            probabilities=probabilities
        )

    def add_probabilities(
        self,
        probabilities : jnp.array
    ) -> FixedPhotonCountMixedFockState:
        """
        Adds a tensor of probabilities to the states probabilities (...,s).

        Args:
            probabilities (jnp.array) : The probabilities (...,s) to be added to the state probabilities.

        Returns:
            FixedPhotonCountMixedFockState : The resultant mixed state with updated probabilities.
        """
        assert self.get_state_count() == validate_tensor_of_vectors(probabilities), 'Attempt to add incompatible probabilities to mixed state.'
        return FixedPhotonCountMixedFockState(self.fock_states, self.probabilities + probabilities)

    def get_state_count(self) -> int:
        """
        Gets the state count (s).

        Returns:
            int : The state count.
        """
        return self.probabilities.shape[-1]

    def get_n(self):
        """
        Gets the photon count (n).

        Returns:
            int : The photon count.
        """
        return self.fock_states.shape[-1]

@functools.cache
def get_complete_combinations(m : int, n : int) -> jnp.array:
    """
    Gets all indistribuishable (combinations) n photons in m modes.

    Returns:
        jnp.array : (N,n) matrix of combinations of indexes for n photons into m modes.
    """
    combs = itertools.combinations_with_replacement(range(m), n)
    return jnp.array(list(combs))