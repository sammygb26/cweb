from .mach_zender_interferometer import MachZenderInterferometer
from ..validation import validate_tensor_of_vectors

from dataclasses import dataclass, field
from jax.tree_util import register_dataclass

import jax.numpy as jnp

@register_dataclass
@dataclass
class ClementsColumn:
    """
    A clements column reprisented my an phase-shifter phases for MZIs assigned to pairs of modes.

    Attributes:
        phases     (jnp.array) : A tensor of phases for each mode of the scheme.
        is_primary (bool)      : Whether this is a primary column (first MZI covers mode 0).
    """
    phases : jnp.array
    is_primary : int = field(metadata=dict(static=True))

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        validate_tensor_of_vectors(self.phases)

    def __str__(self) -> str:
        """
        Gets the string reprisentation of the clements column.

        Returns:
            str : The string representation of the clements column.
        """
        return f'ClementsColumn({self.phases}, {"primary" if self.is_primary else "secondary"})'

    def get_m(self):
        """
        Gets the numbner of modes this column is defined for.
        """
        return self.phases.shape[-1]

    def get_unitary(self):
        """
        Gets a tensor of unitaries encoded by the phases of this column.

        Returns:
            jnp.array : A tensor of unitaries reprisenting the linear transformation of a single photon entering the column.
        """
        # Get offset based on whether this is a primary or secondary column
        offset = 0 if self.is_primary else 1
        m = self.get_m()

        # Setup unitary initialy as 0s
        unitary = jnp.zeros(list(self.phases.shape) + [m], dtype=jnp.complex64)

        # Setup passthrouhg for first and last mode
        unitary = unitary.at[...,-1,-1].set(1)
        unitary = unitary.at[...,0,0].set(1)

        # Iterate through MZIs in column adding to unitary
        for k in range((m  - offset) // 2):
            i = 2 * k + offset
            j = i + 2

            mzi = MachZenderInterferometer(self.phases[...,i:j])

            unitary = unitary.at[...,i:j,i:j].set(mzi.get_unitary())

        return unitary