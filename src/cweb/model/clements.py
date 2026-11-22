from dataclasses import dataclass
from jax.tree_util import register_dataclass

import functools
import jax
import jax.numpy as jnp

from .clements_column import ClementsColumn

from cweb.validation import validate_tensor_of_vectors

@register_dataclass
@dataclass
class Clements:
    """
    A clements schemem made of a layer of diagonal phase shifters
    followed my clements columns (a stack of MZIs).

    Attributes:
        diagonal_phases (jnp.array)    : An array of initial phases shifter sof reach layer.
        columns (list[ClementsColumn]) : A list of layers of MZI stacks.
    """
    diagonal_phases : jnp.array
    columns : list[ClementsColumn]

    @classmethod
    def fully_trainable(cls, m : int) -> Clements:
        """
        Creates a fully trainable clements scheme - e.g a trainable unitary encoded as a clements scheme.

        Arguments:
            m (int) : The number of modes for the clements scheme.        
        """
        columns = [
            ClementsColumn(jnp.zeros(m), i % 2 == 0)
            for i in range(m)
        ]

        return Clements(columns)

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        # Check we have some columns at least.
        assert len(self.columns) != 0, 'Attempt to make clements scheme with no columns.'

        m = self.get_m()

        # Validates diagonal phases match mode count
        m_d = validate_tensor_of_vectors(self.diagonal_phases)
        assert m == m_d, f'Attempt to make clements scheme with diagonal mode count ({m_d}) not matching overall ({m}).'

        # Check all columns match mode count
        for column in self.columns:
            m_c = column.get_m()
            assert m_c == m, f'Attempt to make clements scheme with column mode count ({m_c}) not matching overall ({m}).'

    def get_m(self) -> int:
        """
        Gets the number of modes the clements scheme is for.

        Returns:
            int : The number of modes.
        """
        return self.diagonal_phases.shape[-1]

    @jax.jit
    def get_unitary(self) -> jnp.array:
        """
        Gets the unitary encoded in the clements scheme.

        Returns:
            jnp.array : A tensor of unitaries matching the clements columns dimensions.
        """
        column_unitaries = [c.get_unitary() for c in self.columns]
        diagonal = jnp.diag(jnp.exp(1j * self.diagonal_phases))

        return diagonal @ functools.reduce(jnp.matmul, column_unitaries)