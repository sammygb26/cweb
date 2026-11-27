from dataclasses import dataclass, field
from jax.tree_util import register_dataclass

import jax
import jax.numpy as jnp

from cweb.validation import validate_vector

from .clements_column import ClementsColumn

@register_dataclass
@dataclass
class ReuploadingClementsColumn:
    """
    A reprisentation of a reuploading column.

    Attributes:
        data_dim (int)      : The expected dimension of the input data.
        indices (jax.Array) : The indices into the input data used for reuploading.
        is_primary (bool)   : Whether this columns is primary column (top MZI interats with top mode).
    """
    data_dim   : int       = field(metadata=dict(static=True))
    indices    : jnp.array 
    is_primary : bool      = field(metadata=dict(static=True))

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """

        self.indices = self.indices.astype(jnp.int32)

        assert validate_vector(self.indices) > 0,     'Attempt to make ReuploadingClementsColumn with 0 modes.'

    def get_m(self) -> int:
        """
        Gets the number of modes modeled by the column.

        Returns:
            int : The number of modes represented by the scheme.
        """

        return self.indices.shape[-1]

    @jax.jit
    def get_clements_column(self, data : jax.Array) -> ClementsColumn:
        """
        Gets the clements clements colum representation using the provided data as the phases.

        Args:
            data (jax.Array) : The data to be used for the phases.
    
        Returns:
            ClementsColumn : The concrete clements column made by using the proivded data as the phases.
        """

        assert data.shape[-1] == self.data_dim, 'Attempt to use reuploading column on data which it was not made for.'

        # Reupload using indices
        phases = data[...,jax.lax.stop_gradient(self.indices)]

        return ClementsColumn(
            phases=phases, 
            is_primary=self.is_primary
        )
