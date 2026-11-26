from dataclasses import dataclass, field
from jax.tree_util import register_dataclass

import jax
import jax.numpy as jnp

from cweb.validation import validate_vector
from cweb.model import ClementsColumn

@register_dataclass
@dataclass
class ClementsReuploadingColumn:
    is_reuploading    : bool      = field(metadata=dict(static=True))
    data_dim          : int       = field(metadata=dict(static=True))
    phases_or_indices : jnp.array
    is_primary        : bool      = field(metadata=dict(static=True))

    def __post_init__(self):
        assert validate_vector(self.phases_or_indices) > 0, 'Attempto to make clements reuploading column with 0 modes.'

        # Validate we are using the correct dtype
        if self.is_reuploading:
            assert jnp.issubdtype(self.phases_or_indices, jnp.integer),  'Attempt to make clements reuploading column targetting data without integer indices.'
        else:
            assert jnp.issubdtype(self.phases_or_indices, jnp.floating), 'Attempt to make trainable clements reuploading column without floating phases.'


    def get_m(self) -> int:
        return self.phases_or_indices.shape[-1]

    @jax.jit
    def get_clements_column(self, data : jnp.array) -> ClementsColumn:
        assert data.shape[-1] == self.data_dim, 'Attempt to use reuploading column on data which it was not made for.'

        phases = data[...,self.phases_or_indices] if self.is_reuploading else self.phases_or_indices

        return ClementsColumn(
            phases=phases, 
            is_primary=self.is_primar
        )
