from __future__ import annotations

from dataclasses import dataclass, field
from jax.tree_util import register_dataclass

import jax
import jax.numpy as jnp

from cweb.validation import validate_matrix, validate_vector
from cweb.model import ClementsReuploadingColumn, Clements

@register_dataclass
@dataclass
class ClementsReuploading:
    reuploading_columns : list[ClementsReuploadingColumn]
    data_dim           : int             = field(metadata=dict(static=True))
    diagonal_phases    : jnp.array

    def __post_init__(self):
        # Validate the diagonal phases
        validate_vector(self.diagonal_phases)
        m = self.get_m()

        # Validate modes match column modes
        for column in self.reuploading_columns:
            m_c = column.get_m()
            assert m == m_c, f'Attempt to make ClementsReuploading where column modes {m_c} does not match input modes {m}.'

    def get_m(self) -> int:
        return self.diagonal_phases.shape[-1]

    @jax.jit
    def get_clements(self, data : jnp.array) -> Clements:
        return Clements(
            diagonal_phases=self.diagonal_phases,
            columns=[c.get_clements_column(data) for c in self.reuploading_columns]
        )

