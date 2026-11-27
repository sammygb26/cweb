from __future__ import annotations

from dataclasses import dataclass, field
from jax.tree_util import register_dataclass

import jax
import jax.numpy as jnp

from cweb.validation import validate_vector

from .reuploading_clements_column import ReuploadingClementsColumn
from .clements import Clements
from .clements_column import ClementsColumn

@register_dataclass
@dataclass
class ReuploadingClements:
    """
    Representation of a reuploading clements scheme.

    Attributes:
        diagonal_phases (jax.Array) : The phases for the initial diagonal phases going into the clements scheme.
        columns (list[ReuploadingClementsColumn|ClementsColumn]) : The columns of the scheme (split between reuploading and pure).
        data_dim : The expected data dimension.
    """
    diagonal_phases : jax.Array
    columns         : list[ReuploadingClementsColumn|ClementsColumn]
    data_dim        : int = field(metadata=dict(static=True))

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        # Validate the diagonal phases
        validate_vector(self.diagonal_phases)
        m = self.get_m()

        # Make sure there are some columns
        assert len(self.columns) > 0, 'Attempt to make ReuploadingClements with no columns.'

        # Validate modes match pure column modes
        pure_columns = [c for c in self.columns if type(c) is ClementsColumn]
        for c in pure_columns:
            m_c = c.get_m()
            assert m == m_c, f'Attempt to make ReuploadingClements where column modes {m_c} does not match input modes {m}.'

        # Validate modes match reupload column modes
        reupload_columns = [c for c in self.columns if type(c) is ReuploadingClementsColumn]
        for c in reupload_columns:
            m_c = c.get_m()
            assert c.data_dim == self.data_dim, f'Attempt to make ReuploadingClements with inconsistent data dim (column: {c.data_dim}, overall:{self.data_dim}).'
            assert m == m_c, f'Attempt to make ReuploadingClements where column modes {m_c} does not match input modes {m}.'

        # Make sure all layers are pure or reuploading
        assert len(pure_columns) + len(reupload_columns) == len(self.columns), 'Attempt to make ReuploadingClements with columns which are not pure or reuploading'


    def get_m(self) -> int:
        """
        Gets the number of modes represented by the clements scheme.

        Returns:
            int : The number of modes reprisented by the scheme.
        """

        return self.diagonal_phases.shape[-1]

    @jax.jit
    def get_clements(self, data : jax.Array) -> Clements:
        """
        Combines the reuploading clements scheme with live data to create a clements description.

        Args:
            data (jax.Array) : The data to be used for the reuploading layers.

        Returns:
            Clements : The clements schemem made by combining the phases and live data.
        """

        def get_column(i) -> ClementsColumn:
            column = self.columns[i]
            if type(column) is ReuploadingClementsColumn:
                return column.get_clements_column(data)
            else:
                return column

        return Clements(
            diagonal_phases=self.diagonal_phases,
            columns=[get_column(i) for i in range(len(self.columns))]
        )

