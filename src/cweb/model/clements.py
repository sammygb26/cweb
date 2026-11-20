import functools

import jax
import jax.numpy as jnp

from .clements_column import parse_clements_column
from .clements_column import make_clements_primary_column_params
from .clements_column import make_clements_secondary_column_params

CLEMENTS        = 'clements'
DIAGONAL_PHASES = 'diagonal_phases'
COLUMNS         = 'columns'

def make_fully_trainable_clements_params(dim : int):
    columns = []
    zero_phases = jnp.zeros((dim))
    for i in range(dim):
        make_clements_column_params = [
            make_clements_primary_column_params,
            make_clements_secondary_column_params
        ][i % 2]

        columns.append(make_clements_column_params(phases=zero_phases))

    return make_clements_params(
        diagonal_phases=zero_phases,
        columns=columns
    )

@jax.jit
def make_clements_params(diagonal_phases : jnp.array, columns : list[dict]) -> dict:
    """
    Converts a list of column parameters for a clements scheme into clements params ready to be parsed.

    Note: This function does not verify the columns themselves reprisent a clements scheme or even that
        the layers are mode compatible.

    Args:
        columns (list) : A list of columns params for the clements scheme.

    Returns:
        dict : The parameters for the clements scheme ready to be parsed.
    """

    return {
        CLEMENTS        : [],
        DIAGONAL_PHASES : diagonal_phases,
        COLUMNS         : columns
    }

@jax.jit
def parse_clements(params : dict) -> jnp.array:
    """
    Parses clements parameters into a tensor of unitaries matching the dimension of the clements layers (...,m) or (m).

    Args:
        params (dict) : Parameters for the clements model containing columns with clements column (primary and secondary) parameters.

    Returns:
        jnp.array : A tensor of unitaries (...,m,m) which would be implemented by the clements scheme.
    """
    # Validate parameters
    assert CLEMENTS        in params, 'Attempt to parse non-clements parameters'
    assert DIAGONAL_PHASES in params, 'Attempt to parse clements parameters with no diagonal phases'
    assert COLUMNS         in params, 'Attempt to parse clements parameters no columns'
    assert len(params[COLUMNS]) >= 1, 'Attempt to parse clements parameters with empty columns'

    columns = params[COLUMNS]
    column_unitaries = [parse_clements_column(x) for x in columns]

    diagonal = jnp.diag(jnp.exp(1j * params[DIAGONAL_PHASES]))

    return diagonal @ functools.reduce(jnp.matmul, column_unitaries)
