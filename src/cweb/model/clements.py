import functools

import jax
import jax.numpy as jnp

from .clements_column import parse_clements_column

CLEMENTS = 'clements'
COLUMNS  = 'columns'

@jax.jit
def make_clements_params(columns : list[dict]) -> dict:
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
        CLEMENTS : [],
        COLUMNS  : columns
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
    assert CLEMENTS in params,        'Attempt to parse non-clements parameters'
    assert COLUMNS  in params,        'Attempt to parse clements parameters missing columns'
    assert len(params[COLUMNS]) >= 1, 'Attempt to parse clements parameters with empty columns'

    columns = params[COLUMNS]
    column_unitaries = [parse_clements_column(x) for x in columns]

    return functools.reduce(jnp.matmul, column_unitaries)
