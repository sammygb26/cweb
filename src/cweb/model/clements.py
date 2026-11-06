import functools

import jax
import jax.numpy as jnp

from .clements_column import parse_clements_column

CLEMENTS = 'clements'
COLUMNS  = 'columns'

@jax.jit
def make_clements_params(columns : list[dict]) -> dict:
    return {
        CLEMENTS : [],
        COLUMNS  : columns
    }

@jax.jit
def parse_clements(params : dict) -> jnp:
    # Validate parameters
    assert CLEMENTS in params,        'Attempt to parse non-clements parameters'
    assert COLUMNS  in params,        'Attempt to parse clements parameters missing columns'
    assert len(params[COLUMNS]) >= 1, 'Attempt to parse clements parameters with empty columns'

    columns = params[COLUMNS]
    column_unitaries = [parse_clements_column(x) for x in columns]

    return functools.reduce(jnp.matmul, column_unitaries)
