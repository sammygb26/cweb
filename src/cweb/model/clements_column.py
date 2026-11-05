from .mach_zender_interferometer import mach_zender_interferometer
from ..validation import validate_tensor_of_vectors

import jax
import jax.numpy as jnp

CLEMENTS_PRIMARY_COLUMN   = 'clements_primary_column'
CLEMENTS_SECONDARY_COLUMN = 'clements_secondary_column'
PHASES                    = 'phases'

@jax.jit
def make_clements_primary_column_params(phases : jnp.array) -> dict:
    """
    Parses the phases for a clements primary column (always MZI passthrough on modes 1 and 2).

    Args:
        phases (jnp.array) : Tensor of phases (...,m) including passthrough modes (phases is ignored).

    Returns:
        dict : Dictionary of parameters matching the clements primary column type.
    """
    validate_tensor_of_vectors(phases)
    return {
        CLEMENTS_PRIMARY_COLUMN : [],
        PHASES : phases
    }

@jax.jit
def make_clements_secondary_column_params(phases : jnp.array) -> dict:
    """
    Parses the phases for a clements secondary column (always has a passthrough on mode 1).

    Args:
        phases (jnp.array) : Tensor of phases (...,m) including passthrough modes (phases is ignored).

    Returns:
        dict : Dictionary of parameters matching the clements secondary column type.
    """
    validate_tensor_of_vectors(phases)
    return {
        CLEMENTS_SECONDARY_COLUMN : [],
        PHASES : phases
    }

@jax.jit
def parse_clements_column(params : dict) -> jnp.array:
    """
    Parses params into a clements column (either primary or secondary). The params contain the phases
    (...,m) for an m mode circuit. Some phases are ignored depending on the total number of modes and 
    whether m is odd or even.

    Args:
        params (dict) : A dictionary of parameters containing the phases (...m).

    Returns:
        A tensor of unitaries matching the phases (...,m,m).    
    """
    # Validation
    assert (CLEMENTS_PRIMARY_COLUMN   in params or 
            CLEMENTS_SECONDARY_COLUMN in params)
    assert PHASES                     in params

    phases = params[PHASES]
    m = validate_tensor_of_vectors(A=phases)

    # Get offset based on whether this is a primary or secondary column
    offset = 0 if CLEMENTS_PRIMARY_COLUMN in params else 1

    # Setup unitary initialy as 0s
    unitary = jnp.zeros(list(phases.shape) + [m], dtype=jnp.complex64)

    # Setup passthrouhg for first and last mode
    unitary = unitary.at[...,-1,-1].set(1)
    unitary = unitary.at[...,0,0].set(1)

    # Iterate through MZIs in column adding to unitary
    for k in range((m  - offset) // 2):
        i = 2 * k + offset
        j = i + 2

        mzi = mach_zender_interferometer(phases[...,i:j])

        unitary = unitary.at[...,i:j,i:j].set(mzi)

    return unitary
