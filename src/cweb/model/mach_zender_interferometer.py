import jax
import jax.numpy as jnp

MACH_ZENDER_INTERFEROMETER = 'mach_zender_interferometer'
PHASES                     = 'phases'

def make_mach_zender_interferometer_params(phases : jnp.array) -> dict:
    """
    Makes the parameter dictionary for a basic 2-mode MZI.

    Args:
        phases (jnp.array) : A tensor (...,2) of phases for the MZI.

    Returns:
        dict : Dictionary which can be parsed into an MZI unitary.
    """
    assert phases.shape[-1] == 2

    return {
        MACH_ZENDER_INTERFEROMETER : [],
        PHASES : phases
    }

def parse_mach_zender_interferometer(params : dict) -> jnp.array:
    """
    Parses a set of parameters with type mach_zender_interferometer and phases (...,2) into a tensor of unitaries (...,2).

    Args:
        params (dict) : A parameters dictionary of type mach_zender_interferometer with phases (...,2)

    Returns:
        jnp.array : A tensor of unitaries (...,2)
    """

    assert MACH_ZENDER_INTERFEROMETER in params
    assert PHASES in params

    phases : jnp.array = params[PHASES] 

    return mach_zender_interferometer(phases)


@jax.jit
def mach_zender_interferometer(
    phases : jnp.array
    ) -> jnp.array:
    """
    Converts two phase values into a unitary reprisenting the an MZI interferometer for B(0.5)P(t1,t2)B(0.5)
    where B is a beam splitter and P is a two-mode phase shifter.

    Args:
        theta1 (jnp.array) :

    Returns:
        jnp.array
    """
    assert phases.shape[-1] == 2

    theta1 = phases[...,0]
    theta2 = phases[...,1]

    d12 = (theta1 - theta2) / 2
    m12 = (theta1 + theta2) / 2

    em12 = jnp.exp(1j * m12)
    sd12 = jnp.sin(d12)
    cd12 = jnp.cos(d12)

    unitary = 1j * em12 * jnp.array([
        [sd12,  cd12],
        [cd12, -sd12]
    ])

    return jnp.moveaxis(unitary, (0,1), (-1, -2))