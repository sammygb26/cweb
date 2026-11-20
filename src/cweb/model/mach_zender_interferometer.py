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
    Converts two phase values into a unitary reprisenting the an MZI interferometer for P(t1)B(0.5)P(t2)B(0.5)
    where B is a beam splitter and P is a one-mode phase shifter.

    Args:
        phases (jnp.array) : The phases (...,2), t1 = [...,0], t2 = [...,1]

    Returns:
        jnp.array : A tensor of unitaries (...,2,2) matching the linear unitary of the MZI.
    """
    assert phases.shape[-1] == 2

    theta1 = phases[...,0]
    theta2 = phases[...,1]

    t2o2 = 0.5 * theta2

    et1   = jnp.exp(1j * theta1)
    et2o2 = jnp.exp(1j * t2o2)
    st2o2 = jnp.sin(t2o2)
    ct2o2 = jnp.cos(t2o2)

    unitary = 1j * et2o2 * jnp.array([
        [st2o2 * et1,  ct2o2 * et1],
        [ct2o2      , -st2o2      ]
    ])

    return jnp.moveaxis(unitary, (0,1), (-1, -2))