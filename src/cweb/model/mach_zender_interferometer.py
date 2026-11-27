import dataclasses
import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_vectors

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MachZenderInterferometer:
    """
    A parameterisation of a basic 2-mode MZI. 

    Attributes:
        phases (jnp.array) : A tensor (...,2) of phases for the MZI.
    """
    phases : jnp.array

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        assert validate_tensor_of_vectors(self.phases) == 2

    @jax.jit
    def get_unitary(self):
        """
        Converts the phases into a unitary reprisenting the an MZI interferometer for P(t1)B(0.5)P(t2)B(0.5)
        where B is a beam splitter and P is a one-mode phase shifter.

        Returns:
            jnp.array : A tensor of unitaries (...,2,2) matching the linear optics unitary of the MZI.
        """
        theta1 = self.phases[...,0]
        theta2 = self.phases[...,1]

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
