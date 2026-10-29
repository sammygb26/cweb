import jax.numpy as jnp
import jax
from jax import partial

@jax.jit
def mach_zender_interferometer(theta1, theta2):
    d12 = (theta1 - theta2) / 2
    m12 = (theta1 + theta2) / 2

    em12 = jnp.exp(m12)
    sd12 = jnp.sin(d12)
    cd12 = jnp.cos(d12)

    return (1j / 2) * em12 * jnp.array([
        [sd12,  cd12],
        [cd12, -sd12]
    ])