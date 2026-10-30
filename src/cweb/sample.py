import jax
import jax.numpy as jnp

from jax.scipy.special import factorial

n = 4   # Max number of photons
m = 10  # Number of modes


@jax.jit
def calculate_indexing_matrix(fock : jnp.array):
    """
    Constructs an (...,n,m) indexing matrix from a (...,m) fock state tensor. This is used to convert an (m,m) matrix into
    an (...,n,m)  matrix with rows repeated per the photon count in the fock state.

    Args:
        fock (jnp.array): A tensor of fock states (...,m) where fock[...,i] is the number of photons in mode i.

    Returns:
        jnp.array: A binary tensor (...,n,m) to map (m,m) matrices to (...,n,m) tensor repeating each row per the number of photons.
    """
    start = jnp.cumsum(jnp.roll(fock, shift=1).at[...,0].set(0), axis=-1)
    end   = jnp.cumsum(         fock                           , axis=-1)

    ind = jnp.arange(n)

    after_start = ind[...,:,None] >= start[...,None,:]
    before_end  = ind[...,:,None]  < end  [...,None,:]

    # Convert to integer
    matrix = 1 * (after_start & before_end) 

    return matrix



@jax.jit
def calculate_amplitude(
    U        : jnp.array,
    fock_in  : jnp.array,
    fock_out : jnp.array):
    """
    Calculates the transition amplitude between two for a multi-mode fock states (...,n) for a given unitary tensor (m,m).

    Args:
        U (jnp.array) 
    """

    # Calculate facatorials
    f1 = jnp.sum(factorial(fock_in ), axis=-1)
    f2 = jnp.sum(factorial(fock_out), axis=-1)

    # Transform unitary repeating rows and columns per fock_in and fock_out
    S_in  = calculate_indexing_matrix(fock_in)
    S_out = calculate_indexing_matrix(fock_out)

    p = S_out @ U @ jnp.matrix_transpose(S_in) 

    return jnp.power(f1 * f2, -0.5) * calculate_permanent(p)
