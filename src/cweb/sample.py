import jax
import jax.numpy as jnp

from jax.scipy.special import factorial

from cweb.permanent import calculate_permanent
from cweb.validation import validate_tensor_of_square_matrixces

n = 4   # Max number of photons
m = 10  # Number of modes


@jax.jit
def calculate_indexing_matrix(U : jnp.array, fock : jnp.array):
    """
    Constructs an (n,m) indexing matrix from a (n) fock state tensor. This is used to convert an (m,m) matrix into
    an (n,m) matrix with rows repeated per the photon count in the fock state.

    Args:
        fock (jnp.array): A tensor of fock states (n) where fock[i] is the mode photon i is in.

    Returns:
        jnp.array: A binary tensor (n,m) to map (m,m) matrices to (n,m) tensor repeating each row per the number of photons.
    """
    m = validate_tensor_of_square_matrixces(U)
    return 1 * (jnp.arange(m)[None,:] == fock[:,None])


@jax.jit
def calculate_amplitude(
    U        : jnp.array,
    fock_in  : jnp.array,
    fock_out : jnp.array):
    """
    Calculates the transition amplitude between two for a multi-mode fock states (n) for a given unitary tensor (...,m,m).

    Args:
        U (jnp.array) 
    """
    m = validate_tensor_of_square_matrixces(U)

    # Calculate facatorials
    counts_in  = jnp.sum(jnp.arange(m)[None, :] == fock_in[:, None], axis=0)
    counts_out = jnp.sum(jnp.arange(m)[None, :] == fock_in[:, None], axis=0)

    f1 = jnp.prod(factorial(counts_in ))
    f2 = jnp.prod(factorial(counts_out))

    # Transform unitary repeating rows and columns per fock_in and fock_out
    S_in  = calculate_indexing_matrix(U, fock_in)
    S_out = calculate_indexing_matrix(U, fock_out)


    p = S_out @ U @ S_in.T
    amplitude = jnp.power(f1 * f2, -0.5) * calculate_permanent(p)
    
    jax.debug.print(
        'U={U}\n' +
        'fock_in={fock_in}\n' + 
        'fock_out={fock_out}\n' +
        'counts_in={counts_in}\n' +
        'counts_out={counts_out}\n' +
        'f1={f1}\n' + 
        'f2={f2}\n' + 
        'p={p}\n',
        U=U,
        fock_in=fock_in,
        fock_out=fock_out,
        counts_in=counts_in,
        counts_out=counts_out,
        f1=f1,
        f2=f2,
        p=p)

    return amplitude
