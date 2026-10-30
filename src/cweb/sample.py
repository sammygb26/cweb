import jax
import jax.numpy as jnp

from jax.scipy.special import factorial

n = 4   # Max number of photons
m = 10  # Number of modes


@jax.jit
def calculate_indexing_matrix(fock : jnp.array):
    """
    Constructs an (...,n,m) indexing matrix from a (...,n) fock state tensor. This is used to convert an (m,m) matrix into
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
def calculate_gray_vector(k):
    """
    Calculates a binary vector of the kth element of the gray code.

    Args:
        k (int) : The index intop the gray code.

    Returns:
        (jnp.array) : A vector (n) of the n binary digits of the kth gray code.
    """

    # Create binary vectors of k
    bin_k    = jnp.apply_along_axis(lambda x: (k >>  x     ) & 1, axis=-1, arr=jnp.arange(n))
    bin_k_ls = jnp.apply_along_axis(lambda x: (k >> (x + 1)) & 1, axis=-1, arr=jnp.arange(n))

    gray = bin_k ^ bin_k_ls

    return gray

@jax.jit
def calculate_gray_update(A : jnp.array, k : int) -> jnp.array:
    """
    Calculates a gray update tensor for a given index (int) and input tensor of square matrices (...,n,n).

    The gray code defines and order for subsets of {1...n} where neighbours only differ by the inclusion of
    one element. Within Ryser's formula products of row-sums for each subset are calculated. The row-sums are
    refered to as beta_k in the code. Naively calculating all of these for some k takes O(n^2) time but using the gray
    code neighbouring beta_k only differ by the inclusion of some row so the update instead takles O(n).

    This function calculates the tensor of updates between two neighbouring beta_k values.

    Args:
        A (jnp.array) : The tensor of square matrices (...,n,n) for which we require a gray update calculating.
        k (int)       : The index into the gray code for the update.

    Returns:
        (jnp.array) : A tensor of gray code updates for the row-sums (...,n).
    """
    gray      = calculate_gray_vector(k)
    gray_prev = calculate_gray_vector(k - 1)

    gray_update = gray - gray_prev
    gray_beta_update = A @ gray_update

    return gray_beta_update


@jax.jit
def calculate_permanent(A : jnp.array) -> jnp.array:
    """
    Calculates the permanent of a tensor of square matrices (...,n,n) using ryser's formula.

    Args:
        A (jnp.array): A tensor of square matrixces (...,n,n) for which the permanent tensor will be calculated.

    Returns:
        jnp.array: A tensor of permanents (...)
    """
    jax.debug.print("A={A}", A=A)

    sign_k = (-1) ** (n + 1)
    beta_k = calculate_gray_update(A, 1)
    perm_k = sign_k * jnp.prod(beta_k, axis=-1)

    for k in range(2, 2**n):
        sign_k *= -1
        beta_k = (beta_k + calculate_gray_update(A, k))
        perm_k += sign_k * jnp.prod(beta_k, axis=-1)

    return perm_k


@jax.jit
def calculate_amplitude(
    U        : jnp.array,
    fock_in  : jnp.array,
    fock_out : jnp.array):

    # Calculate facatorials
    f1 = jnp.sum(factorial(fock_in ), axis=-1)
    f2 = jnp.sum(factorial(fock_out), axis=-1)

    # Transform unitary repeating rows and columns per fock_in and fock_out
    S_in  = calculate_indexing_matrix(fock_in)
    S_out = calculate_indexing_matrix(fock_out)

    p = S_out @ U @ jnp.matrix_transpose(S_in) 

    return jnp.power(f1 * f2, -0.5) * calculate_permanent(p)
