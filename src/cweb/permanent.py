import jax
import jax.numpy as jnp

from cweb.validation import validate_tensor_of_square_matrixces

@jax.jit
def calculate_gray_vector(
    A : jnp.array, 
    k : int
    ) -> jnp.array:
    """
    Calculates a binary vector of the kth element of the gray code.

    Args:
        k (int) : The index intop the gray code.

    Returns:
        (jnp.array) : A vector (n) of the n binary digits of the kth gray code.
    """
    n = validate_tensor_of_square_matrixces(A)

    # Create binary vectors of k
    bin_k    = jnp.apply_along_axis(lambda x: (k >>  x     ) & 1, axis=-1, arr=jnp.arange(n))
    bin_k_ls = jnp.apply_along_axis(lambda x: (k >> (x + 1)) & 1, axis=-1, arr=jnp.arange(n))

    gray = bin_k ^ bin_k_ls

    return gray

@jax.jit
def calculate_gray_update(
    A : jnp.array, 
    k : int
    ) -> jnp.array:
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
    gray      = calculate_gray_vector(A, k    )
    gray_prev = calculate_gray_vector(A, k - 1)

    gray_update = gray - gray_prev
    gray_beta_update = A @ gray_update

    return gray_beta_update


@jax.jit
def calculate_permanent(
    A : jnp.array
    ) -> jnp.array:
    """
    Calculates the permanent of a tensor of square matrices (...,n,n) using ryser's formula.

    Args:
        A (jnp.array): A tensor of square matrixces (...,n,n) for which the permanent tensor will be calculated.

    Returns:
        jnp.array: A tensor of permanents (...)
    """
    n = validate_tensor_of_square_matrixces(A)

    # Calculation
    sign = (-1) ** (n + 1)
    beta = calculate_gray_update(A, 1)
    perm = sign * jnp.prod(beta, axis=-1)

    carry = (sign, beta, perm)

    def for_func(k, carry):
        sign, beta, perm = carry

        sign *= -1
        beta = (beta + calculate_gray_update(A, k))
        perm += sign * jnp.prod(beta, axis=-1)

        return (sign, beta, perm)

    _, _, perm = jax.lax.fori_loop(2, 2**n, for_func, carry)

    return perm