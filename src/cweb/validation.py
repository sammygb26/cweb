import jax.numpy as jnp

def validate_tensor_of_square_matrixces(
    A : jnp.array
    ) -> int:
    """
    Validates a tensor of matrices (...,n,n) is a tensor of square matrices (matches shape template).
    
    Args:
        A (jnp.array) : Tensor of square matrices (...,n,n).
    
    Returns:
        int : n the dimension of the square matrices.
    """
    assert len(A.shape) >= 2
    n = A.shape[-1]
    assert A.shape[-2] == n

    return n

def validate_tensor_of_vectors(
        A: jnp.array
    ) -> int:
    """
    Validates a tensor of vectors (...,n) is a tensor of vectors (matches shape template).

    Args:
        A (jnp.array) : tensor of vectors (...,n).

    Returns:
        int : n the dimension of the vectors.
    """
    assert len(A.shape) >= 1
    return A.shape[-1]

def validate_vector(v : jnp.array) -> int:
    """
    Validates a vector (n) is a vector (matches shape template).

    Args:
        v (jnp.array) : A vector (n).

    Returns:
        int : n the dimension of the vector.
    """

    assert len(v.shape) == 1
    return v.shape[0]

def validate_matrix(A : jnp.array) -> tuple[int, int]:
    assert len(A.shape) == 2
    return A.shape[0], A.shape[1]
