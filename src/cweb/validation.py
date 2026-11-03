import jax.numpy as jnp

def validate_tensor_of_square_matrixces(
    A : jnp.array
    ) -> int:
    """
    Validates a tensor of matrices (...,n,n) is a tensor of square matrices (matches shape template).
    
    Args:
        A (jnp.array) : Tensor of square matrices (...,n,n).
    
    Returns:
        (int) : n the dimension of the square matrices.
    """
    assert len(A.shape) >= 2
    n = A.shape[-1]
    assert A.shape[-2] == n

    return n
