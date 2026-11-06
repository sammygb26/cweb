import jax.numpy as jnp
from itertools import permutations

def calculate_permanent_naive(A : jnp.array) -> jnp.array:
    """
    Calculates the permanent tensor (...) of a square matrix tensor (...,n,n) by naively interating over all
    permutations.
    
    Args:
        A (jnp.array): The tensor (...,n,n) of square matrices to calculate the permanents of.

    Returns:
        jnp.array: A tensor of permanents for the input square matrix tensor.
    """
    # Input validation & setup
    assert len(A.shape) >= 2
    n = A.shape[-1]
    assert A.shape[-2] == n

    # Calculation
    per = jnp.zeros_like(A[...,0,0])

    for permutation in permutations(range(n)):
        print(permutation)

        prod = jnp.ones_like(per)
        for i, j in enumerate(permutation):
            prod *= A[...,i,j]

        per += prod

    return per