
from dataclasses import dataclass
from jax.tree_util import register_dataclass

import jax
import jax.numpy as jnp

from cweb.validation import validate_matrix, validate_vector

@register_dataclass
@dataclass
class AffineLayer:
    """
    An implementation of a basic NN affine layer u = Wx + b.

    Attributes:
        weights (jax.Array) : The weight term W (d_out, d_in).
        biases (jax.Array)  : The bias term b (d_out).
    """

    weights : jax.Array
    biases  : jax.Array

    def __post_init__(self):
        """
        Post initialisation validating the arugments passed to the constructor are valid.
        """
        d_out, _ = validate_matrix(self.weights)
        assert d_out == validate_vector(self.biases), 'Attempt to make AffineLayer where bias dimension does not match weight output.'

    def __call__(self, vector : jax.Array) -> jax.Array:
        """
        Passes a vector through the AffineLayer.

        Args:
            vector (jax.Array) : A tensor of vectors (...,d_in) to be transformed.

        Returns:
            jax.Array : The transformed tensor of vectors (...,d_out).
        """
        return jnp.matmul(vector, self.weights.T) + self.biases
