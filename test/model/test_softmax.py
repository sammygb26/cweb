from cweb.model import softmax

import jax
import jax.numpy as jnp

def test_softmax_is_normalised_correctly():
    a, b, c = 7, 5, 2
    input = jax.random.uniform(jax.random.key(0), (a, b, c)) * 2.0 - 1.0
    output = softmax(input)

    total_p = jnp.sum(output, axis=-1)

    assert output.shape == (7, 5, 2)
    assert jnp.allclose(total_p, 1.0)

def test_softmax_for_constant_is_evn():
    a, b, c = 3, 4, 7

    input = jax.random.uniform(jax.random.key(0), (a, b)) * 2.0 - 1.0
    input = input[...,None] * jnp.ones((a, b, c))

    output = softmax(input)

    assert jnp.allclose(output, 1.0 / c)
