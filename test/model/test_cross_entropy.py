from cweb.model import cross_entropy
from cweb.helpers import random_probabilities

import jax
import jax.numpy as jnp

def test_cross_entropy_is_never_larger_than_entropy():
    a, b, c = 7, 4, 9

    p = random_probabilities(jax.random.key(0), (a, b, c))
    q = random_probabilities(jax.random.key(1), (a, b, c))

    h_p = cross_entropy(p, p)
    h_pq = cross_entropy(p, q)

    assert jnp.all( h_p > 0)
    assert jnp.all(h_pq >= h_p)