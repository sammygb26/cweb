from cweb.optimisation import *

import jax
import jax.numpy as jnp

from functools import partial

def test_batch_gradient_descent_on_toy_problem():
    @jax.jit
    def loss_fn(x : jnp.array, data : jnp.array):
        """data (c,a,b) the loss is the polynomial ax^2+bx+c."""
        return data @ jnp.power(x, jnp.array(range(3))).T

    data = jax.random.uniform(jax.random.key(0), (3))
    model = jax.random.uniform(jax.random.key(1), (1)) 

    epochs = 100

    @jax.jit
    def update(carry, step):
        return batch_gradient_descent_update(carry, step, loss_fn, data, 1)

    model, (_, loss) = jax.lax.scan(update, model, jnp.arange(epochs)) 

    # There should be an exact solution...
    c, b, a = data
    x = model

    assert jnp.allclose(loss[-1],  c - (b**2)/(4*a))
    assert jnp.allclose(x, -b/(2*a))

test_batch_gradient_descent_on_toy_problem()