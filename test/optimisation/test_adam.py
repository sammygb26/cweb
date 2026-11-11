from cweb.optimisation import *

import jax
import jax.numpy as jnp

from functools import partial

def test_adam_on_toy_problem():
    @jax.jit
    def loss_fn(x : jnp.array, data : jnp.array):
        """data (c,b,a) the loss is the polynomial ax^2+bx+c."""
        return data @ jnp.power(x, jnp.array(range(3))).T

    data = jax.random.uniform(jax.random.key(0), (3))
    model = jax.random.uniform(jax.random.key(1), (1)) 
    m1 = m2 = jnp.zeros_like(model)

    epochs = 1_000

    @jax.jit
    def update(carry, step):
        model, m1, m2 = carry

        model, m1, m2, loss = adam_update(
            model=model, 
            m1=m1, 
            m2=m2, 
            step=step, 
            loss_fn=loss_fn,
            data=data,
            beta1=0.9, 
            beta2=0.999,
            alpha=1e-3,
            epsilon=1e-8)

        return (model, m1, m2), loss

    (model, _, _), loss = jax.lax.scan(update, (model, m1, m2), jnp.arange(epochs)) 

    # There should be an exact solution...
    c, b, a = data
    x = model

    print(c, b, a, x)

    assert jnp.allclose(loss[-1],  c - (b**2)/(4*a), atol=1e-4)
    assert jnp.allclose(x       , -b/(2*a)         , atol=1e-4)

def test_adam_on_toy_pytree_problem():
    @jax.jit
    def loss_fn(model : dict, data: dict):
        x1 = model['x1']
        x2 = model['x2']

        a1 = data['a1']
        a2 = data['a2']

        x12 = x1 * x1
        x22 = x2 * x2

        return jnp.sum(a1 * x12 + a2 * x22)

    data = {
        'a1' : jnp.array([0.5]),
        'a2' : jnp.array([2.0])
    }

    model = {
        'x1' : jnp.array([0.2]),
        'x2' : jnp.array([1.5])
    }

    epochs = 1_000

    @jax.jit
    def update(carry, step):
        model, m1, m2 = carry

        model, m1, m2, loss = adam_update(
            model=model, 
            m1=m1, 
            m2=m2, 
            step=step, 
            loss_fn=loss_fn,
            data=data,
            beta1=0.9, 
            beta2=0.999,
            alpha=1e-3,
            epsilon=1e-8)

        return (model, m1, m2), loss

    init = adam_get_init(model)
    carry, loss = jax.lax.scan(update, init, jnp.arange(epochs)) 
    model = adam_extract_model_from_carry(carry)

    assert jnp.allclose(jnp.array(jax.tree.leaves(model)), 0.0, atol=1e-4)
    assert jnp.allclose(loss[-1]                         , 0.0, atol=1e-4)

test_adam_on_toy_pytree_problem()
