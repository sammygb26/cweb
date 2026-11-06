import jax

from functools import partial

@partial(jax.jit, static_argnames=['loss_fn'])
def batch_gradient_descent_update(carry, step, loss_fn, data, eps):
    model = carry

    # Calculate loss and grad
    loss = loss_fn(model, data)
    loss_grad = jax.grad(loss_fn)(model, data)

    new_model = jax.tree.map(lambda x, dx: x - eps * dx, model, loss_grad)
    return new_model, (step, loss)

