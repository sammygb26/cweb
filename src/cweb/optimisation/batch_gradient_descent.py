import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnames=['loss_fn'])
def batch_gradient_descent_update(
        model : dict, 
        loss_fn : any, 
        data : any, 
        eps : float
    ) -> tuple[dict, any]:
    """
    Implementation of the batch gradient descent step where 
    loss and loss_grad are calculated and used to upate model
    parameters.

    Args:
        model (dict)  : Model parameters which will be updated (expected as pytree).
        loss_fn (any) : Loss function evaluated with (model, data).
        data (any)    : Data to be passed to loss_fn for this step.
        eps (float)   : The step size to be used.

    Returns:
        tuple[dict, any] : The update dmoel and loss returned by the loss_fn.
    """
    # Calculate loss and grad
    loss = loss_fn(model, data)
    loss_grad = jax.grad(loss_fn)(model, data)

    new_model = jax.tree.map(lambda x, dx: x - eps * dx, model, loss_grad)
    return new_model, loss

