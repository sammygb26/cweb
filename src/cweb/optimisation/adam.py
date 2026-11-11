import jax
import jax.numpy as jnp

from functools import partial

def adam_get_init(model : dict) -> tuple[dict, dict, dict]:
    """
    From a model (pytree) returns a tuple of values holding the adam state.
    That is the model parameters and the moment estimations m1 and m2.

    Args:
        model (dict) : Any pytree reprisenting the model parameters.

    Returns:
        tuple[dict, dict, dict] : A tuple of pytrees which are identicle in structure.
    """
    m1 = m2 = jax.tree.map(lambda x: 0.0 * x, model)
    return model, m1, m2

def adam_extract_model_from_carry(carry : tuple[dict, dict, dict]) -> dict:
    """
    Extracts the model from the carry structure (same as init structure).

    Args:
        carry (tuple[dict, dict, dict]) : A tuple of three identicle pytrees.

    Retrns:
        dict : A pytree for the model parameters out of the carry value.
    """
    return carry[0]

@partial(jax.jit, static_argnames=['loss_fn'])
def adam_update(
        model : dict, 
        m1 : dict,
        m2 : dict,
        step : int,
        loss_fn : any,
        data : any,
        beta1 : float,
        beta2 : float,
        alpha : float,
        epsilon : float
    ) -> tuple[dict, dict, dict, float]:
    """
    Implements a update of the adam optimizer.

    Args:
        model (dict)    : A pytree of the model parameters.
        m1 (dict)       : A pytree of the smoothed first moment of the gradient.
        m2 (dict)       : A pytree of the smoothed second moment of the gradient.
        step (int)      : The current step from when adam started (starts are 0).
        loss_fn (any)   : The loss function used to calculate the gradient and loss.
        data (any)      : The data to be passed to the loss_fn.
        beta1 (float)   : The smoothing constant for the m1 estimate.
        beta2 (float)   : The smoothing constant for the m2 estimate.
        alpha (float)   : The stepsize parameter.
        epsilon (float) : The stability parameter.
    
    Returns:
        tuple[dict, dict, dict, float] : A tuple of the new model parameters
            updated m1 and m2 estimates and the current loss.
    """

    # Calculate loss and grad
    loss = loss_fn(model, data)
    g = jax.grad(loss_fn)(model, data)

    # Calculate updated step size
    b1t = jnp.power(beta1, step + 1)
    b2t = jnp.power(beta2, step + 1)

    alpha_t = alpha * jnp.sqrt(1 - b2t) / (1 - b1t)

    # Calculate loss and moments
    loss = loss_fn(model, data)
    g  = jax.grad(loss_fn)(model, data)
    g2 = jax.tree.map(jnp.square, g)

    # Calculated exponential smoothed moments
    m1 = jax.tree.map(lambda m1, g : beta1 * m1 + (1 - beta1) * g , m1, g )
    m2 = jax.tree.map(lambda m2, g2: beta2 * m2 + (1 - beta1) * g2, m2, g2)

    # Calculate denominator for update
    sqrt_m2_p_eps = jax.tree.map(lambda m2: jnp.sqrt(m2) + epsilon, m2)

    # Calculate updated model
    model = jax.tree.map(lambda model, m1, sqrt_m2_p_eps: model - (alpha_t * m1) / sqrt_m2_p_eps, model, m1, sqrt_m2_p_eps)

    return model, m1, m2, loss
