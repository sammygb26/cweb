import jax
import jax.numpy as jnp

from functools import partial

def adam_get_init(model : dict) -> tuple[dict, dict, dict]:
    m1 = m2 = jax.tree.map(lambda x: 0.0 * x, model)
    return model, m1, m2

def adam_extract_model_from_carry(carry : tuple[dict, dict, dict]) -> dict:
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
        epsilon : float):
    """
    
    STEP IS FROM 0!
    """

    # Calculate loss and grad
    loss = loss_fn(model, data)
    g = jax.grad(loss_fn)(model, data)

    # Calculate updated step size
    b1t = jnp.power(beta1, step + 1)
    b2t = jnp.power(beta2, step + 1)

    alpha_t = jnp.sqrt(1 - b2t) / (1 - b1t)

    # Calculate loss and moments
    loss = loss_fn(model, data)
    g  = jax.grad(loss_fn)(model, data)
    g2 = jax.tree.map(jnp.square, g)

    # Calculated exponential smoothed moments
    m1 = jax.tree.map(lambda m1, g : beta1 * m1 + (1 - beta1) * g , m1, g )
    m2 = jax.tree.map(lambda m2, g2: beta2 * m2 + (1 - beta1) * g2, m2, g2)


    # Calculate denominator for update
    sqrt_m2_p_eps = jax.tree.map(lambda m2: jnp.sqrt(m2) + epsilon, m2)

    #jax.debug.log(
    #    'm1={m1}\n' +
    #    'm2={m2}\n' +
    #    'alpha_t={alpha_t}\n' +
    #    'sqrt_m2_p_eps={sqrt_m2_p_eps}\n' +
    #    '',
    #    m1=m1,
    #    m2=m2,
    #    alpha_t=alpha_t,
    #    sqrt_m2_p_eps=sqrt_m2_p_eps)

    # Calculate updated model
    model = jax.tree.map(lambda model, m1, sqrt_m2_p_eps: model - (alpha_t * m1) / sqrt_m2_p_eps, model, m1, sqrt_m2_p_eps)

    return model, m1, m2, loss
