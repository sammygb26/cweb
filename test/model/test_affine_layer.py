from cweb.model import AffineLayer

import jax
import jax.numpy as jnp

def test_affine_layer_can_be_eye():
    d = 7

    al_eye = AffineLayer(
        weights=jnp.eye(d),
        biases=jnp.zeros(d)
    )

    rand_v = jax.random.uniform(jax.random.key(0), (3, 9, d))

    rand_v_transformed = al_eye(rand_v)

    assert jnp.allclose(rand_v, rand_v_transformed)

def test_affine_layer_with_non_square_weights_gives_eepxected_output():
    d_in =  7
    d_out = 9

    al = AffineLayer(
        weights=jax.random.uniform(jax.random.key(0), (d_out, d_in)),
        biases=jax.random.uniform(jax.random.key(0), (d_out))
    )

    a, b, c = 3, 2, 6

    v_in = jax.random.uniform(jax.random.key(0), (a, b, c, d_in))
    v_out = al(v_in)

    assert v_out.shape == (a, b, c, d_out)

