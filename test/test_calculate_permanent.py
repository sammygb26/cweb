from cweb import permanent as p
from test_helpers.permanent import calculate_permanent_naive

import jax
import jax.numpy as jnp
import jax.scipy as jspy

def test_permanent_of_eye_is_1():
    assert jnp.isclose(p.calculate_permanent(jnp.eye( 4)), 1.0)
    assert jnp.isclose(p.calculate_permanent(jnp.eye( 6)), 1.0)

def test_permanent_of_ones_is_n_facotiral():
    assert jnp.isclose(p.calculate_permanent(jnp.ones(( 4,  4))), jspy.special.factorial(4))
    assert jnp.isclose(p.calculate_permanent(jnp.ones(( 7,  7))), jspy.special.factorial(7))

def test_permanent_of_random_matrices():
    k1 = jax.random.key(0)
    k2 = jax.random.key(1)

    # Some random tensor dimensions
    a, b = 2, 3

    real = jax.random.uniform(k1, (a, b, 3, 3))
    imag = jax.random.uniform(k2, (a, b, 3, 3))

    rand = real + 1j * imag

    rysers = p.calculate_permanent(rand)
    naive  = calculate_permanent_naive(rand)

    print(
        f'rysers = {rysers}\n',
        f'naive  = {naive}\n'
    )

    assert jnp.allclose(rysers, naive)

