from cweb import sample as s
from test_helpers.permanent import calculate_permanent_naive

import jax
import jax.numpy as jnp
import jax.scipy as jspy


def test_permanent_of_eye_is_1():
    assert jnp.isclose(s.calculate_permanent(jnp.eye(s.n)), 1.0)

def test_permanent_of_ones_is_n_facotiral():
    assert jnp.isclose(s.calculate_permanent(jnp.ones((s.n, s.n))), jspy.special.factorial(s.n))

def test_permanent_of_random_matrices():
    k1 = jax.random.key(0)
    k2 = jax.random.key(1)

    # Some random tensor dimensions
    a, b = 2, 3

    real = jax.random.uniform(k1, (a, b, s.n, s.n))
    imag = jax.random.uniform(k2, (a, b, s.n, s.n))

    rand = real + 1j * imag

    rysers = s.calculate_permanent(rand)
    naive  = calculate_permanent_naive(rand)

    print(
        f'rysers = {rysers}\n',
        f'naive  = {naive}\n'
    )

    assert jnp.allclose(rysers, naive)

