from cweb.data import get_random_batch

import jax
import jax.numpy as jnp

def test_batches_are_of_expected_size():
    data = jax.random.uniform(jax.random.key(0), (10, 4, 3))

    batch_size = 13

    data_batch = get_random_batch(
        key=jax.random.key(1),
        step=0,
        batch_size=batch_size,
        data=data)

    assert data_batch.shape == (batch_size, 4, 3)

def test_batches_of_different_steps_are_different():
    data = jax.random.uniform(jax.random.key(0), (10, 4, 3))

    batch_size = 13

    b1 = get_random_batch(
        key=jax.random.key(1),
        step=0,
        batch_size=batch_size,
        data=data)

    b2 = get_random_batch(
        key=jax.random.key(1),
        step=1,
        batch_size=batch_size,
        data=data)

    # At least some of the elements should be different
    assert b1.shape == b2.shape
    assert jnp.any(b1 != b2)

