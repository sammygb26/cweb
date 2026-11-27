from cweb.model import ReuploadingClements, Clements, ClementsColumn, ReuploadingClementsColumn
from cweb.states import FixedPhotonCountMixedFockState
from cweb.optimisation import adam_update, adam_get_init

import cweb.sample as sa
import cweb.helpers as h

import jax
import jax.numpy as jnp

def test_can_perform_optimisation_without_error():
    # Make some random data
    data_dim = 10
    data_size = 7

    data = jax.random.uniform(jax.random.key(0), (data_size, data_dim))

    # Initialise mode
    m = 4

    init_model = ReuploadingClements(
        diagonal_phases=h.random_phases(jax.random.key(0), m),
        columns=[
            ReuploadingClementsColumn(data_dim, jnp.array([0, 7, 9, 9]), is_primary=True),
            ClementsColumn(h.random_phases(jax.random.key(1), m),        is_primary=False),
            ReuploadingClementsColumn(data_dim, jnp.array([1, 2, 3, 4]), is_primary=True),
            ClementsColumn(h.random_phases(jax.random.key(2), m),        is_primary=False),
        ],
        data_dim=data_dim
    )

    # Setup some photonic input state
    n = 4

    input_state = FixedPhotonCountMixedFockState.complete(
        m=m, 
        n=n, 
        probabilities=h.random_probabilities(jax.random.key(3), h.complete_count(m, n))
    )

    # Define some loss function
    @jax.jit
    def loss_fn(model : ReuploadingClements, data : jax.Array) -> jax.Array:
        cle : Clements = model.get_clements(data)
        uni = cle.get_unitary()

        output_state = sa.apply_unitary_and_measure(uni, input_state)
        ps = output_state.probabilities

        ps_flipped_signs = ps * jnp.power(-1, jnp.arange(ps.shape[-1]))

        return jnp.mean(jnp.abs(jnp.sum(ps_flipped_signs, axis=-1)))

    # Do some optimisation
    @jax.jit
    def update(carry, step : int):
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
            alpha=1.0,
            epsilon=1e-8)

        return (model, m1, m2), loss

    epochs = int(10)
    init = adam_get_init(init_model)
    carry, _ = jax.lax.scan(update, init, jnp.arange(epochs))

    final_model, _, _ = carry

    assert type(final_model) is ReuploadingClements


        
