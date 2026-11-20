import cweb.optimisation as o
import cweb.model        as m
import cweb.utils        as u
import cweb.helpers      as h

import jax
import jax.numpy as jnp

import pandas as pd

import jax_tqdm
import os
import pathlib

def execute_run(l : u.Logger, dl : u.DataLogger, dim : int):

    l.log(f'\nStarting dim {dim} run...')

    key = jax.random.key(dim)
    unitary = jax.random.orthogonal(key, dim, dtype=jnp.complex64)
    epochs = int(50_000)

    # Generate and randomize clements scheme
    clements_params = m.make_fully_trainable_clements_params(dim=dim)
    clements_params = h.randomize_tree_phases(
        key=key, 
        tree=clements_params
    )

    # Print a summary of the trainable model to check
    print(
        'Clements params\n',
        jax.tree.map(lambda x: x.shape, clements_params)
    )

    # Save the parameters before training
    dl.save('initial_params.pkl', clements_params)
    dl.save('run_params.pkl', {'dim' : dim, 'epochs' : epochs})

    # Definition of the objective
    @jax.jit
    def loss_fn(clements_params, unitary):
        clements_unitary = m.parse_clements(clements_params)
        return jnp.mean(jnp.abs(clements_unitary - unitary))

    # Definition of how adam will be used
    @jax_tqdm.scan_tqdm(epochs)
    def update(carry, step):
        model, m1, m2 = carry

        model, m1, m2, loss = o.adam_update(
            model=model,
            m1=m1,
            m2=m2,
            step=step,
            loss_fn=loss_fn,
            data=unitary,
            beta1=0.9,
            beta2=0.999,
            alpha=1e-2,
            epsilon=1e-8)

        jax.debug.log('model={model}\nm1={m1}\nm2={m2}', model=model, m1=m1, m2=m2)

        return (model, m1, m2), loss

    # Execute adam for {epochs}
    init = o.adam_get_init(clements_params)
    (clements_params, _, _), loss = jax.lax.scan(update, init, jnp.arange(epochs))

    # Log the output
    l.log(f'Completed dim {dim} run. Final loss {loss[-1]}')
    l.log(
        f'Final abs difference\n' +
        f'{jnp.abs(unitary - m.parse_clements(clements_params))}'
    )

    # Save the loss and final model for later
    dl.save('final_params.pkl', clements_params)
    dl.save_df('loss.csv', pd.DataFrame({'loss' : loss}))


def main():
    # Setup output folder
    output_folder = str(
        pathlib.Path.home()
        /'work'
        /'cweb'
        /'clements_reprsents_unitary'
    )

    # Initialise logger for run
    l = u.Logger(output_folder)
    l.log('Running "clements represents unitary"...')

    # Iterate throughj dimensions and execute run
    for dim in [2, 3, 5, 8, 12]:
        dl = u.DataLogger(os.path.join(output_folder, f'{dim}'))
        execute_run(l, dl, dim)

    l.log('Finished!')

if __name__ == '__main__':
    main()