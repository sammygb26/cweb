import jax
import jax.numpy as jnp

from jax.scipy.special import factorial

from cweb.permanent import calculate_permanent
from cweb.validation import validate_tensor_of_square_matrixces
from cweb.validation import validate_vector

from cweb.states import FixedPhotonCountMixedFockState

@jax.jit
def calculate_indexing_matrix(
    U : jnp.array, 
    fock : jnp.array
    ) -> jnp.array:
    """
    Constructs an (n,m) indexing matrix from a (n) fock state tensor. This is used to convert an (m,m) matrix into
    an (n,m) matrix with rows repeated per the photon count in the fock state.

    Args:
        fock (jnp.array): A tensor of fock states (n) where fock[i] is the mode photon i is in.

    Returns:
        jnp.array: A binary tensor (n,m) to map (m,m) matrices to (n,m) tensor repeating each row per the number of photons.
    """
    m = validate_tensor_of_square_matrixces(U)
    return 1 * (jnp.arange(m)[None,:] == fock[:,None])


@jax.jit
def calculate_amplitude(
    U        : jnp.array,
    fock_in  : jnp.array,
    fock_out : jnp.array
    ) -> jnp.array:
    """
    Calculates the transition amplitude between two for a multi-mode fock states (n) for a given unitary tensor (...,m,m).

    Args:
        U        (jnp.array) : A tensor of m mode unitary matrices to be validated (...,m,m)
        fock_in  (jnp.array) : A vector of photon mode indicies (n) for the input fock state.
        fock_out (jnp.array) : A vector of photon mode indicies (n) for the output fock state.

    Returns:
        (jnp.array) : A tensor of amplitudes for the input unitaries (...).
    """
    m = validate_tensor_of_square_matrixces(U)
    assert validate_vector(fock_in) == validate_vector(fock_out)

    # Calculate facatorials
    counts_in  = jnp.sum(jnp.arange(m)[None, :] == fock_in [:, None], axis=0)
    counts_out = jnp.sum(jnp.arange(m)[None, :] == fock_out[:, None], axis=0)

    f1 = jnp.prod(factorial(counts_in ))
    f2 = jnp.prod(factorial(counts_out))

    # Transform unitary repeating rows and columns per fock_in and fock_out
    S_in  = calculate_indexing_matrix(U, fock_in)
    S_out = calculate_indexing_matrix(U, fock_out)


    p = S_out @ U @ S_in.T
    a = jnp.power(f1 * f2, -0.5) * calculate_permanent(p)

    return a

@jax.jit
def calculate_probability(
    U        : jnp.array,
    fock_in  : jnp.array,
    fock_out : jnp.array
    ) -> jnp.array:
    """
    Calculates the probability of an output fock state being measured for a given input fock state for each unitary in a tensor of unitaries (...,m,m).

    Args:
        U        (jnp.array) : A tensor of m mode unitary matrices to be validated (...,m,m)
        fock_in  (jnp.array) : A vector of photon mode (in the range 0...m-1) indicies (n) for the input fock state.
        fock_out (jnp.array) : A vector of photon mode (in the range 9...m-1) indicies (n) for the output fock state.

    Returns:
        (jnp.array) : A tensor of probabilities for the input unitaries (...).
    """
    amplitude = calculate_amplitude(U, fock_in, fock_out)
    return jnp.real(amplitude * amplitude.conj())

@jax.jit
def calculate_probabilities(
        U         : jnp.array, 
        fock_in   : jnp.array,
        fock_outs : jnp.array,
        ) -> jnp.array:
    """
    Calculates the probabilities of all outputs maintaining photon number.
    """
    m = validate_tensor_of_square_matrixces(U)
    n = validate_vector(fock_in)

    @jax.jit
    def f(carry, fock_out):
        return carry, calculate_probability(U, fock_in, fock_out)

    _, probabilities = jax.lax.scan(f, (), fock_outs)
    probabilities = jnp.moveaxis(probabilities, 0, -1)

    return probabilities
    

# TODO: Test with multiindex U matching input probabilities.

@jax.jit
def apply_unitary_and_measure(
        U : jnp.array,
        input_state : FixedPhotonCountMixedFockState 
    ) -> FixedPhotonCountMixedFockState:

    # Validate the input
    m = validate_tensor_of_square_matrixces(U)

    n           = input_state.get_n()
    state_count = input_state.get_state_count()

    init = FixedPhotonCountMixedFockState.complete(m, n)

    @jax.jit
    def scan_fn(output_state : FixedPhotonCountMixedFockState, i):
        # Get the next fock state and its probability
        fock_in = input_state.fock_states  [i]
        p       = input_state.probabilities[i]

        # Calculate the probabilities for the output states
        ps = calculate_probabilities(U, fock_in, output_state.fock_states)

        # Add those probabilities scales by the probability of the input state
        output_state = output_state.add_probabilities(ps * p)

        return output_state, None
        
    final_state, _ = jax.lax.scan(scan_fn, init, jnp.arange(state_count))
    return final_state
