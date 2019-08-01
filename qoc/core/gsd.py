"""
gsd.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

import os

import autograd.numpy as anp
from autograd.extend import Box
import numpy as np
import scipy.linalg as la

from qoc.models import (MagnusPolicy, OperationPolicy, GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerPolicy, GrapeResult,
                        InterpolationPolicy, Reporter)
from qoc.standard import (Adam, ForbidStates, TargetInfidelity, expm)
from qoc.util import (PAULI_X, PAULI_Y, conjugate_transpose,
                      matrix_to_column_vector_list, matmuls, ans_jacobian,
                      complex_to_real_imag_vec,
                      real_imag_to_complex_vec)


### MAIN METHODS ###

def grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                iteration_count, param_count, pulse_step_count,
                                pulse_time,
                                grape_schroedinger_policy=GrapeSchroedingerPolicy.TIME_EFFICIENT,
                                initial_params=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                log_iteration_step=100,
                                magnus_policy=MagnusPolicy.M2,
                                max_param_amplitudes=None,
                                operation_policy=OperationPolicy.CPU,
                                optimizer=Adam(),
                                save_file_name=None, save_iteration_step=0,
                                save_path=None, system_step_multiplier=1,):
    """
    a method to optimize the evolution of a set of states under the
    schroedinger equation for time-discrete control parameters
    Args:
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    grape_schroedinger_policy :: qoc.GrapeSchroedingerPolicy - how to perform
        the main integration of GRAPE, can be optimized for time or memory
    hamiltonian :: (params :: numpy.ndarray, time :: float)
                    -> hamiltonian :: numpy.ndarray
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the control parameters
        and evolution time
    initial_params :: numpy.ndarray - the values to use for the
        parameters for the first iteration,
        This array should have shape (pulse_step_count, parameter_count)
    initial_states :: numpy.ndarray - a list of the states (column vectors)
        to evolve
        A column vector is specified as np.array([[0], [1], [2]]).
        A column vector is NOT a row vector np.array([0, 1, 2]).
    interpolation_policy :: qoc.InterpolationPolicy - how to interpolate
        optimization parameters where they are not defined
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: int - how often to write to stdout,
        set 0 to disable logging
    magnus_policy :: qoc.MagnusPolicy - the method to use for the magnus
        expansion
    max_param_amplitudes :: numpy.ndarray - These are the absolute values at
        which to clip the parameters if they achieve +max_amplitude
        or -max_amplitude. This array should have shape
        (parameter_count). The default maximum amplitudes will
        be 1 if not specified. 
    operation_policy :: qoc.OperationPolicy - how computations should be performed,
        e.g. CPU, GPU, CPU-sparse, GPU-spares, etc.
    optimizer :: qoc.models.Optimizer - an instance of an optimizer to perform
        gradient-based optimization
    param_count :: int - the number of control parameters required to be
        passed to the hamiltonian at each time step
    pulse_step_count :: int - the number of time steps at which the pulse
        should be updated (optimized)
    pulse_time :: float - the duration of the control pulse, also the
        evolution time
    save_file_name :: str - this will identify the save file
    save_iteration_step :: int - how often to write to the save file,
        set 0 to disable saving
    save_path :: str - the directory to create the save file in,
        the directory will be created if it does not exist
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps
    Returns:
    result :: qoc.GrapeResult - useful information about the optimization
    """
    # Initialize parameters.
    initial_params, max_param_amplitudes = _initialize_params(initial_params,
                                                              max_param_amplitudes,
                                                              pulse_time,
                                                              pulse_step_count,
                                                              param_count)
    
    # Initialize optimizer.
    optimizer.initialize(complex_to_real_imag_vec(initial_params).shape)
    
    # Construct the grape state.
    hilbert_size = initial_states[0].shape[0]
    gstate = GrapeSchroedingerDiscreteState(costs, grape_schroedinger_policy,
                                            hamiltonian, hilbert_size,
                                            interpolation_policy, iteration_count,
                                            log_iteration_step, magnus_policy,
                                            max_param_amplitudes, operation_policy,
                                            optimizer, param_count, pulse_step_count,
                                            pulse_time, save_file_name, save_iteration_step,
                                            save_path, system_step_multiplier)
    
    # Perform initial log and save.
    if gstate.log_iteration_step != 0:
        gstate.log_initial()
    if gstate.save_iteration_step != 0:
        gstate.save_initial(initial_params, initial_states)

    # Switch on GRAPE implementation method.
    if gstate.grape_schroedinger_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        error, grads, params, states = _grape_schroedinger_discrete_time(gstate,
                                                                         initial_params,
                                                                         initial_states,)
    else:
        error, grads, params, states = _grape_schroedinger_discrete_memory(gstate,
                                                                           initial_params,
                                                                           initial_states,)

    return GrapeResult(error, grads, params, states)


### HELPER METHODS ###

def _grape_schroedinger_discrete_time(gstate, params, states):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    params :: numpy.ndarray - the initial params
    states :: numpy.ndarray - the initial states
    Returns:
    error :: numpy.ndarray - the total cost error at the final time step of
        the last iteration,
    grads :: numpy.ndarray - the gradients at the final time step of the last iteration,
                             same shape as params
    params :: numpy.ndarray - the parameters at the final time step of the last iteration
    states :: numpy.ndarray - the states at the final time step of the last iteration
    Returns: none
    """
    # Track mutable objects using a reporter.
    # Autograd does not allow multiple return values from
    # a differentiable function.
    reporter = Reporter()

    # Determine optimization constants.
    final_iteration = gstate.iteration_count - 1
            
    # Run optimization for the given number of iterations.
    for iteration in range(gstate.iteration_count):
        # Compute the gradient of the costs with respect
        # to params.
        total_error, grads = _gsd_compute_ans_jac(gstate, params,
                                                  reporter, states)
        # Remove states from autograd box.
        if isinstance(reporter.states, Box):
            reporter.states = reporter.states._value

        # Log and save progress.
        if (gstate.log_iteration_step != 0
              and np.mod(iteration, gstate.log_iteration_step) == 0):
            gstate.log(total_error, grads, iteration, params, reporter.states)
            
        if (gstate.save_iteration_step != 0
              and np.mod(iteration, gstate.save_iteration_step) == 0):
            gstate.save(total_error, grads, iteration, params, reporter.states)

        # Update params.
        if iteration != final_iteration:
            params = real_imag_to_complex_vec(
                gstate.optimizer.update(complex_to_real_imag_vec(grads),
                                        complex_to_real_imag_vec(params)))
            _clip_params(gstate.max_param_amplitudes, params)
    #ENDFOR
    return total_error, grads, params, reporter.states


def _grape_schroedinger_discrete_compute(gstate, params, reporter,
                                         states):
    """
    Compute the costs for one evolution cycle.
    Args:
    dt :: float - the time step between evolution steps
    gstate :: qoc.GrapeSchroedingerDiscreteState - static objects
    params :: numpy.ndarray - the control parameters
    reporter :: qoc.Reporter - a reporter for mutable objects
    states :: numpy.ndarray - the initial states to evolve
    Returns:
    total_error :: numpy.ndarray - total error of the evolution
    """
    total_error = 0
    for time_step in range(gstate.final_time_step + 1):
        pulse_step, _ = anp.divmod(time_step, gstate.system_step_multiplier)
        is_final_step = time_step == gstate.final_time_step
        t = time_step * gstate.dt
        
        # Evolve.
        # Get the parameters to use for magnus expansion interpolation.
        magnus_param_indices = gstate.magnus_param_indices(gstate.dt, params,
                                                           pulse_step, t,
                                                           is_final_step)
        magnus_params = params[magnus_param_indices,]
        # If magnus_params includes only one parameter array,
        # wrap it in another dimension.
        if magnus_params.ndim == 1:
            magnus_params = anp.expand_dims(magnus_params, axis=0)
        magnus = gstate.magnus(gstate.dt, magnus_params, pulse_step, t, is_final_step)
        unitary = expm(-1j * magnus)
        states = anp.matmul(unitary, states)
        
        # Compute cost.
        if is_final_step:
            for i, cost in enumerate(gstate.costs):
                error = cost.cost(params, states, time_step)
                total_error = total_error + error
            #ENDFOR
            reporter.states = states
        else:
            for i, step_cost in enumerate(gstate.step_costs):
                error = step_cost.cost(params, states, time_step)
                total_error = total_error + error
    #ENDFOR
    return total_error


# Real wrapper for gsd_compute.
_gsd_compute_real = (lambda *args, **kwargs:
                     anp.real(_grape_schroedinger_discrete_compute(*args,
                                                                   **kwargs)))


# Value and jacobian of gsd_compute.
_gsd_compute_ans_jac = ans_jacobian(_gsd_compute_real, 1)


def _grape_schroedinger_discrete_memory(gstate, params, states):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use the memory efficient method.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    params :: numpy.ndarray - the initial params
    states :: numpy.ndarray - the initial states
    Returns:
    error :: numpy.ndarray - the errors at the final time step of the last iteration,
                             same shape as the cost function list
    grads :: numpy.ndarray - the gradients at the final time step of the last iteration,
                             same shape as params
    params :: numpy.ndarray - the parameters at the final time step of the last iteration
    states :: numpy.ndarray - the states at the final time step of the last iteration
    """
    pass


def _initialize_params(initial_params, max_param_amplitudes,
                       pulse_time,
                       pulse_step_count, param_count):
    """
    Sanitize the initial_params and max_param_amplitudes.
    Generate both if either was not specified.
    Args:
    initial_params :: numpy.ndarray - the user specified initial parameters
    max_param_amplitudes :: numpy.ndarray - the user specified max
        param amplitudes
    pulse_time :: float - the duration of the pulse
    pulse_step_count :: int - number of pulse steps
    param_count :: int - number of parameters per pulse step

    Returns:
    params :: numpy.ndarray - the initial parameters
    max_param_amplitudes :: numpy.ndarray - the maximum parameter
        amplitudes
    """
    if max_param_amplitudes is None:
        max_param_amplitudes = np.ones(param_count)
        
    if initial_params is None:
        params = _gen_params_cos(pulse_time, pulse_step_count, param_count,
                                 max_param_amplitudes)
    else:
        # If the user specified initial params, check that they conform to
        # max param amplitudes.
        for i, step_params in enumerate(initial_params):
            if not np.less_equal(step_params, max_param_amplitudes).all():
                raise ValueError("Expected that initial_params specified by "
                                 "user conformed to max_param_amplitudes, but "
                                 "found conflict at step {} with {} and {}"
                                 "".format(i, step_params, max_param_amplitudes))
        #ENDFOR
        params = initial_params

    return params, max_param_amplitudes
            

def _gen_params_cos(pulse_time, pulse_step_count, param_count,
                    max_param_amplitudes, periods=10.):
    """
    Create a parameter set using a cosine function.
    Args:
    pulse_time :: float - the duration of the pulse
    pulse_step_count :: int - the number of time steps at which
        parameters are discretized
    param_count :: int - how many parameters are at each time step
    max_param_amplitudes :: numpy.ndarray - an array of shape
        (parameter_count) that,
        at each point, specifies the +/- value at which the parameter
        should be clipped
    periods :: float - the number of periods that the wave should complete
    Returns:
    params :: np.ndarray(pulse_step_count, param_count) - paramters for
        the specified pulse_step_count and param_count with a cosine fit
    """
    period = np.divide(pulse_step_count, periods)
    b = np.divide(2 * np.pi, period)
    params = np.zeros((pulse_step_count, param_count))
    
    # Create a wave for each parameter over all time
    # and add it to the parameters.
    for i in range(param_count):
        max_amplitude = max_param_amplitudes[i]
        _params = (np.divide(max_amplitude, 4)
                   * np.cos(b * np.arange(pulse_step_count))
                   + np.divide(max_amplitude, 2))
        params[:, i] = _params
    #ENDFOR

    return params + 1j * params


def _clip_params(max_param_amplitudes, params):
    """
    Me: I need a little taken off the top.
    Barber: Say no more.
    Args:
    max_param_amplitudes :: numpy.ndarray - an array, shaped like
        the params at each axis0 position, that specifies the maximum
        absolute value of the parameters
    params :: numpy.ndarray - the parameters to be clipped
    Returns: none
    """
    for i in range(params.shape[1]):
        max_amp = max_param_amplitudes[i]
        params[:,i] = np.clip(params[:,i], -max_amp, max_amp)


### MODULE TESTS ###


def _test():
    """
    Run test on the module.
    """
    # Test grape schroedinger discrete.

    # Evolving the state under this hamiltonian for this time should
    # perform an iSWAP. See p. 31, e.q. 109 of
    # https://arxiv.org/abs/1904.06560.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array([[1,   0,   0, 0],
                              [0,   0, -1j, 0],
                              [0, -1j,   0, 0],
                              [0,   0,   0, 1]])
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda params, t: params[0] * _hamiltonian
    initial_states = matrix_to_column_vector_list(np.eye(hilbert_size, dtype=np.complex128))
    target_states = matrix_to_column_vector_list(iswap_unitary)
    costs = [TargetInfidelity(target_states)]
    param_count = 1
    pulse_time = np.divide(np.pi, 2)
    pulse_step_count = 10
    system_step_multiplier = 1000
    iteration_count = 1
    initial_params = np.ones((pulse_step_count, param_count), dtype=np.complex128)
    magnus_policy = MagnusPolicy.M6
    log_iteration_step = 0
    save_iteration_step = 0
    save_path = None
    save_file_name = None
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         save_file_name=save_file_name,
                                         save_iteration_step=save_iteration_step,
                                         save_path=save_path,
                                         system_step_multiplier=system_step_multiplier)
    assert(np.allclose(result.error, 0))
    assert(np.allclose(result.states, target_states, atol=1e-03))
    exit(0)

    # Evolving under the zero hamiltonian should yield no change
    # in the system. Furthermore, not using parameters should
    # mean that their gradients are zero.
    # It is OK if autograd throws a warning here:
    # "UserWarning: Output seems independent of input."
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    _hamiltonian = np.zeros((hilbert_size, hilbert_size))
    hamiltonian = lambda params, t: _hamiltonian
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(identity_matrix)
    costs = [TargetInfidelity(target_states)]
    param_count = 1
    pulse_time = 10
    pulse_step_count = 10
    system_step_multiplier = 1
    iteration_count = 10
    initial_params = np.ones((pulse_step_count, param_count), dtype=np.complex128)
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         system_step_multiplier=system_step_multiplier)
    assert(result.grads.all() == 0)
    assert(np.allclose(initial_params, result.params))
    assert(np.allclose(initial_states, result.states))

    # Some nontrivial gradients should appear at each time step
    # if we evolve a nontrivial hamiltonian and penalize
    # a state against itself at each time step. Note that
    # the hamiltonian is not hermitian here.
    hilbert_size = 4
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda params, t: (params[0] * _hamiltonian)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    param_count = 1
    pulse_time = 10
    pulse_step_count = 10
    initial_params, max_param_amplitudes = _initialize_params(None, None,
                                                              pulse_time, pulse_step_count,
                                                              param_count)
    costs = [ForbidStates(forbidden_states, pulse_step_count)]
    iteration_count = 100
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_param_amplitudes=max_param_amplitudes)
    assert(result.grads.all() != 0)
    assert(not np.array_equal(result.params, initial_params))

    # If we use complex parameters on a hermitian hamiltonian,
    # the complex parameters should have no contribution to the
    # hamiltonian.
    _hamiltonian_dagger = conjugate_transpose(_hamiltonian)
    hamiltonian = lambda params, t: (params[0] * _hamiltonian
                                     + (anp.conjugate(params[0])
                                        * _hamiltonian_dagger))
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_param_amplitudes=max_param_amplitudes)
    assert(result.grads.imag.all() == 0)

    # Parameters should be clipped if they grow too large.
    # You can log result.parameters from the test above
    # that uses the same hamiltonian to see that
    # each of result.params is greater than 0.8 + 0.8j.
    hilbert_size = 4
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda params, t: (params[0] * _hamiltonian)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    param_count = 1
    pulse_time = 10
    pulse_step_count = 10
    initial_params, max_param_amplitudes = _initialize_params(None, None,
                                                              pulse_time, pulse_step_count,
                                                              param_count)
    max_param_amplitudes = np.repeat(0.8 + 0.8j, param_count)
    costs = [ForbidStates(forbidden_states, pulse_step_count)]
    iteration_count = 100
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_param_amplitudes=max_param_amplitudes)
    
    for i in range(result.params.shape[1]):
        assert(np.less_equal(np.abs(result.params[:,i]),
                             np.abs(max_param_amplitudes[i])).all())


if __name__ == "__main__":
    _test()
