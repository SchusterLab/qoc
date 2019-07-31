"""
grape.py - a module to expose optimization methods for the GRAPE
algorithm

conventions:
dy_dx denotes the gradient of y with respect to x
h refers to the hamiltonian
u refers to the unitary ~ exp^(-j * h * dt)
e refers to the control parameters
"""

from copy import copy
import os
import time

import autograd.numpy as anp
from autograd import elementwise_grad
import h5py
import numpy as np
import scipy.linalg as la

from qoc.models import (MagnusPolicy, OperationPolicy, GrapeSchroedingerPolicy,
                        GrapeSchroedingerDiscreteState, GrapeResult, EvolveResult,
                        InterpolationPolicy)
from qoc.standard import (Adam, TargetInfidelity, ForbidStates, expm)
from qoc.util import (PAULI_X, PAULI_Y, conjugate_transpose,
                      matrix_to_column_vector_list, matmuls, ans_jacobian,)


### MAIN METHODS ###

def grape_schroedinger_discrete(hamiltonian, initial_states,
                                param_count, costs, iteration_count,
                                pulse_time, pulse_step_count,
                                system_step_multiplier=1, optimizer=Adam(),
                                magnus_policy=MagnusPolicy.M2,
                                operation_policy=OperationPolicy.CPU,
                                grape_schroedinger_policy=GrapeSchroedingerPolicy.TIME_EFFICIENT,
                                initial_params=None, max_param_amplitudes=None,
                                log_iteration_step=100,
                                save_iteration_step=0, save_path=None,
                                save_file_name=None):
    """
    a method to optimize the evolution of a set of states under the
    schroedinger equation for time-discrete control parameters
    Args:
    hamiltonian :: (params :: numpy.ndarray, time :: float)
                    -> hamiltonian :: numpy.ndarray
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the control parameters
        and evolution time
    initial_states :: numpy.ndarray - a list of the states (column vectors)
        to evolve
        A column vector is specified as np.array([[0], [1], [2]]).
        A column vector is NOT a row vector np.array([0, 1, 2]).
    param_count :: int - the number of control parameters required at each
         optimization (pulse) time step
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    iteration_count :: int - the number of iterations to optimize for
    pulse_time :: float - the duration of the control pulse, also the
        evolution time
    pulse_step_count :: int - the number of time steps at which the pulse
        should be updated (optimized)
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps
    optimizer :: qoc.models.Optimizer - an instance of an optimizer to perform
        gradient-based optimization
    magnus_policy :: qoc.MagnusPolicy - the method to use for the magnus
        expansion
    operation_policy :: qoc.OperationPolicy - how computations should be performed,
        e.g. CPU, GPU, CPU-sparse, GPU-spares, etc.
    grape_schroedinger_policy :: qoc.GrapeSchroedingerPolicy - how to perform
        the main integration of GRAPE, can be optimized for time or memory
    initial_params :: numpy.ndarray - the values to use for the
        parameters for the first iteration,
        This array should have shape (pulse_step_count, parameter_count)
    max_param_amplitudes :: numpy.ndarray - These are the absolute values at
        which to clip the parameters if they achieve +max_amplitude
        or -max_amplitude. This array should have shape
        (parameter_count). The default maximum amplitudes will
        be 1 if not specified. 
    log_iteration_step :: int - how often to write to stdout,
        set 0 to disable logging
    save_iteration_step :: int - how often to write to the save file,
        set 0 to disable saving
    save_path :: str - the directory to create the save file in,
        the directory will be created if it does not exist
    save_file_name :: str - this will identify the save file
    Returns:
    result :: qoc.models.grapestate.GrapeResult - the result of the optimization
    """
    # Initialize parameters.
    initial_params, max_param_amplitudes = _initialize_params(initial_params,
                                                              max_param_amplitudes,
                                                              pulse_time,
                                                              pulse_step_count,
                                                              param_count)
    
    # Initialize optimizer.
    optimizer.initialize((pulse_step_count, param_count))
    
    # Construct the grape state.
    hilbert_size = initial_states[0].shape[0]
    gstate = GrapeSchroedingerDiscreteState(costs, iteration_count,
                                            max_param_amplitudes,
                                            pulse_time, pulse_step_count,
                                            optimizer, operation_policy,
                                            hilbert_size,
                                            log_iteration_step,
                                            save_iteration_step,
                                            save_path, save_file_name,
                                            hamiltonian, magnus_policy,
                                            param_count, system_step_multiplier,
                                            grape_schroedinger_policy)

    # Perform initial log and save.
    if gstate.log_iteration_step != 0:
        gstate.log_initial()
    if gstate.save_iteration_step != 0:
        gstate.save_initial(initial_params, initial_states)

    # Switch on GRAPE implementation method.
    if gstate.grape_schroedinger_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        error, _, params, states = _grape_schroedinger_discrete_time(gstate,
                                                                     initial_params,
                                                                     initial_states,)
    else:
        error, _, params, states = _grape_schroedinger_discrete_memory(gstate,
                                                                       initial_params,
                                                                       initial_states,)

    return GrapeResult(error, params, states)


### HELPER METHODS ###

def _grape_schroedinger_discrete_time(gstate, initial_states, params):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    initial_states :: numpy.ndarray - the initial states
    params :: numpy.ndarray - the initial params

    Returns:
    error :: numpy.ndarray - the errors at the final time step of the last iteration,
                             same shape as the cost function list
    grads :: numpy.ndarray - the gradients at the final time step of the last iteration,
                             same shape as params
    params :: numpy.ndarray - the parameters at the final time step of the last iteration
    states :: numpy.ndarray - the states at the final time step of the last iteration
    """
    final_iteration = gstate.iteration_count - 1
    final_time_step = (gstate.pulse_step_count - 1) * gstate.system_step_multiplier
    dt = gstate.pulse_time / final_time_step

    # Run optimization for the given number of iterations.
    for iteration in range(gstate.iteration_count):
        # Compute the gradient of the costs with respect
        # to params.
        error, grads = _gsd_compute_ans_jac(dt, final_time_step, gstate,
                                            initial_states, params)
        
        # Log and save progress.
        if (gstate.log_iteration_step != 0
              and np.mod(iteration, gstate.log_iteration_step) == 0):
            gstate.log(error, grads, iteration)
            
        if (gstate.save_iteration_step != 0
              and np.mod(iteration, gstate.save_iteration_step) == 0):
            gstate.save(error, grads, iteration, params, states)

        # Update params.
        if iteration != final_iteration:
            gstate.optimizer.update(grads, params)
    #ENDFOR
            
    return error, grads, params, None


def _grape_schroedinger_discrete_compute(dt, final_time_step,
                                         gstate, states, params):
    """
    Compute the costs for one evolution cycle.
    Args:
    dt :: float - the time step between evolution steps
    final_time_step :: int - the index of the last evolution step
    gstate :: qoc.GrapeSchroedingerDiscreteState - the grape state
    states :: numpy.ndarray - the initial states to evolve
    params :: numpy.ndarray - the control parameters
    
    Returns:
    total_error :: numpy.ndarray - total error of the evolution
    """
    total_error = 0
    for time_step in range(final_time_step + 1):
        is_final_step = time_step == final_time_step
        t = time_step * dt

        # Evolve.
        # Get the parameters to use for magnus expansion interpolation.
        magnus_param_indices = gstate.magnus_param_indices(dt, params,
                                                           time_step, t,
                                                           is_final_step)
        magnus_params = params[magnus_param_indices,]
        # If magnus_params includes only one parameter array,
        # wrap it in another dimension.
        if magnus_params.ndim == 1:
            magnus_params = anp.expand_dims(magnus_params, axis=0)
        magnus = gstate.magnus(dt, magnus_params, time_step, t, is_final_step)
        unitary = expm(-1j * magnus)
        states = anp.matmul(unitary, states)
        
        # Compute cost.
        if is_final_step:
            for cost in gstate.costs:
                total_error = total_error + cost.cost(params, states, time_step)
        else:
            for step_cost in gstate.step_costs:
                total_error = total_error + step_cost.cost(params, states, time_step)
    #ENDFOR
    return total_error


# Real wrapper for gsd_compute.
_gsd_compute_real = (lambda *args, **kwargs:
                     anp.real(_grape_schroedinger_discrete_compute(*args,
                                                                   **kwargs)))


# Value and jacobian of gsd_compute.
_gsd_compute_ans_jac = ans_jacobian(_gsd_compute_real, 4)


def _grape_schroedinger_discrete_memory(gstate, states, params):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use the memory efficient method.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    states :: numpy.ndarray - the initial states
    params :: numpy.ndarray - the initial params

    Returns:
    states :: numpy.ndarray - the states at the final time step of the last iteration
    params :: numpy.ndarray - the parameters at the final time step of the last iteration
    error :: numpy.ndarray - the errors at the final time step of the last iteration,
                             same shape as the cost function list
    grads :: numpy.ndarray - the gradients at the final time step of the last iteration,
                             same shape as params
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
    if max_param_amplitudes == None:
        max_param_amplitudes = np.ones(param_count)
        
    if initial_params == None:
        params = _gen_params_cos(pulse_time, pulse_step_count, param_count,
                                 max_param_amplitudes)
        params = params + 1j * params
    else:
        # If the user specified initial params, check that they conform to
        # max param amplitudes.
        for i, step_params in enumerate(initial_params):
            if not np.less(step_params, max_param_amplitudes).all():
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

    return params


### MODULE TESTS ###


def _test():
    """
    Run test on the module.
    """
    # Test grape schroedinger discrete.
    
    # Our main integration implementation should yield the same
    # gradients as autograd would.
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    _hamiltonian_dagger = conjugate_transpose(_hamiltonian)
    hamiltonian = lambda params, t: (params[0] * _hamiltonian
                                     + anp.conjugate(params[0]) * _hamiltonian_dagger)
    magnus_policy = MagnusPolicy.M2
    hilbert_size = 4
    initial_states = matrix_to_column_vector_list(np.eye(hilbert_size, dtype=np.complex128))
    forbidden_states = ([[(1 / np.sqrt(2)) * np.array([[0], [0], [1], [1]], dtype=np.complex128)]]
                        * len(initial_states))
    target_states = matrix_to_column_vector_list(np.eye(hilbert_size, dtype=np.complex128))
    pulse_time = 10
    pulse_step_count = 1000
    system_step_multiplier = 1
    step_count = (pulse_step_count - 1) * system_step_multiplier + 1
    costs = [TargetInfidelity(target_states),
             ForbidStates(forbidden_states, step_count)]
    iteration_count = 1
    param_count = 2
    optimizer = Adam()
    optimizer.initialize((pulse_step_count, param_count))
    operation_policy = OperationPolicy.CPU
    interpolation_policy = InterpolationPolicy.LINEAR
    log_iteration_step = 0
    save_iteration_step = 0
    save_path = None
    save_file_name = None
    grape_schroedinger_policy = GrapeSchroedingerPolicy.TIME_EFFICIENT
    initial_params, max_param_amplitudes = _initialize_params(None, None, pulse_time,
                                                              pulse_step_count, param_count)

    gstate = GrapeSchroedingerDiscreteState(costs, grape_schroedinger_policy,
                                            hamiltonian, hilbert_size,
                                            interpolation_policy,
                                            iteration_count,
                                            log_iteration_step,
                                            magnus_policy,
                                            max_param_amplitudes,
                                            operation_policy,
                                            optimizer,
                                            param_count,
                                            pulse_step_count,
                                            pulse_time,
                                            save_file_name,
                                            save_iteration_step,
                                            save_path,
                                            system_step_multiplier,)
    g2_start_time = time.perf_counter()
    g2_error, g2_grads, g2_params, g2_states =  _grape_schroedinger_discrete_time(gstate,
                                                                                  initial_states,
                                                                                  initial_params)
    g2_end_time = time.perf_counter()
    g2_run_time = g2_end_time - g2_start_time
    g2_total_error = np.sum(g2_error, axis=0)
    print("g2_run_time:{}\n"
          "g2_total_error:{}\n"
          "g2_error:\n{}\n"
          "g2_grads.shape:{}\n"
          "g2_grads:\n{}\n"
          "g2_params.shape:{}\n"
          "g2_params:\n{}\n"
          "g2_states:\n{}"
          "".format(g2_run_time, g2_total_error,
                    g2_error,
                    g2_grads.shape,
                    g2_grads,
                    g2_params.shape,
                    g2_params, g2_states))

    # assert(np.allclose(grads, grads_autograd))

    
    # # test evolve
    # # Evolving the state under no system hamiltonian should
    # # do nothing to the state.
    # d = 2
    # identity_matrix = np.eye(d)
    # zero_matrix = np.zeros((d, d))
    # system_hamiltonian = lambda params, t : zero_matrix
    # initial_states = [np.array([[1], [0]])]
    # expected_states = initial_states
    # pulse_time = 10
    # system_step_count = 10
    # magnus_method = MagnusMethod.M2
    # result = evolve_schroedinger(system_hamiltonian, initial_states,
    #                              pulse_time, system_step_count,
    #                              magnus_method=magnus_method,)
    # final_states = np.array(result.final_states)
    # for i, expected_state in enumerate(expected_states):
    #     final_state = final_states[i]
    #     assert(np.allclose(expected_state, final_state))
    # #ENDFOR
    
    # # Evolving the state under this hamiltonian for this time should
    # # perform an iSWAP. See p. 31, e.q. 109 of
    # # https://arxiv.org/abs/1904.06560.
    # d = 4
    # identity_matrix = np.eye(d)
    # iswap_unitary = np.array([[1, 0, 0, 0],
    #                           [0, 0, -1j, 0],
    #                           [0, -1j, 0, 0],
    #                           [0, 0, 0, 1]])
    # hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
    #                                  + np.kron(PAULI_Y, PAULI_Y))
    # system_hamiltonian = lambda params, t: hamiltonian
    # initial_states = matrix_to_column_vector_list(identity_matrix)
    # expected_states = matrix_to_column_vector_list(iswap_unitary)
    # pulse_time = np.divide(np.pi, 2)
    # system_step_count = 10
    # magnus_method = MagnusMethod.M2
    # result = evolve_schroedinger(system_hamiltonian, initial_states,
    #                              pulse_time, system_step_count,
    #                              magnus_method=magnus_method,)
    # final_states = np.array(result.final_states)
    # for i, expected_state in enumerate(expected_states):
    #     final_state = final_states[i]
    #     assert(np.allclose(expected_state, final_state))
    # #ENDFOR


if __name__ == "__main__":
    _test()
