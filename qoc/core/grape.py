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

import autograd.numpy as anp
from autograd import elementwise_grad
import h5py
import numpy as np
import scipy.linalg as la

from qoc.models import (MagnusPolicy, OperationPolicy, GrapeSchroedingerPolicy,
                        GrapeSchroedingerDiscreteState, GrapeResult, EvolveResult)
from qoc.standard import (Adam, TargetInfidelity, ForbidStates, expm)
from qoc.util import (PAULI_X, PAULI_Y, conjugate_transpose,
                      matrix_to_column_vector_list, matmuls)


# MAIN METHODS

def grape_schroedinger_discrete(hamiltonian, initial_states,
                                param_count, costs, iteration_count,
                                pulse_time, pulse_step_count,
                                system_step_multiplier=1, optimizer=Adam(),
                                magnus_policy=MagnusPolicy,
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
    params, max_param_amplitudes = _initialize_params(initial_params,
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
    gstate.save_initial(initial_states, params)

    if gstate.grape_schroedinger_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        states, params, error, _ = _grape_schroedinger_discrete_time(gstate, initial_states, params)
    else:
        states, params, error, _ = _grape_schroedinger_discrete_memory(gstate, initial_states, params)

    return GrapeResult(states, params, error)


# # TODO: Incorporate parameters into evolve_schroedinger.
# def evolve_schroedinger(system_hamiltonian, initial_states,
#                         pulse_time, pulse_step_count,
#                         system_step_multiplier=1,
#                         params=None,
#                         magnus_method=MagnusPolicy.M2,
#                         operation_type=OperationType.CPU):
#     """
#     Evolve a set of states under the schroedinger equation.
#     Args:
#     system_hamiltonian :: (time :: float, params :: numpy.ndarray)
#                           -> hamiltonian :: numpy.ndarray
#       - an autograd compatible (https://github.com/HIPS/autograd) function that
#         returns the system hamiltonian given the evolution time
#         and control parameters
#     initial_states :: [numpy.ndarray] - a list of the states
#         (column vectors) to evolve
#     pulse_time :: float - the duration of the control pulse, also the
#         evolution time
#     pulse_step_count :: int - the number of time steps at which the system
#         should evolve
#     system_step_multiplier :: int - this factor will be used to determine how
#         many steps inbetween each pulse step the system should evolve,
#         control parameters will be interpolated at these steps
#     params :: numpy.ndarray - an array of length pulse_step_count that should
#         be used to supply the sytem_hamiltonian with control parameters.
#         If no params are specified, then None will be passed in their place
#         to the system hamiltonian function
#     magnus_method :: qoc.MagnusMethod - the method to use for the magnus
#         expansion
#     operation_type :: qoc.OperationType - how computations should be performed,
#         e.g. CPU, GPU, sparse, etc.
#     Returns:
#     result :: qoc.models.grapestate.GrapeResult - the result of the evolution
#     """
#     # the time step over which the system will evolve
#     system_step_count = pulse_step_count * system_step_multiplier
#     dt = np.divide(pulse_time, system_step_count)

#     # choose the appropriate magnus expansion wrapper
#     if magnus_method == MagnusMethod.M2:
#         magnus = _magnus_m2
#     elif magnus_method == MagnusMethod.M4:
#         magnus = _magnus_m4
#     else:
#         magnus = _magnus_m6

#     # elvove under the schroedinger equation
#     states = copy(initial_states)
#     for i in range(system_step_count):
#         t = i * dt
#         hamiltonian = magnus(system_hamiltonian, t, dt)
#         unitary = la.expm(-1j * hamiltonian)
#         for j, state in enumerate(states):
#             states[j] = np.matmul(unitary, state)
#         #ENDFOR
#     #ENDFOR

#     return EvolveResult(states)


### HELPER METHODS ###

def _grape_schroedinger_discrete_time(gstate, initial_states, params):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use the time efficient method.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    initial_states :: numpy.ndarray - the initial states
    params :: numpy.ndarray - the initial params

    Returns:
    states :: numpy.ndarray - the states at the final time step of the last iteration
    params :: numpy.ndarray - the parameters at the final time step of the last iteration
    error :: numpy.ndarray - the errors at the final time step of the last iteration,
                             same shape as the cost function list
    grads :: numpy.ndarray - the gradients at the final time step of the last iteration,
                             same shape as params
    """
    # Compute necessary variables.
    final_step_index = (gstate.pulse_step_count - 1) * gstate.system_step_multiplier
    dt = gstate.pulse_time / final_step_index
    cost_count = len(gstate.costs)
    step_cost_count = len(gstate.step_costs)
    states_count = len(initial_states)

    # Allocate memory.
    unitaries = np.zeros((final_step_index + 1, gstate.hilbert_size, gstate.hilbert_size),
                         dtype=np.complex128)
    dunitaries_dmagnus = np.zeros((final_step_index + 1, gstate.hilbert_size, gstate.hilbert_size),
                                  dtype=np.complex128)
    states_cache = np.zeros((final_step_index + 1, states_count, gstate.hilbert_size, 1),
                            dtype=np.complex128)
    cost_propagators = np.zeros((cost_count, states_count, 1, gstate.hilbert_size),
                                dtype=np.complex128)
    step_cost_propagators = np.zeros((final_step_index + 1, step_cost_count,
                                      states_count, 1, gstate.hilbert_size),
                                     dtype=np.complex128)
    for i in range(gstate.iteration_count):
        # Reset accumulators.
        states = initial_states
        error = np.zeros(cost_count)
        grads = np.zeros_like(params)
        
        # Evolve states and compute costs.
        for j in range(final_step_index + 1):
            final_step = j == final_step_index
            pulse_step_index, pulse_step_remainder = np.divmod(j, gstate.system_step_multiplier)
            t = j * dt

            # Evolve states.
            h = gstate.magnus(params, j, t, dt)
            unitaries[j], dunitaries_dmagnus[j] = la.expm_frechet(-1j * h, h)
            states = np.matmul(unitaries[j], states)
            states_cache[j] = states

            # Compute costs. Compute gradients of costs with respect to the states
            # and parameters at step j.
            if final_step:
                for k, cost in enumerate(gstate.costs):
                    error[k] += cost.cost(params, states, j)
                    cost_propagators[k] = cost.dcost_dstates(params, states, j)
                    grads += cost.dcost_dparams(params, states, j)
            else:
                for k, step_cost in enumerate(gstate.step_costs):
                    error[gstate.step_cost_indices[k]] += step_cost.cost(params, states, j)
                    step_cost_propagators[j][k] = step_cost.dcost_dstates(params, states, j)
        #ENDFOR

        # Compute gradients and devolve cost propagators.
        for j in np.flip(range(final_step_index + 1)):
            first_step = j == 0
            pulse_step_index, pulse_step_remainder = np.divmod(j, gstate.system_step_multiplier)
            t = j * dt
            if first_step:
                states = initial_states
            else:
                states = states_cache[j - 1]
            active_step_cost_indices = final_step_index - np.arange(j - final_step_index)
            
            # Compute gradients.
            dmagnus_dparams, param_indices, _ = gstate.dmagnus_dparams(t, params, j, dt)
            dunitary_dparams = np.matmul(dunitaries_dmagnus[j], dmagnus_dparams)
            grads += matmuls()
            for state_cost_propagators in cost_propagators:
                for k, state_cost_propagator in enumerate(state_cost_propagators):
                    grads += matmuls(state_cost_propagator, dunitary_dparams, states[k])
            #ENDFOR
            for k in active_step_cost_indices:
                for state_step_cost_propagators in step_cost_propagators[k]:
                    for l, state_step_cost_propagator in state_step_cost_propagators:
                        grads += matmuls(state_step_cost_propagator, dunitary_dparams, states[l])
            #ENDFOR

            # Devolve cost propagators.
            unitary = unitaries[j]
            for k, state_cost_propagators in enumerate(cost_propagators):
                for l, state_cost_propagator in enumerate(state_cost_propagators):
                    cost_propagators[k][l] = np.matmul(state_cost_propagator, unitary)
            for k in active_step_cost_indices:
                for l, state_step_cost_propagators in enumerate(step_cost_propagators[k]):
                    for m, state_step_cost_propagator in enumerate(state_step_cost_propagators):
                        step_cost_propagators[k][l][m] = np.matmul(state_step_cost_propagator, unitary)
            #ENDFOR
        #ENDFOR
    #ENDFOR

    return states, error, params, grads


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

def _grape_schroedinger_discrete_compute(gstate, states, params):
    """
    Compute the costs for one iteration of grape.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - the grape state
    states :: [numpy.ndarray] - the initial states to evolve
    params :: numpy.ndarray - the initial control parameters
    
    Returns:
    total_error :: numpy.ndarray - total error of the optimization
    """
    final_step_index = (gstate.pulse_step_count - 1) * gstate.system_step_multiplier
    dt = gstate.pulse_time / final_step_index

    # Evolve the final states and compute cost along the way.
    total_error = 0
    for j in range(final_step_index + 1):
        final_step = j == final_step_index
        pulse_step_index, pulse_step_remainder = anp.divmod(j, gstate.system_step_multiplier)
        t = j * dt

        # Evolve.
        if final_step:
            h = gstate.magnus(params, pulse_step_index, t, dt, sentinel=True)
        else:
            h = gstate.magnus(params, pulse_step_index, t, dt)
        u = expm(-1j * h)
        states = [anp.matmul(u, state) for state in states]
        
        # Compute cost.
        if final_step:
            for cost in gstate.costs:
                total_error = total_error + cost.cost(states, j, params)
        else:
            for step_cost in gstate.step_costs:
                total_error = total_error + step_cost.cost(states, j, params)
    #ENDFOR

    return total_error


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
    states = matrix_to_column_vector_list(np.eye(hilbert_size, dtype=np.complex128))
    forbidden_states = ([[(1 / np.sqrt(2)) * np.array([[0], [0], [1], [1]], dtype=np.complex128)]]
                        * len(states))
    target_states = matrix_to_column_vector_list(np.eye(hilbert_size, dtype=np.complex128))
    pulse_time = 10
    pulse_step_count = 10
    system_step_multiplier = 1
    step_count = (pulse_step_count - 1) * system_step_multiplier + 1
    costs = [TargetInfidelity(target_states),
             ForbidStates(forbidden_states, step_count)]
    iteration_count = 1
    param_count = 2
    optimizer = Adam()
    optimizer.initialize((pulse_step_count, param_count))
    operation_policy = OperationPolicy.CPU
    log_iteration_step = 0
    save_iteration_step = 0
    save_path = None
    save_file_name = None
    grape_schroedinger_policy = GrapeSchroedingerPolicy.TIME_EFFICIENT
    params, max_param_amplitudes = _initialize_params(None, None, pulse_time,
                                                      pulse_step_count, param_count)
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
    err = grads = None
    states, params, error, grads =  _grape_schroedinger_discrete_time(gstate, states, params)

    err_compute = _grape_schroedinger_discrete_compute(gstate, states, params)
    grads_autograd = None
    grads_autograd = (elementwise_grad(_grape_schroedinger_discrete_compute, 2)
                      (gstate, states, params))
    print("err: {}\ngrads:\n{}\nerr_compute: {}\ngrads_autograd:\n{}"
          "".format(err, grads, err_compute, grads_autograd))
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
