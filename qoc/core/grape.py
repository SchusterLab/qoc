"""
grape.py - a module to expose optimization methods for the GRAPE
algorithm

conventions:
dy_dx denotes the gradient of y with respect to x
h refers to the hamiltonian
u refers to the unitary exp^(-j * h * dt)
e refers to the control parameters
"""

from copy import copy
import os

import h5py
import numpy as np
import scipy.linalg as la

from qoc.core.maths import (interpolate_trapezoid,
                      magnus_m2, magnus_m4, magnus_m6)
from qoc.util import (PAULI_X, PAULI_Y, matrix_to_column_vector_list)
from qoc.models import (MagnusMethod, OperationType, GrapeResult, EvolveResult,
                        Adam)

### MAIN METHODS ###

def grape_schroedinger_discrete(system_hamiltonian, parameter_count,
                                initial_states, costs, iteration_count,
                                pulse_time, pulse_step_count,
                                system_step_multiplier=1, optimizer=Adam(),
                                magnus_method=MagnusMethod.M2,
                                operation_type=OperationType.CPU,
                                initial_parameters=None,
                                max_parameter_amplitudes=None,
                                log_iteration_step=100,
                                save_iteration_step=0, save_file_name=None,
                                save_path=None):
    """
    a method to optimize the evolution of a set of states under the
    schroedinger equation for time-discrete control parameters
    Args:
    system_hamiltonian :: (time :: float, params :: numpy.ndarray)
                          -> hamiltonian :: numpy.ndarray
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the evolution time
        and control parameters
    parameter_count :: int - the number of control parameters required at each
         optimization (pulse) time step
    initial_states :: [numpy.ndarray] - a list of the states
        (column vectors) to evolve
        A column vector is specified as np.array([[0], [1], [2]]).
        A column vector is NOT a row vector np.array([0, 1, 2]).
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
    magnus_method :: qoc.MagnusMethod - the method to use for the magnus
        expansion
    operation_type :: qoc.OperationType - how computations should be performed,
        e.g. CPU, GPU, sparse, etc.
    initial_parameters :: numpy.ndarray - These are the values to use for the
        parameters for the first iteration.
        This array should have shape (pulse_step_count, parameter_count)
    max_parameter_amplitudes :: numpy.ndarray - These are the values at
        which to clip the parameters if they achieve +max_amplitude
        or -max_amplitude. This array should have shape
        (parameter_count). The default maximum amplitudes will
        be 1 if not specified. 
    log_iteration_step :: int - how often to write to stdout,
        set 0 to disable logging
    save_iteration_step :: int - how often to write to the save file,
        set 0 to disable saving
    save_file_name :: str - this will identify the save file
    save_path :: str - the directory to create the save file in,
        the directory will be created if it does not exist
    Returns:
    result :: qoc.models.grapestate.GrapeResult - the result of the optimization
    """
    # Create the save file if one should be created.
    save_file_path = None
    if save_iteration_step != 0:
        save_file_path = _create_save_file_path(save_path, save_file_name)
        # Save initial information to file.
        with h5py.File(save_file_path, "w") as save_file:
            save_file.create_dataset("parameter_count", data=parameter_count)
            save_file.create_dataset("initial_states", data=initial_states)
            save_file.create_dataset("cost_names",
                                     data=["{}".format(cost) for cost in costs])
            save_file.create_dataset("iteration_count", data=iteration_count)
            save_file.create_dataset("pulse_time", data=pulse_time)
            save_file.create_dataset("pulse_step_count", data=pulse_step_count)
            save_file.create_dataset("system_step_multiplier",
                                     data=pulse_step_count)
            save_file.create_dataset("optimizer", data="{}".format(optimizer))
            save_file.create_dataset("magnus_method",
                                     data="{}".format(magnus_method))
            save_file.create_dataset("operation_type",
                                     data="{}".format(operation_type))
            save_file.create_dataset("initial_parameters",
                                     data=initial_parameters)
    #ENDIF
    
    # Initialize parameters.
    params_shape = (pulse_step_count, parameter_count)
    if max_parameter_amplitudes == None:
        max_parameter_amplitudes = np.ones(parameter_count)
    if initial_parameters == None:
        params = _gen_params_cos(pulse_time, pulse_step_count,
                                 max_parameter_amplitudes,
                                     parameter_count)
    else:
        # Ensure the initial parameters have the right shape.
        if initial_parameters.shape != params_shape:
            raise ValueError("initial_parameters must have shape "
                             "(pulse_step_count, parameter_count) "
                             "but has shape {}"
                             "".format(initial_parameters.shape))
        # Ensure the initial parameters have the correct amplitude.
        for i in range(parameter_count):
            if not np.less(max_parameter_amplitudes[i],
                           initial_parameters[:, i]):
                raise ValueError("initial_parameters must have "
                                 "amplitude that conforms to "
                                 "max_parameter_amplitudes")
        #ENDFOR
        params = initial_parameters

    # Initialize optimizer.
    optimizer.initialize(params_shape)

    # Seperate step costs.
    step_costs = [cost for cost in costs if cost.requires_step_evaluation]

    # Define the time step over which the system will evolve.
    system_step_count = pulse_step_count * system_step_multiplier
    dt = np.divide(pulse_time, system_step_count)

    # Choose the appropriate magnus expansion wrapper.
    if magnus_method == MagnusMethod.M2:
        magnus_expansion = _magnus_m2
    elif magnus_method == MagnusMethod.M4:
        magnus_expansion = _magnus_m4
    else:
        magnus_expansion = _magnus_m6

    # Run optimization for iteration_count iterations.
    final_step_index = (pulse_step_count - 1) * system_step_multiplier
    for i in range(iteration_count):
        states = copy(initial_states)
        error = 0
        # Evolve the states for every system step.
        for j in range(final_step_index + 1):
            # Determine if this is a pulse step (if so which?).
            pulse_step_index, pulse_step_remainder = np.divmod(j, system_step_multiplier)
            pulse_step = pulse_step_remainder == 0
            # Determine if this is the final step.
            final_step = j == final_step_index
            # If this is a pulse step, and not the last step,
            # update the parameters.
            if pulse_step and not final_step:
                params_left = params[pulse_step_index]
                params_right = params[pulse_step_index + 1]

            # Compute the hamiltonian at this step.
            t = j * dt
            hamiltonian = magnus_expansion(system_hamiltonian, t, dt,
                                           params_left, params_right)
            unitary = la.expm(-1j * hamiltonian)
            # Evolve the states under the hamiltonian at this step.
            for k, state in enumerate(states):
                states[k] = np.matmul(unitary, state)
            
            # If this is a pulse step, and not the final step,
            # compute step costs.
            if pulse_step and not final_step:
                for step_cost in step_costs:
                    error += step_cost.compute(pulse_step, states, params)
            #ENDIF
            # If this is the final step, count all penalties.
            if final_step:
                for cost in costs:
                    error += cost.compute(pulse_step, state, params)
            #ENDIF
        #ENDFOR

        # TODO: Backpropagate all gradients.
        grads = np.zeros_like(params)
        for j in np.flip(range(final_step_index + 1))):
            # Determine if this is a pulse step (if so which?).
            pulse_step_index, pulse_step_remainder = np.divmod(j, system_step_multiplier)
            pulse_step = pulse_step_remainder == 0
            # Determine if this is the first step.
            first_step = j == 0
            # If this is a pulse step, and not the first step,
            # update the parameters.
            if pulse_step and not first_step:
                params_left = params[pulse_step_index]
                params_right = params[pulse_step_index - 1]
                
            # Compute the hamiltonian at this step.
            t = j * dt
            hamiltonian = -1j * magnus_expansion(system_hamiltonian, t, dt,
                                                 params_left, params_right)
            unitary, du_dh = la.expm_frechet(hamiltonian, hamiltonian)
        #ENDFOR
    #ENDFOR
            

        
    

# TODO: Incorporate parameters into evolve_schroedinger.
def evolve_schroedinger(system_hamiltonian, initial_states,
                        pulse_time, pulse_step_count,
                        system_step_multiplier=1,
                        params=None,
                        magnus_method=MagnusMethod.M2,
                        operation_type=OperationType.CPU):
    """
    Evolve a set of states under the schroedinger equation.
    Args:
    system_hamiltonian :: (time :: float, params :: numpy.ndarray)
                          -> hamiltonian :: numpy.ndarray
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the evolution time
        and control parameters
    initial_states :: [numpy.ndarray] - a list of the states
        (column vectors) to evolve
    pulse_time :: float - the duration of the control pulse, also the
        evolution time
    pulse_step_count :: int - the number of time steps at which the system
        should evolve
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps
    params :: numpy.ndarray - an array of length pulse_step_count that should
        be used to supply the sytem_hamiltonian with control parameters.
        If no params are specified, then None will be passed in their place
        to the system hamiltonian function
    magnus_method :: qoc.MagnusMethod - the method to use for the magnus
        expansion
    operation_type :: qoc.OperationType - how computations should be performed,
        e.g. CPU, GPU, sparse, etc.
    Returns:
    result :: qoc.models.grapestate.GrapeResult - the result of the evolution
    """
    # the time step over which the system will evolve
    system_step_count = pulse_step_count * system_step_multiplier
    dt = np.divide(pulse_time, system_step_count)

    # choose the appropriate magnus expansion wrapper
    if magnus_method == MagnusMethod.M2:
        magnus_expansion = _magnus_m2
    elif magnus_method == MagnusMethod.M4:
        magnus_expansion = _magnus_m4
    else:
        magnus_expansion = _magnus_m6

    # elvove under the schroedinger equation
    states = copy(initial_states)
    for i in range(system_step_count):
        t = i * dt
        hamiltonian = magnus_expansion(system_hamiltonian, t, dt)
        unitary = la.expm(-1j * hamiltonian)
        for j, state in enumerate(states):
            states[j] = np.matmul(unitary, state)
        #ENDFOR
    #ENDFOR

    return EvolveResult(states)


### HELPER METHODS ###

_MAGNUS_M4_C1 = np.divide(1, 2) - np.divide(np.sqrt(3), 6)
_MAGNUS_M4_C2 = np.divide(1, 2) + np.divide(np.sqrt(3), 6)
_MAGNUS_M6_C1 = np.divide(1, 2) - np.divide(np.sqrt(15), 10)
_MAGNUS_M6_C2 = np.divide(1, 2)
_MAGNUS_M6_C3 = np.divide(1, 2) + np.divide(np.sqrt(15), 10)

def _magnus_m2(system_hamiltonian, t, dt, params_left=None,
                      params_right=None):
    """
    Evaluate the m2 magnus expansion of the system hamiltonian,
    that depends on control parameters, between time t and t + dt
    which have params_left and params_right,
    respectively. See https://arxiv.org/abs/1709.06483 for details.
    We take our own liberties here and do not evaluate at the midpoint
    between t and t + dt but instead evaluate at t.
    Args:
    system_hamiltonian :: (time :: float, params :: np.ndarray) ->
                          hamiltonian :: np.ndarray
        - the hamiltonian to expand
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    params_left :: np.ndarray - the parameters at time step t which
        define the left hand side of the interpolation
    params_right :: np.ndarray - the parameters at time step t + dt
        which define the right hand side of the interpolation
    Returns:
    magnus :: np.ndarray - the m2 magnus expansion of the sytem hamiltonian
    """
    return magnus_m2(system_hamiltonian(t, params_left), dt)


def _magnus_m4(system_hamiltonian, t, dt, params_left=None,
                      params_right=None):
    """
    Evaluate the m4 magnus expansion of the system hamiltonian,
    that depends on control parameters, between time t and t + dt
    which have params_left and params_right,
    respectively. See https://arxiv.org/abs/1709.06483 for details.
    Args:
    system_hamiltonian :: (time :: float, params :: np.ndarray) ->
                          hamiltonian :: np.ndarray
        - the hamiltonian to expand
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    params_left :: np.ndarray - the parameters at time step t which
        define the left hand side of the interpolation
    params_right :: np.ndarray - the parameters at time step t + dt
        which define the right hand side of the interpolation
    Returns:
    magnus :: np.ndarray - the m4 magnus expansion of the sytem hamiltonian
    """
    t1 = t + dt * _MAGNUS_M4_C1
    t2 = t + dt * _MAGNUS_M4_C2
    # if parameters were supplied, interpolate on them
    if params_left and params_right:
        params1 = interpolate_trapezoid(params_left, params_right, t, t + dt,
                                        t1)
        params2 = interpolate_trapezoid(params_left, params_right, t, t + dt,
                                        t2)
    else:
        params1 = 0
        params2 = 0
    a1 = system_hamiltonian(t1, params1)
    a2 = system_hamiltonian(t2, params2)
    return magnus_m4(a1, a2, dt)


def _magnus_m6(system_hamiltonian, t, dt, params_left=None, params_right=None):
    """
    Evaluate the m6 magnus expansion of the system hamiltonian,
    that depends on control parameters, between time t and t + dt
    which have params_left and params_right,
    respectively. See https://arxiv.org/abs/1709.06483 for details.
    Args:
    system_hamiltonian :: (time :: float, params :: np.ndarray) ->
                          hamiltonian :: np.ndarray
        - the hamiltonian to expand
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    params_left :: np.ndarray - the parameters at time step t which
        define the left hand side of the interpolation
    params_right :: np.ndarray - the parameters at time step t + dt
        which define the right hand side of the interpolation
    Returns:
    magnus :: np.ndarray - the m6 magnus expansion of the sytem hamiltonian
    """
    t1 = t + dt * _MAGNUS_M6_C1
    t2 = t + dt * _MAGNUS_M6_C2
    t3 = t + dt * _MAGNUS_M6_C3
    # if parameters were supplied, interpolate on them
    if params_left and params_right:
        params1 = interpolate_trapezoid(params_left, params_right, t, t + dt,
                                        t1)
        params2 = interpolate_trapezoid(params_left, params_right, t, t + dt,
                                        t2)
        params3 = interpolate_trapezoid(params_left, params_right, t, t + dt,
                                        t3)
    else:
        params1 = 0
        params2 = 0
        params3 = 0
    a1 = system_hamiltonian(t1, params1)
    a2 = system_hamiltonian(t2, params2)
    a3 = system_hamiltonian(t2, params3)
    return magnus_m6(a1, a2, a3, dt)


def _create_save_file_path(save_path, save_file_name):
    """
    Create the full path to an h5 file using the base name
    save_file_name in the path save_path. File name conflicts are avoided
    by appending a numeric prefix to the file name. This method assumes
    that all objects in save_path that contain _{save_file_name}.h5
    are created with this convention.
    """
    # Ensure the path exists.
    os.makedirs(save_path, exist_ok=True)
    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory. 
    max_numeric_prefix = -1
    for file_name in os.listdir(save_path):
        if file_name.contains("_{}.h5".format(save_file_name)):
            max_numeric_prefix = max(int(file_name.split("_")[0]),
                                     max_numeric_prefix)
    #ENDFOR
    save_file_name_augmented = ("{:05d}_{}.h5"
                                "".format(max_numeric_prefix + 1,
                                          save_file_name))
    return os.path.join(save_path, save_file_name_augmented)


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
    # test grape_schoredinger_discrete
    # Evolving the state under no system hamiltonian should
    # do nothing to the state.
    d = 2
    identity_matrix = np.eye(d)
    zero_matrix = np.zeros((d, d))
    system_hamiltonian = lambda params, t : zero_matrix
    initial_states = [np.array([[1], [0]])]
    expected_states = initial_states
    pulse_time = 10
    system_step_count = 10
    magnus_method = MagnusMethod.M2
    result = evolve_schroedinger(system_hamiltonian, initial_states,
                                 pulse_time, system_step_count,
                                 magnus_method=magnus_method,)
    final_states = np.array(result.final_states)
    for i, expected_state in enumerate(expected_states):
        final_state = final_states[i]
        assert(np.allclose(expected_state, final_state))
    #ENDFOR
    
    # Evolving the state under this hamiltonian for this time should
    # perform an iSWAP. See p. 31, e.q. 109 of
    # https://arxiv.org/abs/1904.06560.
    d = 4
    identity_matrix = np.eye(d)
    iswap_unitary = np.array([[1, 0, 0, 0],
                              [0, 0, -1j, 0],
                              [0, -1j, 0, 0],
                              [0, 0, 0, 1]])
    hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    system_hamiltonian = lambda params, t: hamiltonian
    initial_states = matrix_to_column_vector_list(identity_matrix)
    expected_states = matrix_to_column_vector_list(iswap_unitary)
    pulse_time = np.divide(np.pi, 2)
    system_step_count = 10
    magnus_method = MagnusMethod.M2
    result = evolve_schroedinger(system_hamiltonian, initial_states,
                                 pulse_time, system_step_count,
                                 magnus_method=magnus_method,)
    final_states = np.array(result.final_states)
    for i, expected_state in enumerate(expected_states):
        final_state = final_states[i]
        assert(np.allclose(expected_state, final_state))
    #ENDFOR


if __name__ == "__main__":
    _test()
