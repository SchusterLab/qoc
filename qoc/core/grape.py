"""
grape.py - a module to expose optimization methods for the GRAPE
algorithm
"""

from copy import deepcopy

import numpy as np
import scipy.linalg as la

from qoc.core.maths import (interpolate_trapezoid,
                      magnus_m2, magnus_m4, magnus_m6)
from qoc.util.mathutil import (PAULI_X, PAULI_Y)
from qoc.models import (MagnusMethod, OperationType, GrapeResult, EvolveResult)

### MAIN METHODS ###

def grape_schroedinger_discrete(system_hamiltonian, parameter_count,
                                initial_states, costs, iteration_count,
                                pulse_time, pulse_step_count,
                                system_step_multiplier, optimizer,
                                magnus_method, operation_type,
                                initial_parameters, log_iteration_step,
                                save_iteration_step, save_file_name,
                                save_file_path):
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
         optimization time step
    initial_states :: [numpy.ndarray] - a list of the states
        (column vectors) to evolve
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    iteration_count :: int - the number of iterations to optimize for
    pulse_time :: float - the duration of the control pulse, also the
        evolution time
    pulse_step_count :: int - the number of time steps at which the pulse
        should be optimized
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps
    optimizer :: qoc.models.Optimizer - an instance of an optimizer to perform
        gradient-based optimization
    magnus_method :: qoc.MagnusMethod - the method to use for the magnus
        expansion
    operation_type :: qoc.OperationType - how computations should be performed,
        e.g. CPU, GPU, sparse, etc.
    initial_parameters :: numpy.ndarray - values to use for the parameters for
        the first iteration
    log_iteration_step :: int - how often to write to stdout,
        set 0 to disable logging
    save_iteration_step :: int - how often to write to the save file,
        set 0 to disable saving
    save_file_name :: str - this will identify the save file
    save_file_path :: str - the directory to create the save file in,
        the directory will be created if it does not exist
    Returns:
    result :: qoc.models.grapestate.GrapeResult - the result of the optimization
    """
    pass


def evolve_schroedinger(system_hamiltonian, initial_states,
                        evolution_time, pulse_step_count,
                        system_step_multiplier=1,
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
    evolution_time :: float - the duration of evolution
    pulse_step_count :: int - the number of time steps at which the system
        should evolve
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps
    params :: numpy.ndarray - an array of length pulse_step_count that should
        be used to supply the sytem_hamiltonian with control parameters
    magnus_method :: qoc.MagnusMethod - the method to use for the magnus
        expansion
    operation_type :: qoc.OperationType - how computations should be performed,
        e.g. CPU, GPU, sparse, etc.
    Returns:
    result :: qoc.models.grapestate.GrapeResult - the result of the evolution
    """
    # the time step over which the system will evolve
    system_step_count = pulse_step_count * system_step_multiplier
    dt = np.divide(evolution_time, system_step_count)

    # choose the appropriate magnus expansion wrapper
    if magnus_method == MagnusMethod.M2:
        magnus = _magnus_m2
    elif magnus_method == MagnusMethod.M4:
        magnus = _magnus_m4
    else:
        magnus = _magnus_m6

    # elvove under the schroedinger equation
    states = deepcopy(initial_states)
    for i in range(system_step_count):
        t = i * dt
        expanded_hamiltonian = magnus(system_hamiltonian, t, dt)
        unitary = la.expm(-1j * expanded_hamiltonian)
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
        which define the left hand side of the interpolation
    Returns:
    magnus :: np.ndarray - the m2 magnus expansion of the sytem hamiltonian
    """
    a1 = system_hamiltonian(t, params_left)
    return magnus_m2(a1, dt)


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
        which define the left hand side of the interpolation
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
        which define the left hand side of the interpolation
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
    result = evolve_schroedinger(system_hamiltonian, initial_states,
                                 pulse_time, system_step_count,)
    final_states = np.array(result.final_states)
    for i, expected_state in enumerate(expected_states):
        final_state = final_states[i]
        assert(np.allclose(expected_state, final_state))
    
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
    result = evolve_schroedinger(system_hamiltonian, initial_states,
                                 pulse_time, system_step_count,)
    final_states = np.array(result.final_states)
    for i, expected_state in enumerate(expected_states):
        final_state = final_states[i]
        assert(np.allclose(expected_state, final_state))



if __name__ == "__main__":
    _test()
