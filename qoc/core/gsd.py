"""
gsd.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

import os

from autograd import jacobian
import autograd.numpy as anp
from autograd.extend import Box
import numpy as np
import scipy.linalg as la

from qoc.core.common import (gen_params_cos, initialize_params,
                             slap_params, strip_params,)
from qoc.models import (MagnusPolicy, OperationPolicy, GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerPolicy, GrapeResult,
                        InterpolationPolicy, Dummy)
from qoc.standard import (ans_jacobian, Adam, SGD, ForbidStates, TargetInfidelity,
                          ParamValue, expm,
                          PAULI_X, PAULI_Y, PAULI_Z, conjugate_transpose,
                          matrix_to_column_vector_list, matmuls,
                          complex_to_real_imag_flat,
                          real_imag_to_complex_flat,
                          get_annihilation_operator,
                          get_creation_operator)


### MAIN METHODS ###

def grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                iteration_count, param_count, pulse_step_count,
                                pulse_time,
                                grape_schroedinger_policy=GrapeSchroedingerPolicy.TIME_EFFICIENT,
                                initial_params=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                log_iteration_step=100,
                                magnus_policy=MagnusPolicy.M2,
                                max_param_norms=None,
                                operation_policy=OperationPolicy.CPU,
                                optimizer=Adam(),
                                save_file_name=None, save_iteration_step=0,
                                save_path=None, system_step_multiplier=1,):
    """
    a method to optimize the evolution of a set of states under the
    schroedinger equation for time-discrete control parameters

    Args:
    costs :: tuple(qoc.models.Cost) - the cost functions to guide optimization
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
    max_param_norms :: numpy.ndarray - These are the absolute values at
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
    initial_params, max_param_norms = initialize_params(initial_params,
                                                        max_param_norms,
                                                        pulse_time,
                                                        pulse_step_count,
                                                        param_count)
    
    # Construct the grape state.
    hilbert_size = initial_states[0].shape[0]
    gstate = GrapeSchroedingerDiscreteState(costs, grape_schroedinger_policy,
                                            hamiltonian, hilbert_size,
                                            initial_params,
                                            initial_states,
                                            interpolation_policy, iteration_count,
                                            log_iteration_step, magnus_policy,
                                            max_param_norms, operation_policy,
                                            optimizer, param_count, pulse_step_count,
                                            pulse_time, save_file_name, save_iteration_step,
                                            save_path, system_step_multiplier)
    gstate.log_and_save_initial()

    # Transform the initial parameters to their optimizer
    # friendly form.
    initial_params = strip_params(gstate, initial_params)
    
    # Switch on the GRAPE implementation method.
    if gstate.grape_schroedinger_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        result = _grape_schroedinger_discrete_time(gstate, initial_params)
    else:
        result = _grape_schroedinger_discrete_memory(gstate, initial_params)

    return result


### HELPER METHODS ###

def _grape_schroedinger_discrete_time(gstate, initial_params):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.

    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    initial_params :: numpy.ndarray - the transformed initial_params

    Returns: 
    result :: qoc.GrapeResult - an object that tracks important information
        about the optimization
    """
    # Autograd does not allow multiple return values from
    # a differentiable function.
    # Scipy's minimization algorithms require us to provide
    # functions that they evaluate on their own schedule.
    # The best solution to track mutable objects, that I can think of,
    # is to use a reporter object.
    reporter = GrapeResult()
    gstate.optimizer.run((gstate, reporter), _gsd_compute_wrap,
                         gstate.iteration_count, initial_params,
                         _gsd_compute_jacobian_wrap)

    return reporter


def _gsd_compute(params, gstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Args:
    params :: numpy.ndarray - the control parameters
    gstate :: qoc.GrapeSchroedingerDiscreteState - static objects
    reporter :: qoc.Dummy - a reporter for mutable objects

    Returns:
    total_error :: numpy.ndarray - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    costs = gstate.costs
    dt = gstate.dt
    final_pulse_step = gstate.final_pulse_step
    final_time_step = gstate.final_time_step
    get_magnus_expansion = gstate.magnus
    get_magnus_param_indices = gstate.magnus_param_indices
    states = gstate.initial_states
    step_costs = gstate.step_costs
    system_step_multiplier = gstate.system_step_multiplier
    total_error = 0
    
    # Compute the total error for this evolution.
    for time_step in range(final_time_step + 1):
        pulse_step, _ = anp.divmod(time_step, system_step_multiplier)
        is_final_pulse_step = pulse_step == final_pulse_step
        is_final_time_step = time_step == final_time_step
        time = time_step * dt
        
        # Get the parameters to use for magnus expansion interpolation.
        # New parameters are used every pulse step.
        # If magnus_params includes only one parameter array,
        # wrap it in another dimension.
        magnus_param_indices = get_magnus_param_indices(dt, params,
                                                        pulse_step, time,
                                                        is_final_pulse_step)
        magnus_params = params[magnus_param_indices,]
        if magnus_params.ndim == 1:
            magnus_params = anp.expand_dims(magnus_params, axis=0)
            
        # Evolve the states.
        magnus = get_magnus_expansion(dt, magnus_params, pulse_step, time,
                                      is_final_pulse_step)
        unitary = expm(magnus)
        states = anp.matmul(unitary, states)
        
        # Compute cost every time step.
        if is_final_time_step:
            for i, cost in enumerate(costs):
                error = cost.cost(params, states, time_step)
                total_error = total_error + error
            #ENDFOR

            # Report information.
            reporter.last_states = states
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(params, states, time_step)
                total_error = total_error + error
            #ENDFOR

    #ENDFOR
    
    return anp.real(total_error)


# Wrapper to do intermediary work before passing control to _gsd_compute.
_gsd_compute_wrap = (lambda params, gstate, reporter:
                     _gsd_compute(slap_params(gstate, params),
                                  gstate, reporter))

# Value and jacobian of gsd_compute.
_gsd_compute_ans_jacobian = ans_jacobian(_gsd_compute, 0)


def _gsd_compute_jacobian_wrap(params, gstate, reporter):
    """
    Do intermediary work before passing control to _gsd_compute_ans_jacobian.
    Args:
    params :: numpy.ndarray - the control parameters in optimizer format
    gstate :: qoc.GrapeSchroedingerDiscreteState - static objects
    reporter :: qoc.Dummy - a reporter for mutable objects
    Returns:
    jac :: numpy.ndarray - the jacobian of the cost function with
        respect to params in optimizer format
    """
    params = slap_params(gstate, params)
    total_error, jacobian = _gsd_compute_ans_jacobian(params, gstate, reporter)
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy.
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if gstate.complex_params:
        jacobian = np.conjugate(jacobian)

    # Remove states from autograd box.
    if isinstance(reporter.last_states, Box):
        reporter.last_states = reporter.last_states._value
    
    # Report information.
    gstate.log_and_save(total_error, jacobian, reporter.iteration,
                        params, reporter.last_states)
    reporter.iteration += 1
    
    # Update last configuration.
    reporter.last_error = total_error
    reporter.last_grads = jacobian
    reporter.last_params = params

    # Update minimum configuration.
    if total_error < reporter.best_error:
        reporter.best_error = total_error
        reporter.best_grads = jacobian
        reporter.best_params = params
        reporter.best_states = reporter.last_states

    return strip_params(gstate, jacobian)


# TOOD: Implement me.
def _grape_schroedinger_discrete_memory(gstate, params):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use the memory efficient method.
    Args:
    gstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    params :: numpy.ndarray - the initial params

    Returns:
    result :: qoc.GrapeResult - the optimization result
    """
    reporter = GrapeResult()
    return reporter


### MODULE TESTS ###

_BIG = 100

def _test_grape_schroedinger_discrete():
    """
    Run end-to-end test on grape_schroedinger_discrete.
    """
    ## Test grape schroedinger discrete. ##

    # Check that the error, states, and gradients are what
    # we expect them to be for a hand-checked test case.
    # This implicitly tests _gsd_compute.
    hilbert_size = 2
    a = get_annihilation_operator(hilbert_size)
    a_dagger = get_creation_operator(hilbert_size)
    def hamiltonian(params, time):
        p0 = params[0]
        p0c = anp.conjugate(p0)
        return anp.array([[0, p0c],
                          [p0, 0]])
    state0 = np.array([[1], [0]])
    initial_states = np.stack((state0,))
    target0 = np.array([[1j], [0]])
    target_states = np.stack((target0,))
    forbid0_0 = np.array([[1j], [1]]) / np.sqrt(2)
    forbid_states0 = np.stack((forbid0_0,))
    forbid_states = np.stack((forbid_states0,))
    pulse_step_count = 2
    system_step_multiplier = 1
    param_count = 1
    iteration_count = 1
    total_step_count = pulse_step_count * system_step_multiplier
    pulse_time = total_step_count
    costs = (TargetInfidelity(target_states,),
             # TargetInfidelityTime(total_step_count, target_states,),
             # ParamValue()
             )
    optimizer = SGD()
    initial_params = np.array([[1], [1+1j]])
    max_param_norms = np.array([5])
    log_iteration_step = 0

    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time,
                                         initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         max_param_norms=max_param_norms,
                                         optimizer=optimizer,)
    
    m0 = -1j * hamiltonian(initial_params[0], 0)
    m1 = -1j * hamiltonian(initial_params[1], 0)
    expm0 = la.expm(m0)
    expm1 = la.expm(m1)
    # Hand computed gradients.
    s0 = np.matmul(expm0, state0)
    s0_0 = s0[0, 0]
    s0_1 = s0[1, 0]
    g0 = -expm1[0, 0] * s0_1 - np.conjugate(expm1[0, 1] * s0_1)
    g1 = -expm1[1, 0] * s0_0 - np.conjugate(expm1[1, 1] * s0_1)
    expected_last_error = 0
    expected_last_grads = (g0, g1)
    expected_last_states = 0

    print("result.last_grads:\n{}"
          "".format(result.last_grads))
    print("expected_last_grads:\n{}"
          "".format(expected_last_grads))

    # assert(np.allclose(result.last_error, expected_last_error))
    # assert(np.allclose(result.last_grads, expected_last_grads))
    # assert(np.allclose(result.last_states, expected_last_states))
    exit(0)
    
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
    costs = (TargetInfidelity(target_states),)
    param_count = 1
    pulse_time = np.divide(np.pi, 2)
    pulse_step_count = 10
    system_step_multiplier = 1000
    iteration_count = 1
    initial_params = np.ones((pulse_step_count, param_count), dtype=np.complex128)
    magnus_policy = MagnusPolicy.M6
    log_iteration_step = 0
    save_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         save_iteration_step=save_iteration_step,
                                         system_step_multiplier=system_step_multiplier)
    assert(np.allclose(result.last_error, 0))
    assert(np.allclose(result.last_states, target_states))

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
    assert(np.allclose(result.last_grads, np.zeros_like(result.last_grads)))
    assert(np.allclose(initial_params, result.last_params))
    assert(np.allclose(initial_states, result.last_states))

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
    initial_params, max_param_norms = initialize_params(None, None,
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
                                         max_param_norms=max_param_norms)
    assert(not (np.equal(result.last_grads, np.zeros_like(result.last_grads)).any()))
    assert(not (np.equal(result.last_params, initial_params).any()))

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
                                         max_param_norms=max_param_norms)
    assert(np.allclose(result.last_grads.imag, np.zeros_like(result.last_grads.imag)))

    # TOOD: Rework this test when parameter clipping gets reworked.
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
    initial_params, max_param_norms = initialize_params(None, None,
                                                              pulse_time, pulse_step_count,
                                                              param_count)
    max_param_norms = np.repeat(0.8 + 0.8j, param_count)
    costs = [ForbidStates(forbidden_states, pulse_step_count)]
    iteration_count = 100
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, param_count, pulse_step_count,
                                         pulse_time, initial_params=initial_params,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_param_norms=max_param_norms)
    
    # for i in range(result.params.shape[1]):
    #     assert(np.less_equal(np.abs(result.params[:,i]),
    #                          np.abs(max_param_norms[i])).all())


def _test():
    """
    Run tests on the module.
    """
    # _test_grads()
    _test_grape_schroedinger_discrete()


if __name__ == "__main__":
    _test()
