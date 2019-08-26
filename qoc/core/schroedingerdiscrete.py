"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

import os

from autograd import jacobian
import autograd.numpy as anp
from autograd.extend import Box
import numpy as np
import scipy.linalg as la

from qoc.core.common import (gen_controls_cos, initialize_controls,
                             slap_controls, strip_controls,)
from qoc.models import (MagnusPolicy, OperationPolicy, GrapeSchroedingerDiscreteState,
                        PerformancePolicy, GrapeResult,
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

def evolve_schroedinger_discrete(control_step_count, evolution_time,)

def grape_schroedinger_discrete(control_count, control_step_count,
                                costs, evolution_time, hamiltonian, initial_states,
                                initial_controls=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                iteration_count=1000, 
                                log_iteration_step=10,
                                magnus_policy=MagnusPolicy.M2,
                                max_control_norms=None,
                                operation_policy=OperationPolicy.CPU,
                                optimizer=Adam(),
                                performance_policy=PerformancePolicy.TIME,
                                save_file_path=None, save_iteration_step=0,
                                system_step_multiplier=1,):
    """
    a method to optimize the evolution of a set of states under the
    schroedinger equation for time-discrete control parameters

    Args:
    control_count :: int - the number of control parameters required to be
        passed to the hamiltonian at each time step
    control_step_count :: int - the number of time steps in the evolution
        where control parameters are fit
    evolution_time :: float - the duration of system's evolution
    costs :: iterable(qoc.models.cost.Cost) - the cost functions to guide optimization
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                    -> hamiltonian :: ndarray (hilbert_size x hilbert_size)
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the control parameters
        and evolution time
    initial_controls :: ndarray (control_step_count x control_count)
        - the values to use for the controls for the first iteration,
    initial_states :: ndarray (state_count x hilbert_size x 1)
        - a list of the states to evolve
        A column vector is specified as np.array([[0], [1], [2]]).
        A column vector is NOT a row vector np.array([[0, 1, 2]]).
    interpolation_policy :: qoc.InterpolationPolicy - how to interpolate
        controls where they are not defined
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: int - how often to write to stdout,
        set to 0 to disable logging
    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy
        - the method to use for the magnus expansion
    max_control_norms :: ndarray (control_count) - These are the absolute values at
        which to clip the controls if their norm exceeds these values.
        The default maximum norms will be 1 if not specified. 
    operation_policy :: qoc.models.operationpolicy.OperationPolicy
        - this policy encapsulates the array class that is used and how operations
        should be performed on those arrays,
        e.g. CPU, GPU, CPU-sparse, GPU-spares, etc.
    optimizer :: qoc.models.optimizer.Optimizer
        - an instance of an optimizer to perform gradient-based optimization
    performance_policy :: qoc.PerformancePolicy - minimize the usage of this
        resource
    save_file_path :: str - the full path to an h5 file where
        information should be saved
    save_iteration_step :: int - how often to write to the save file,
        set 0 to disable saving
    system_step_multiplier :: int - this factor will be used to determine how
        many steps inbetween each pulse step the system should evolve,
        control parameters will be interpolated at these steps

    Returns:
    result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
        - useful information about the optimization
    """
    # Initialize controls.
    initial_controls, max_control_norms = initialize_controls(initial_controls,
                                                              max_control_norms,
                                                              evolution_time,
                                                              control_step_count,
                                                              control_count)
    
    # Construct the grape state.
    hilbert_size = initial_states[0].shape[0]
    pstate = GrapeSchroedingerDiscreteState(costs, performance_policy,
                                            hamiltonian, hilbert_size,
                                            initial_controls,
                                            initial_states,
                                            interpolation_policy, iteration_count,
                                            log_iteration_step, magnus_policy,
                                            max_control_norms, operation_policy,
                                            optimizer, control_count, control_step_count,
                                            evolution_time, save_file_name, save_iteration_step,
                                            save_path, system_step_multiplier)
    pstate.log_and_save_initial()

    # Transform the initial parameters to their optimizer
    # friendly form.
    initial_controls = strip_controls(pstate, initial_controls)
    
    # Switch on the GRAPE implementation method.
    if pstate.performance_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        result = _grape_schroedinger_discrete_time(pstate, initial_controls)
    else:
        result = _grape_schroedinger_discrete_memory(pstate, initial_controls)

    return result


### HELPER METHODS ###

def _grape_schroedinger_discrete_time(pstate, initial_controls):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.

    Args:
    pstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    initial_controls :: ndarray - the transformed initial_controls

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
    pstate.optimizer.run((pstate, reporter), _evaluate_schroedinger_discrete_wrap,
                         pstate.iteration_count, initial_controls,
                         _evaluate_schroedinger_discrete_jacobian_wrap)

    return reporter


def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState - static objects
    reporter :: qoc.Dummy - a reporter for mutable objects

    Returns:
    total_error :: ndarray - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    dt = pstate.dt
    final_pulse_step = pstate.final_pulse_step
    final_time_step = pstate.final_time_step
    get_magnus_expansion = pstate.magnus
    get_magnus_param_indices = pstate.magnus_param_indices
    states = pstate.initial_states
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0
    
    # Compute the total error for this evolution.
    for time_step in range(final_time_step + 1):
        pulse_step, _ = anp.divmod(time_step, system_step_multiplier)
        is_final_pulse_step = pulse_step == final_pulse_step
        is_final_time_step = time_step == final_time_step
        time = time_step * dt
        
        # Get the parameters to use for magnus expansion interpolation.
        # New parameters are used every pulse step.
        # If magnus_controls includes only one parameter array,
        # wrap it in another dimension.
        magnus_param_indices = get_magnus_param_indices(dt, controls,
                                                        pulse_step, time,
                                                        is_final_pulse_step)
        magnus_controls = controls[magnus_param_indices,]
        if magnus_controls.ndim == 1:
            magnus_controls = anp.expand_dims(magnus_controls, axis=0)
            
        # Evolve the states.
        magnus = get_magnus_expansion(dt, magnus_controls, pulse_step, time,
                                      is_final_pulse_step)
        unitary = expm(magnus)
        states = anp.matmul(unitary, states)
        
        # Compute cost every time step.
        if is_final_time_step:
            for i, cost in enumerate(costs):
                error = cost.cost(controls, states, time_step)
                total_error = total_error + error
            #ENDFOR

            # Report information.
            reporter.last_states = states
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(controls, states, time_step)
                total_error = total_error + error
            #ENDFOR

    #ENDFOR
    
    return anp.real(total_error)


# Wrapper to do intermediary work before passing control to _evaluate_schroedinger_discrete.
_evaluate_schroedinger_discrete_wrap = (lambda controls, pstate, reporter:
                     _evaluate_schroedinger_discrete(slap_controls(pstate, controls),
                                  pstate, reporter))

# Value and jacobian of gsd_compute.
_evaluate_schroedinger_discrete_ans_jacobian = ans_jacobian(_evaluate_schroedinger_discrete, 0)


def _evaluate_schroedinger_discrete_jacobian_wrap(controls, pstate, reporter):
    """
    Do intermediary work before passing control to _evaluate_schroedinger_discrete_ans_jacobian.
    Args:
    controls :: ndarray - the control parameters in optimizer format
    pstate :: qoc.GrapeSchroedingerDiscreteState - static objects
    reporter :: qoc.Dummy - a reporter for mutable objects
    Returns:
    jac :: ndarray - the jacobian of the cost function with
        respect to controls in optimizer format
    """
    controls = slap_controls(pstate, controls)
    total_error, jacobian = _evaluate_schroedinger_discrete_ans_jacobian(controls, pstate, reporter)
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        jacobian = np.conjugate(jacobian)

    # Remove states from autograd box.
    if isinstance(reporter.last_states, Box):
        reporter.last_states = reporter.last_states._value
    
    # Report information.
    pstate.log_and_save(total_error, jacobian, reporter.iteration,
                        controls, reporter.last_states)
    reporter.iteration += 1
    
    # Update last configuration.
    reporter.last_error = total_error
    reporter.last_grads = jacobian
    reporter.last_controls = controls

    # Update minimum configuration.
    if total_error < reporter.best_error:
        reporter.best_error = total_error
        reporter.best_grads = jacobian
        reporter.best_controls = controls
        reporter.best_states = reporter.last_states

    return strip_controls(pstate, jacobian)


# TOOD: Implement me.
def _grape_schroedinger_discrete_memory(pstate, controls):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use the memory efficient method.
    Args:
    pstate :: qoc.GrapeSchroedingerDiscreteState - information required to
         perform the optimization
    controls :: ndarray - the initial controls

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
    # This implicitly tests _evaluate_schroedinger_discrete.
    hilbert_size = 2
    a = get_annihilation_operator(hilbert_size)
    a_dagger = get_creation_operator(hilbert_size)
    def hamiltonian(controls, time):
        p0 = controls[0]
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
    control_step_count = 2
    system_step_multiplier = 1
    control_count = 1
    iteration_count = 1
    total_step_count = control_step_count * system_step_multiplier
    evolution_time = total_step_count
    costs = (TargetInfidelity(target_states,),
             # TargetInfidelityTime(total_step_count, target_states,),
             # ParamValue()
             )
    optimizer = SGD()
    initial_controls = np.array([[1], [1+1j]])
    max_control_norms = np.array([5])
    log_iteration_step = 0

    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time,
                                         initial_controls=initial_controls,
                                         log_iteration_step=log_iteration_step,
                                         max_control_norms=max_control_norms,
                                         optimizer=optimizer,)
    
    m0 = -1j * hamiltonian(initial_controls[0], 0)
    m1 = -1j * hamiltonian(initial_controls[1], 0)
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

    assert(np.allclose(result.last_error, expected_last_error))
    assert(np.allclose(result.last_grads, expected_last_grads))
    assert(np.allclose(result.last_states, expected_last_states))
    
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
    hamiltonian = lambda controls, t: controls[0] * _hamiltonian
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    costs = (TargetInfidelity(target_states),)
    control_count = 1
    evolution_time = np.divide(np.pi, 2)
    control_step_count = 10
    system_step_multiplier = 1000
    iteration_count = 1
    initial_controls = np.ones((control_step_count, control_count), dtype=np.complex128)
    magnus_policy = MagnusPolicy.M6
    log_iteration_step = 0
    save_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time, initial_controls=initial_controls,
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
    hamiltonian = lambda controls, t: _hamiltonian
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(identity_matrix)
    costs = [TargetInfidelity(target_states)]
    control_count = 1
    evolution_time = 10
    control_step_count = 10
    system_step_multiplier = 1
    iteration_count = 10
    initial_controls = np.ones((control_step_count, control_count), dtype=np.complex128)
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time, initial_controls=initial_controls,
                                         log_iteration_step=log_iteration_step,
                                         system_step_multiplier=system_step_multiplier)
    assert(np.allclose(result.last_grads, np.zeros_like(result.last_grads)))
    assert(np.allclose(initial_controls, result.last_controls))
    assert(np.allclose(initial_states, result.last_states))

    # Some nontrivial gradients should appear at each time step
    # if we evolve a nontrivial hamiltonian and penalize
    # a state against itself at each time step. Note that
    # the hamiltonian is not hermitian here.
    hilbert_size = 4
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda controls, t: (controls[0] * _hamiltonian)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    control_count = 1
    evolution_time = 10
    control_step_count = 10
    initial_controls, max_control_norms = initialize_controls(None, None,
                                                              evolution_time, control_step_count,
                                                              control_count)
    costs = [ForbidStates(forbidden_states, control_step_count)]
    iteration_count = 100
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time, initial_controls=initial_controls,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_control_norms=max_control_norms)
    assert(not (np.equal(result.last_grads, np.zeros_like(result.last_grads)).any()))
    assert(not (np.equal(result.last_controls, initial_controls).any()))

    # If we use complex parameters on a hermitian hamiltonian,
    # the complex parameters should have no contribution to the
    # hamiltonian.
    _hamiltonian_dagger = conjugate_transpose(_hamiltonian)
    hamiltonian = lambda controls, t: (controls[0] * _hamiltonian
                                     + (anp.conjugate(controls[0])
                                        * _hamiltonian_dagger))
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time, initial_controls=initial_controls,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_control_norms=max_control_norms)
    assert(np.allclose(result.last_grads.imag, np.zeros_like(result.last_grads.imag)))

    # TOOD: Rework this test when parameter clipping gets reworked.
    # Parameters should be clipped if they grow too large.
    # You can log result.parameters from the test above
    # that uses the same hamiltonian to see that
    # each of result.controls is greater than 0.8 + 0.8j.
    hilbert_size = 4
    _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda controls, t: (controls[0] * _hamiltonian)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    control_count = 1
    evolution_time = 10
    control_step_count = 10
    initial_controls, max_control_norms = initialize_controls(None, None,
                                                              evolution_time, control_step_count,
                                                              control_count)
    max_control_norms = np.repeat(0.8 + 0.8j, control_count)
    costs = [ForbidStates(forbidden_states, control_step_count)]
    iteration_count = 100
    magnus_policy = MagnusPolicy.M2
    log_iteration_step = 0
    result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
                                         iteration_count, control_count, control_step_count,
                                         evolution_time, initial_controls=initial_controls,
                                         log_iteration_step=log_iteration_step,
                                         magnus_policy=magnus_policy,
                                         max_control_norms=max_control_norms)
    
    # for i in range(result.controls.shape[1]):
    #     assert(np.less_equal(np.abs(result.controls[:,i]),
    #                          np.abs(max_control_norms[i])).all())


def _test():
    """
    Run tests on the module.
    """
    # _test_grads()
    _test_grape_schroedinger_discrete()


if __name__ == "__main__":
    _test()
