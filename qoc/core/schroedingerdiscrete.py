"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

import os

from autograd import jacobian
import autograd.numpy as anp
from autograd.extend import Box
import numpy as np

from qoc.core.common import (initialize_controls,
                             slap_controls, strip_controls,)
from qoc.models import (MagnusPolicy, OperationPolicy,
                        PerformancePolicy,
                        InterpolationPolicy, Dummy,
                        magnus_m2, magnus_m4, magnus_m6,
                        EvolveSchroedingerDiscreteState,
                        EvolveSchroedingerResult,
                        interpolate_linear,)
from qoc.standard import (ans_jacobian, Adam, SGD, ForbidStates, TargetInfidelity,
                          ParamValue, expm,
                          PAULI_X, PAULI_Y, PAULI_Z, conjugate_transpose,
                          matrix_to_column_vector_list, matmuls,
                          complex_to_real_imag_flat,
                          real_imag_to_complex_flat,
                          get_annihilation_operator,
                          get_creation_operator)

### MAIN METHODS ###

def evolve_schroedinger_discrete(control_step_count, evolution_time,
                                 hamiltonian, initial_states,
                                 controls=None, costs=list(),
                                 interpolation_policy=InterpolationPolicy.LINEAR,
                                 magnus_policy=MagnusPolicy.M2,
                                 operation_policy=OperationPolicy.CPU,
                                 system_step_multiplier=1,):
    """
    Evolve a set of state vectors under the schroedinger equation
    and compute the optimization error.

    Args:
    control_step_count :: int
    controls :: ndarray (control_step_count x control_count)
    costs :: iterable(qoc.models.cost.Cost)
    evolution_time :: float
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                   -> hamiltonian :: ndarray (hilbert_size x hilbert_size)
    initial_states :: ndarray (state_count x hilbert_size x 1)
    interpolation_policy :: qoc.models.interpolationpolicy.InterpolationPolicy
    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy
    operation_policy :: qoc.models.operationpolicy.OperationPolicy
    system_step_multiplier :: int

    Returns:
    result
    """
    pstate = EvolveSchroedingerDiscreteState(control_step_count,
                                             costs, evolution_time,
                                             hamiltonian, initial_states,
                                             interpolation_policy,
                                             magnus_policy, operation_policy,
                                             system_step_multiplier,)
    result = EvolveSchroedingerResult()
    _ = _evaluate_schroedinger_discrete(controls, pstate, result)

    return result


def grape_schroedinger_discrete(control_count, control_step_count,
                                costs, evolution_time, hamiltonian, initial_states,
                                complex_controls=False,
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
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Args:
    complex_controls
    control_count
    control_step_count
    costs
    evolution_time
    hamiltonian
    initial_states
    interpolation_policy
    iteration_count
    log_iteration_step
    magnus_policy
    max_control_norms
    operation_policy
    optimizer
    performance_policy
    save_file_path
    save_iteration_step
    system_step_multiplier

    Returns:
    result
    """
    # Initialize the controls.
    initial_controls, max_control_norms = initialize_controls(complex_controls,
                                                              control_count,
                                                              control_step_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms,)
    # Construct the program state.
    pstate = GrapeSchroedingerDiscreteState(complex_controls, control_count,
                                            control_step_count, costs,
                                            evolution_time, hamiltonian,
                                            initial_controls,
                                            initial_states,
                                            interpolation_policy,
                                            iteration_count,
                                            log_iteration_step, magnus_policy,
                                            max_control_nroms, operation_policy,
                                            optimizer, performance_policy,
                                            save_file_path, save_iteration_step,
                                            system_step_multiplier,)
    pstate.log_and_save_initial()

    # Switch on the GRAPE implementation method.
    if performance_policy == GrapeSchroedingerPolicy.TIME_EFFICIENT:
        result = _grape_schroedinger_discrete_time(pstate)
    else:
        result = _grape_schroedinger_discrete_memory(pstate)

    return result


### HELPER METHODS ###

def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    total_error :: ndarray - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    dt = pstate.dt
    hamiltonian = pstate.hamiltonian
    final_control_step = pstate.final_control_step
    final_system_step = pstate.final_system_step
    interpolation_policy = pstate.interpolation_policy
    magnus_policy = pstate.magnus_policy
    operation_policy = pstate.operation_policy
    states = pstate.initial_states
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0
    
    # Compute the total error for this evolution.
    for system_step in range(final_system_step + 1):
        control_step, _ = anp.divmod(system_step, system_step_multiplier)
        is_final_control_step = control_step == final_control_step
        is_final_system_step = system_step == final_system_step
        time = system_step * dt

        # Evolve the states.
        states = _evolve_step_schroedinger_discrete(dt, hamiltonian, states,
                                                    time,
                                                    control_sentinel=is_final_control_step,
                                                    control_step=control_step,
                                                    controls=controls,
                                                    interpolation_policy=interpolation_policy,
                                                    magnus_policy=magnus_policy,
                                                    operation_policy=operation_policy,)
        
        # Compute cost every time step.
        if is_final_system_step:
            for i, cost in enumerate(costs):
                error = cost.cost(controls, states, system_step)
                total_error = total_error + error
            #ENDFOR
            reporter.final_states = states
            reporter.total_error = total_error
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(controls, states, system_step)
                total_error = total_error + error
            #ENDFOR
    #ENDFOR
    
    return total_error


_M4_T1 = np.divide(1, 2) - np.divide(np.sqrt(3), 6)
_M4_T2 = np.divide(1, 2) + np.divide(np.sqrt(3), 6)
_M6_T1 = np.divide(1, 2) - np.divide(np.sqrt(15), 10)
_M6_T2 = np.divide(1, 2)
_M6_T3 = np.divide(1, 2) + np.divide(np.sqrt(15), 10)

def _evolve_step_schroedinger_discrete(dt, hamiltonian, states, time,
                                       control_sentinel=False,
                                       control_step=0,
                                       controls=None,
                                       interpolation_policy=InterpolationPolicy.LINEAR,
                                       magnus_policy=MagnusPolicy.M2,
                                       operation_policy=OperationPolicy.CPU,):
    """
    Use the exponential series method via magnus expansion to evolve the state vectors
    to the next time step under the schroedinger equation for time-discrete controls.
    Magnus expansions are implemented using the methods described in
    https://arxiv.org/abs/1709.06483.

    Args:
    control_sentinel
    control_step
    controls
    dt
    hamiltonian
    interpolation_policy
    magnus_policy
    operation_policy
    states
    time
    
    Returns:
    states
    """
    controls_exist = not (controls is None)
    
    if magnus_policy == MagnusPolicy.M2:
        if controls_exist:
            c1 = controls[control_step]
        else:
            c1 = None
        a1 = -1j * hamiltonian(c1, time)
        magnus = magnus_m2(a1, dt, operation_policy=operation_policy)
    elif magnus_policy == MagnusPolicy.M4:
        t1 = time + dt * _M4_T1
        t2 = time + dt * _M4_T2
        if controls_exist:
            if control_sentinel:
                controls_left = controls[control_step - 1]
                controls_right = controls[control_step]
                time_left = time - dt
                time_right = time
            else:
                controls_left = controls[control_step]
                controls_right = controls[control_step + 1]
                time_left = time
                time_right = time + dt
            if interpolation_policy == InterpolationPolicy.LINEAR:
                c1 = interpolate_linear(time_left, time_right, t1,
                                        controls_left, controls_right)
                c2 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right)
            else:
                raise ValueError("The interpolation policy {} is not implemented"
                                 "for this method.".format(interpolation_policy))
        else:
            c1 = c2 = None
        a1 = -1j * hamiltonian(c1, t1)
        a2 = -1j * hamiltonian(c2, t2)
        magnus = magnus_m4 (a1, a2, dt)
    elif magnus_policy == MagnusPolicy.M6:
        t1 = time + dt * _M6_T1
        t2 = time + dt * _M6_T2
        t3 = time + dt * _M6_T3
        if controls_exist:
            if control_sentinel:
                controls_left = controls[control_step - 1]
                controls_right = controls[control_step]
                time_left = time - dt
                time_right = time
            else:
                controls_left = controls[control_step]
                controls_right = controls[control_step + 1]
                time_left = time
                time_right = time + dt
            if interpolation_policy == InterpolationPolicy.LINEAR:
                c1 = interpolate_linear(time_left, time_right, t1,
                                        controls_left, controls_right,
                                        operation_policy=operation_policy)
                c2 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right,
                                        operation_policy=operation_policy)
                c3 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right,
                                        operation_policy=operation_policy)
            else:
                raise ValueError("The interpolation policy {} is not implemented"
                                 "for this method.".format(interpolation_policy))
        else:
            c1 = c2 = c3 = None
        a1 = -1j * hamiltonian(c1, t1)
        a2 = -1j * hamiltonian(c2, t2)
        a3 = -1j * hamiltonian(c3, t3)
        magnus = magnus_m6(a1, a2, a3, dt,
                           operation_policy=operation_policy)
    #ENDIF

    step_unitary = expm(magnus, operation_policy=operation_policy)
    states = matmuls(step_unitary, states,
                     operation_policy=operation_policy)

    return states


def _grape_schroedinger_discrete_time(pstate):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.

    Args:
    pstate :: qoc.models.GrapeSchroedingerDiscreteState - information required to
         perform the optimization

    Returns: 
    result :: qoc.models.GrapeResult - an object that tracks important information
        about the optimization
    """
    # TODO: multiple reporters?
    # Autograd does not allow multiple return values from
    # a differentiable function.
    # Scipy's minimization algorithms require us to provide
    # functions that they evaluate on their own schedule.
    # The best solution to track mutable objects, that I can think of,
    # is to use a reporter object.
    reporter = GrapeResult()
    initial_controls = strip_controls(pstate.complex_controls)
    pstate.optimizer.run((pstate, reporter), _evaluate_schroedinger_discrete_wrap,
                         pstate.iteration_count, initial_controls,
                         _evaluate_schroedinger_discrete_jacobian_wrap)

    return reporter


# # Wrapper to do intermediary work before passing control to _evaluate_schroedinger_discrete.
# _evaluate_schroedinger_discrete_wrap = (lambda controls, pstate, reporter:
#                      _evaluate_schroedinger_discrete(slap_controls(pstate, controls),
#                                   pstate, reporter))

# # Value and jacobian of gsd_compute.
# _evaluate_schroedinger_discrete_ans_jacobian = ans_jacobian(_evaluate_schroedinger_discrete, 0)


# def _evaluate_schroedinger_discrete_jacobian_wrap(controls, pstate, reporter):
#     """
#     Do intermediary work before passing control to _evaluate_schroedinger_discrete_ans_jacobian.
#     Args:
#     controls :: ndarray - the control parameters in optimizer format
#     pstate :: qoc.GrapeSchroedingerDiscreteState - static objects
#     reporter :: qoc.Dummy - a reporter for mutable objects
#     Returns:
#     jac :: ndarray - the jacobian of the cost function with
#         respect to controls in optimizer format
#     """
#     controls = slap_controls(pstate, controls)
#     total_error, jacobian = _evaluate_schroedinger_discrete_ans_jacobian(controls, pstate, reporter)
#     # Autograd defines the derivative of a function of complex inputs as
#     # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
#     # For optimization, we care about df_dz = du_dx + i * du_dy.
#     if pstate.complex_controls:
#         jacobian = np.conjugate(jacobian)

#     # Remove states from autograd box.
#     if isinstance(reporter.last_states, Box):
#         reporter.last_states = reporter.last_states._value
    
#     # Report information.
#     pstate.log_and_save(total_error, jacobian, reporter.iteration,
#                         controls, reporter.last_states)
#     reporter.iteration += 1
    
#     # Update last configuration.
#     reporter.last_error = total_error
#     reporter.last_grads = jacobian
#     reporter.last_controls = controls

#     # Update minimum configuration.
#     if total_error < reporter.best_error:
#         reporter.best_error = total_error
#         reporter.best_grads = jacobian
#         reporter.best_controls = controls
#         reporter.best_states = reporter.last_states

#     return strip_controls(pstate, jacobian)


# # TOOD: Implement me.
# def _grape_schroedinger_discrete_memory(pstate, controls):
#     """
#     Perform GRAPE for the schroedinger equation with time discrete parameters.
#     Use the memory efficient method.
#     Args:
#     pstate :: qoc.GrapeSchroedingerDiscreteState - information required to
#          perform the optimization
#     controls :: ndarray - the initial controls

#     Returns:
#     result :: qoc.GrapeResult - the optimization result
#     """
#     reporter = GrapeResult()
#     return reporter


### MODULE TESTS ###

_BIG = 10
_MAGNUS_POLICIES = (MagnusPolicy.M2, MagnusPolicy.M4, MagnusPolicy.M6,)

def _random_complex_matrix(matrix_size):
    """
    Generate a random, square, complex matrix of size `matrix_size`.
    """
    return (np.random.rand(matrix_size, matrix_size)
            + 1j * np.random.rand(matrix_size, matrix_size))


def _random_hermitian_matrix(matrix_size):
    """
    Generate a random, square, hermitian matrix of size `matrix_size`.
    """
    matrix = _random_complex_matrix(matrix_size)
    return np.divide(matrix + conjugate_transpose(matrix), 2)


def _test_evolve_schroedinger_discrete():
    """
    Run end-to-end test on the evolve_schroedinger_discrete
    function.
    """
    from qutip import mesolve, Qobj, Options

    # Test that evolving states under a hamiltonian yields
    # a known result. Use e.q. 109 of 
    # https://arxiv.org/abs/1904.06560.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array(((1,   0,   0, 0),
                              (0,   0, -1j, 0),
                              (0, -1j,   0, 0),
                              (0,   0,   0, 1)))
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
                                     + np.kron(PAULI_Y, PAULI_Y))
    hamiltonian = lambda controls, time: hamiltonian_matrix
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    evolution_time = np.divide(np.pi, 2)
    control_step_count = int(1e3)
    for magnus_policy in _MAGNUS_POLICIES:
        result = evolve_schroedinger_discrete(control_step_count, evolution_time,
                                              hamiltonian, initial_states,
                                              magnus_policy=magnus_policy)
        final_states = result.final_states
        assert(np.allclose(final_states, target_states))
    #ENDFOR
    # Note that qutip only gets the same result within 1e-5 error.
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = list()
    e_ops = list()
    hamiltonian_qutip = Qobj(hamiltonian_matrix)
    options = Options(nsteps=control_step_count)
    for i, initial_state in enumerate(initial_states):
        initial_state_qutip = Qobj(initial_state)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,
                         options=options)
        final_state = result.states[-1].full()
        target_state = target_states[i]
        assert(np.allclose(final_state, target_state, atol=1e-5))
    #ENDFOR

    # Test that evolving states under a random hamiltonian yields
    # a result similar to qutip.
    hilbert_size = 4
    initial_state = np.divide(np.ones((hilbert_size, 1)),
                              np.sqrt(hilbert_size))
    initial_states = np.stack((initial_state,))
    initial_state_qutip = Qobj(initial_state)
    control_step_count = int(1e3)
    evolution_time = 1
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = e_ops = list()
    options = Options(nsteps=control_step_count)
    for _ in range(_BIG):
        hamiltonian_matrix = _random_hermitian_matrix(hilbert_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        hamiltonian_qutip = Qobj(hamiltonian_matrix)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,
                         options=options)
        final_state_qutip = result.states[-1].full()
        for magnus_policy in _MAGNUS_POLICIES:
            result = evolve_schroedinger_discrete(control_step_count,
                                                  evolution_time, hamiltonian,
                                                  initial_states,
                                                  magnus_policy=magnus_policy)
            final_state = result.final_states[0]
            assert(np.allclose(final_state, final_state_qutip, atol=1e-4))
        #ENDFOR
    #ENDFOR
        

def _test_grape_schroedinger_discrete():
    """
    Run end-to-end test on the grape_schroedinger_discrete function.
    """
    # TODO: implement me
    pass
    # # TOOD: Rework this test when parameter clipping gets reworked.
    # # Parameters should be clipped if they grow too large.
    # # You can log result.parameters from the test above
    # # that uses the same hamiltonian to see that
    # # each of result.controls is greater than 0.8 + 0.8j.
    # hilbert_size = 4
    # _hamiltonian = np.divide(1, 2) * (np.kron(PAULI_X, PAULI_X)
    #                                  + np.kron(PAULI_Y, PAULI_Y))
    # hamiltonian = lambda controls, t: (controls[0] * _hamiltonian)
    # initial_states = np.array([[[0], [1], [0], [0]]])
    # forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    # control_count = 1
    # evolution_time = 10
    # control_step_count = 10
    # initial_controls, max_control_norms = initialize_controls(None, None,
    #                                                           evolution_time, control_step_count,
    #                                                           control_count)
    # max_control_norms = np.repeat(0.8 + 0.8j, control_count)
    # costs = [ForbidStates(forbidden_states, control_step_count)]
    # iteration_count = 100
    # magnus_policy = MagnusPolicy.M2
    # log_iteration_step = 0
    # result = grape_schroedinger_discrete(costs, hamiltonian, initial_states,
    #                                      iteration_count, control_count, control_step_count,
    #                                      evolution_time, initial_controls=initial_controls,
    #                                      log_iteration_step=log_iteration_step,
    #                                      magnus_policy=magnus_policy,
    #                                      max_control_norms=max_control_norms)
    
    # for i in range(result.controls.shape[1]):
    #     assert(np.less_equal(np.abs(result.controls[:,i]),
    #                          np.abs(max_control_norms[i])).all())


def _test():
    """
    Run tests on the module.
    """
    _test_evolve_schroedinger_discrete()
    _test_grape_schroedinger_discrete()


if __name__ == "__main__":
    _test()
