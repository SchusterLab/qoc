"""
linbladdiscrete.py - This module defines methods to
evolve a set of density matrices under the
lindblad master equation using time-discrete
control parameters.
"""

from autograd.extend import Box
import numpy as np

from qoc.core.common import (clip_control_norms,
                             initialize_controls,
                             slap_controls, strip_controls,)
from qoc.models import (Dummy,
                        EvolveLindbladDiscreteState,
                        EvolveLindbladResult,
                        get_linear_interpolator,
                        integrate_rkdp5,
                        interpolate_linear,
                        InterpolationPolicy,
                        OperationPolicy,
                        get_lindbladian,
                        GrapeLindbladDiscreteState,
                        GrapeLindbladResult,)
from qoc.standard import (Adam, ans_jacobian, commutator, conjugate,
                          conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_lindblad_discrete(control_step_count, evolution_time,
                             initial_densities,
                             controls=None, costs=list(),
                             hamiltonian=None,
                             interpolation_policy=InterpolationPolicy.LINEAR,
                             lindblad_data=None,
                             operation_policy=OperationPolicy.CPU,
                             system_step_multiplier=1,):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute the optimization error.

    Args:
    control_step_count :: int - the number of time intervals in the 
        evolution time in which the controls are spaced, or, if no controls
        are specified, the number of time steps in which the evolution time
        interval should be broken up
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution
    costs :: iterable(qoc.models.Cost) - the cost functions to guide
        optimization
    evolution_time :: float - the length of time the system should evolve for
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    initial_densities :: ndarray (density_count x hilbert_size x hilbert_size)
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_data :: (time :: float) -> (tuple(operators :: ndarray, dissipators :: ndarray))
        - a function to generate the dissipation constants and lindblad operators for a given time,
          an array of operators should be returned even if there 
          are zero or one dissipator and operator pairs
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    system_step_multiplier :: int - the multiple of control_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps

    Returns:
    result :: qoc.models.EvolveLindbladResult - information
        about the evolution
    """
    pstate = EvolveLindbladDiscreteState(control_step_count,
                                         costs,
                                         evolution_time,
                                         hamiltonian, initial_densities,
                                         interpolation_policy,
                                         lindblad_data,
                                         operation_policy,
                                         system_step_multiplier,)
    result = EvolveLindbladResult()
    _ = _evaluate_lindblad_discrete(controls, pstate, result)

    return result


def grape_lindblad_discrete(control_count, control_step_count,
                            costs, evolution_time, initial_densities,
                            complex_controls=False,
                            hamiltonian=None,
                            initial_controls=None,
                            interpolation_policy=InterpolationPolicy.LINEAR,
                            iteration_count=1000,
                            lindblad_data=None,
                            log_iteration_step=10,
                            max_control_norms=None,
                            minimum_error=0,
                            operation_policy=OperationPolicy.CPU,
                            optimizer=Adam(),
                            save_file_path=None, save_iteration_step=0,
                            system_step_multiplier=1,):
    """
    This method optimizes the evolution of a set of states under the lindblad
    equation for time-discrete control parameters.

    Args:
    complex_controls
    control_count
    control_step_count
    costs
    evolution_time
    hamiltonian
    initial_controls
    initial_densities
    interpolation_policy
    iteration_count
    lindblad_data
    log_iteration_step
    max_control_norms
    minimum_error
    operation_policy
    optimizer
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
    pstate = GrapeLindbladDiscreteState(complex_controls, control_count,
                                        control_step_count, costs,
                                        evolution_time, hamiltonian,
                                        initial_controls,
                                        initial_densities,
                                        interpolation_policy,
                                        iteration_count,
                                        lindblad_data,
                                        log_iteration_step,
                                        max_control_norms, minimum_error,
                                        operation_policy,
                                        optimizer,
                                        save_file_path, save_iteration_step,
                                        system_step_multiplier,)
    pstate.log_and_save_initial()

    # Autograd does not allow multiple return values from
    # a differentiable function.
    # Scipy's minimization algorithms require us to provide
    # functions that they evaluate on their own schedule.
    # The best solution to track mutable objects, that I can think of,
    # is to use a reporter object.
    reporter = Dummy()
    reporter.iteration = 0
    result = GrapeLindbladResult()
    
    # Convert the controls from cost function format to optimizer format.
    initial_controls = strip_controls(pstate.complex_controls, pstate.initial_controls)
    
    # Run the optimization.
    pstate.optimizer.run(_eld_wrap, pstate.iteration_count, initial_controls,
                         _eldj_wrap, args=(pstate, reporter, result))

    return result


### HELPER METHODS ###

def _eld_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls
    to _evaluate_lindblad_discrete.

    Args:
    controls
    pstate
    reporter
    result

    Returns:
    total_error
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    clip_control_norms(pstate.max_control_norms, controls)

    # Evaluate the cost function.
    return _evaluate_lindblad_discrete(controls, pstate, reporter)


def _eldj_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls to 
    the jacobian of _evaluate_indblad_discrete.

    Args:
    controls
    pstate
    reporter
    result

    Returns:
    grads
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    clip_control_norms(pstate.max_control_norms, controls)

    # Evaluate the jacobian.
    total_error, grads = (ans_jacobian(_evaluate_lindblad_discrete, 0)
                          (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = conjugate(grads)

    # The states need to be unwrapped from their autograd box.
    if isinstance(reporter.final_densities, Box):
        final_densities = reporter.final_densities._value

    # Update best configuration.
    if total_error < result.best_total_error:
        result.best_controls = controls
        result.best_final_densities = final_densities
        result.best_iteration = reporter.iteration
        result.best_total_error = total_error
    
    # Save and log optimization progress.
    pstate.log_and_save(controls, final_densities, total_error,
                        grads, reporter.iteration,)
    reporter.iteration += 1

    # Convert the gradients from cost function to optimizer format.
    grads = strip_controls(pstate.complex_controls, grads)
    
    return grads


def _evaluate_lindblad_discrete(controls, pstate, reporter):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute associated optimization costs for time-discrete
    control parameters.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.models.GrapeLindbladDiscreteState
        or qoc.models.EvolveLindbladDiscreteState - the program state
    reporter :: any - the object to keep track of relevant information

    Returns:
    total_error :: float - the optimization cost for the provided controls
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    densities = pstate.initial_densities
    dt = pstate.dt
    evolution_time = pstate.evolution_time
    hamiltonian = pstate.hamiltonian
    final_control_step = pstate.final_control_step
    control_step_count = final_control_step + 1
    final_system_step = pstate.final_system_step
    interpolation_policy = pstate.interpolation_policy
    lindblad_data = pstate.lindblad_data
    operation_policy = pstate.operation_policy
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0
    rhs_lindbladian = _get_rhs_lindbladian(control_step_count,
                                           controls,
                                           evolution_time,
                                           hamiltonian,
                                           interpolation_policy,
                                           lindblad_data,)

    densities = integrate_rkdp5(rhs_lindbladian, evolution_time, 0, densities)
    total_error = costs[0].cost(controls, densities, 0)
    reporter.final_densities = densities
    reporter.total_error = total_error

    # for system_step in range(final_system_step + 1):
    #     control_step, _ = divmod(system_step, system_step_multiplier)
    #     is_final_control_step = control_step == final_control_step
    #     is_final_system_step = system_step == final_system_step
    #     time = system_step * dt

    #     # Evolve the density matrices.
    #     densities = integrate_rkdp5(rhs_lindbladian, time + dt, time, densities)

    #     # Compute the costs.
    #     if is_final_system_step:
    #         for i, cost in enumerate(costs):
    #             error = cost.cost(controls, densities, system_step)
    #             total_error = total_error + error
    #         #ENDFOR
    #         reporter.final_densities = densities
    #         reporter.total_error = total_error
    #     else:
    #         for i, step_cost in enumerate(step_costs):
    #             error = step_cost.cost(controls, densities, system_step)
    #             total_error = total_error + error
    #         #ENDFOR
    # #ENDFOR

    return total_error


def _get_rhs_lindbladian(control_step_count=None,
                         controls=None,
                         evolution_time=None,
                         hamiltonian=None,
                         interpolation_policy=InterpolationPolicy.LINEAR,
                         lindblad_data=None,):
    """
    Produce a function that returns the lindbladian at any point in time.

    Arguments:
    control_step_count
    controls
    dt
    hamiltonian
    interpolation_policy
    lindblad_data

    Returns:
    rhs_lindbladian :: (time :: float, densities :: ndarray (density_count x hilbert_size x hilbert_size))
                       -> lindbladian :: ndarray (hilbert_size x hilbert_size)
        - A function that returns the right-hand side of the lindblad master equation
        dp_dt = rhs(p, t)
    """
    # Construct an interpolator for the controls if controls were specified.
    # Otherwise, construct a dummy function.
    if ((not (controls is None))
      and (not (control_step_count is None))
      and (not (evolution_time is None))):
        if interpolation_policy == InterpolationPolicy.LINEAR:
            dt_controls = evolution_time / control_step_count
            control_times = dt_controls * np.arange(control_step_count)
            get_controls = get_linear_interpolator(control_times, controls)
        else:
            raise NotImplementedError("This operation does not yet support the interpolation "
                                      "policy {}."
                                      "".format(interpolation_policy))
    else:
        get_controls = lambda time: None

    # Construct dummy functions if the hamiltonian or lindblad functions were not specified.
    if hamiltonian is None:
        hamiltonian = lambda controls, time: None
        
    if lindblad_data is None:
        lindblad_data = lambda time: (None, None)
        
    def rhs(time, densities):
        controls_ = get_controls(time)
        hamiltonian_ = hamiltonian(controls_, time)
        dissipators, operators = lindblad_data(time)
        lindbladian = get_lindbladian(densities, dissipators, hamiltonian_, operators)
        
        return lindbladian
    #ENDDEF

    return rhs


### MODULE TESTS ###

_BIG = int(1e1)

def _test_evolve_lindblad_discrete():
    """
    Run end-to-end tests on evolve_lindblad_discrete.
    """
    import numpy as np
    from qutip import mesolve, Qobj, Options
    
    from qoc.standard import (conjugate_transpose,
                              SIGMA_X, SIGMA_Y,
                              matrix_to_column_vector_list,
                              SIGMA_PLUS, SIGMA_MINUS,)

    def _generate_complex_matrix(matrix_size):
        return (np.random.rand(matrix_size, matrix_size)
                + 1j * np.random.rand(matrix_size, matrix_size))
    
    def _generate_hermitian_matrix(matrix_size):
        matrix = _generate_complex_matrix(matrix_size)
        return (matrix + conjugate_transpose(matrix)) * 0.5

    # Test that evolution WITH a hamiltonian and WITHOUT lindblad operators
    # yields a known result.
    # Use e.q. 109 from
    # https://arxiv.org/pdf/1904.06560.pdf.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array(((1,   0,   0, 0),
                              (0,   0, -1j, 0),
                              (0, -1j,   0, 0),
                              (0,   0,   0, 1)))
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    initial_densities = np.matmul(initial_states, conjugate_transpose(initial_states))
    target_densities = np.matmul(target_states, conjugate_transpose(target_states))
    system_hamiltonian = ((1/ 2) * (np.kron(SIGMA_X, SIGMA_X)
                              + np.kron(SIGMA_Y, SIGMA_Y)))
    hamiltonian = lambda controls, time: system_hamiltonian
    control_step_count = int(1e3)
    evolution_time = np.pi / 2
    result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                      initial_densities, hamiltonian=hamiltonian)
    final_densities = result.final_densities
    print("target_densities:\n{}"
          "".format(target_densities))
    print("final_densities:\n{}"
          "".format(final_densities))
    exit(0)
    assert(np.allclose(final_densities, target_densities))
    # Note that qutip only gets this result within 1e-5 error.
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = list()
    e_ops = list()
    options = Options(nsteps=control_step_count)
    for i, initial_density in enumerate(initial_densities):
        result = mesolve(Qobj(system_hamiltonian),
                         Qobj(initial_density),
                         tlist, c_ops, e_ops,
                         options=options)
        final_density = result.states[-1].full()
        target_density = target_densities[i]
        assert(np.allclose(final_density, target_density, atol=1e-5))
    #ENDFOR

    # Test that evolution WITHOUT a hamiltonian and WITH lindblad operators
    # yields a known result.
    # This test ensures that dissipators are working correctly.
    # Use e.q.14 from
    # https://inst.eecs.berkeley.edu/~cs191/fa14/lectures/lecture15.pdf.
    hilbert_size = 2
    gamma = 2
    lindblad_dissipators = np.array((gamma,))
    lindblad_operators = np.stack((SIGMA_MINUS,))
    lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
    evolution_time = 1.
    control_step_count = int(1e3)
    inv_sqrt_2 = 1 / np.sqrt(2)
    a0 = np.random.rand()
    c0 = 1 - a0
    b0 = np.random.rand()
    b0_star = np.conj(b0)
    initial_density_0 = np.array(((a0,        b0),
                                  (b0_star,   c0)))
    initial_densities = np.stack((initial_density_0,))
    gt = gamma * evolution_time
    expected_final_density = np.array(((1 - c0 * np.exp(- gt),    b0 * np.exp(-gt/2)),
                                       (b0_star * np.exp(-gt/2), c0 * np.exp(-gt))))
    result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                      initial_densities,
                                      lindblad_data=lindblad_data)
    final_density = result.final_densities[0]
#    assert(np.allclose(final_density, expected_final_density))

    # Test that evolution WITH a random hamiltonian and WITH random lindblad operators
    # yields a similar result to qutip.
    # Note that the allclose tolerance may need to be adjusted.
    matrix_size = 4
    for i in range(_BIG):
        # Evolve under lindbladian.
        hamiltonian_matrix = _generate_hermitian_matrix(matrix_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        lindblad_operator_count = np.random.randint(1, matrix_size)
        lindblad_operators = np.stack([_generate_complex_matrix(matrix_size)
                                      for _ in range(lindblad_operator_count)])
        lindblad_dissipators = np.ones((lindblad_operator_count))
        lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
        density_matrix = _generate_hermitian_matrix(matrix_size)
        initial_densities = np.stack((density_matrix,))
        evolution_time = 1
        control_step_count = int(1e4)
        result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                          initial_densities,
                                          hamiltonian=hamiltonian,
                                          lindblad_data=lindblad_data)
        final_density = result.final_densities[0]

        # Evolve under lindbladian with qutip.
        hamiltonian_qutip =  Qobj(hamiltonian_matrix)
        initial_density_qutip = Qobj(density_matrix)
        lindblad_operators_qutip = [Qobj(lindblad_operator)
                                    for lindblad_operator in lindblad_operators]
        e_ops_qutip = list()
        tlist = np.linspace(0, evolution_time, control_step_count)
        options = Options(nsteps=control_step_count)
        result_qutip = mesolve(hamiltonian_qutip,
                               initial_density_qutip,
                               tlist,
                               lindblad_operators_qutip,
                               e_ops_qutip,)
        final_density_qutip = result_qutip.states[-1].full()
        
        assert(np.allclose(final_density, final_density_qutip, atol=1e-6))
    #ENDFOR


def _test_grape_lindblad_discrete():
    """
    Run end-to-end test on the grape_lindblad_discrete function.

    NOTE: We mostly care about the tests for evolve_lindblad_discrete.
    For grape_lindblad_discrete we care that everything is being passed
    through functions properly, but autograd has a really solid testing
    suite and we trust that their gradients are being computed
    correctly.
    """
    import numpy as np
    
    from qoc.standard import (conjugate_transpose,
                              ForbidDensities, SIGMA_X, SIGMA_Y,)
    
    # Test that parameters are clipped if they grow too large.
    hilbert_size = 4
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                            + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, t: (controls[0] * hamiltonian_matrix)
    initial_states = np.array([[[0], [1], [0], [0]]])
    initial_densities = np.matmul(initial_states, conjugate_transpose(initial_states))
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    forbidden_densities = np.matmul(forbidden_states, conjugate_transpose(forbidden_states))
    control_count = 1
    evolution_time = 10
    control_step_count = 10
    max_norm = 1e-10
    max_control_norms = np.repeat(max_norm, control_count)
    costs = [ForbidDensities(forbidden_densities, control_step_count)]
    iteration_count = 100
    log_iteration_step = 0
    result = grape_lindblad_discrete(control_count, control_step_count,
                                     costs, evolution_time,
                                     initial_densities,
                                     hamiltonian=hamiltonian,
                                     iteration_count=iteration_count,
                                     log_iteration_step=log_iteration_step,
                                     max_control_norms=max_control_norms)
    for i in range(result.best_controls.shape[1]):
        assert(np.less_equal(np.abs(result.best_controls[:,i]),
                             max_control_norms[i]).all())


def _test():
    """
    Run tests on the module.
    """
    _test_evolve_lindblad_discrete()
    # _test_grape_lindblad_discrete()


if __name__ == "__main__":
    _test()
