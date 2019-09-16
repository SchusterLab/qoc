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
                        integrate_rkdp5,
                        interpolate_linear_set,
                        InterpolationPolicy,
                        OperationPolicy,
                        get_lindbladian,
                        GrapeLindbladDiscreteState,
                        GrapeLindbladResult,)
from qoc.standard import (Adam, ans_jacobian, commutator, conjugate,
                          conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_lindblad_discrete(evolution_time, initial_densities,
                             system_eval_count,
                             control_eval_count=0,
                             controls=None,
                             cost_eval_step=1,
                             costs=list(),
                             hamiltonian=None,
                             interpolation_policy=InterpolationPolicy.LINEAR,
                             lindblad_data=None,
                             save_file_path=None,
                             save_intermediate_densities=False):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute the optimization error.

    Arguments:
    evolution_time
    initial_densities
    system_eval_count

    control_eval_count
    controls
    cost_eval_step
    costs
    hamiltonian
    interpolation_policy
    lindblad_data
    save_file_path
    save_intermediate_densities

    Returns:
    result
    """
    pstate = EvolveLindbladDiscreteState(control_eval_count,
                                         cost_eval_step, costs,
                                         evolution_time, hamiltonian,
                                         initial_densities,
                                         interpolation_policy,
                                         lindblad_data, save_file_path,
                                         save_intermediate_densities,
                                         system_eval_count)
    result = EvolveLindbladResult()
    _ = _evaluate_lindblad_discrete(controls, pstate, result)

    return result


def grape_lindblad_discrete(control_count, control_eval_count,
                            costs, evolution_time, initial_densities,
                            system_eval_count,
                            complex_controls=False,
                            cost_eval_step=1,
                            hamiltonian=None,
                            impose_control_conditions=None,
                            initial_controls=None,
                            interpolation_policy=InterpolationPolicy.LINEAR,
                            iteration_count=1000,
                            lindblad_data=None,
                            log_iteration_step=10,
                            max_control_norms=None,
                            min_error=0,
                            optimizer=Adam(),
                            save_file_path=None, save_iteration_step=0,):
    """
    This method optimizes the evolution of a set of states under the lindblad
    equation for time-discrete control parameters.

    Arguments:
    control_count
    control_eval_count
    costs
    evolution_time
    initial_densities
    system_eval_count

    complex_controls
    cost_eval_step
    hamiltonian
    impose_control_conditions
    initial_controls
    interpolation_policy
    iteration_count
    lindblad_data
    log_iteration_step
    max_control_norms
    min_error
    optimizer
    save_file_path
    save_iteration_step

    Returns:
    result
    """
    # Initialize the controls.
    initial_controls, max_control_norms = initialize_controls(complex_controls,
                                                              control_count,
                                                              control_eval_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms,)
    # Construct the program state.
    pstate = GrapeLindbladDiscreteState(complex_controls,
                                        control_count,
                                        control_eval_count, cost_eval_step, costs,
                                        evolution_time, hamiltonian,
                                        impose_control_conditions,
                                        initial_controls,
                                        initial_densities,
                                        interpolation_policy, iteration_count,
                                        lindblad_data,
                                        log_iteration_step, max_control_norms,
                                        min_error, optimizer,
                                        save_file_path, save_iteration_step,
                                        system_eval_count,)
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
    error
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    # Rescale the controls to their maximum norm.
    clip_control_norms(controls,
                       pstate.max_control_norms)
    # Impose user boundary conditions.
    if pstate.impose_control_conditions:
        controls = pstate.impose_control_conditions(controls)

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
    # Rescale the controls to thier maximum norm.
    clip_control_norms(controls, pstate.max_control_norms)
    # Impose user boundary conditions.
    if pstate.impose_control_conditions:
        controls = pstate.impose_control_conditions(controls)

    # Evaluate the jacobian.
    error, grads = (ans_jacobian(_evaluate_lindblad_discrete, 0)
                    (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = conjugate(grads)

    # The densities need to be unwrapped from their autograd box.
    if isinstance(reporter.final_densities, Box):
        final_densities = reporter.final_densities._value

    # Update best configuration.
    if error < result.best_error:
        result.best_controls = controls
        result.best_error = error
        result.best_final_densities = final_densities
        result.best_iteration = reporter.iteration
    
    # Save and log optimization progress.
    pstate.log_and_save(controls, error, final_densities,
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

    Arguments:
    controls :: ndarray - the control parameters
    pstate :: qoc.models.GrapeLindbladDiscreteState
        or qoc.models.EvolveLindbladDiscreteState - the program state
    reporter :: any - the object to keep track of relevant information

    Returns:
    error :: float - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    control_eval_times = pstate.control_eval_times
    cost_eval_step = pstate.cost_eval_step
    costs = pstate.costs
    densities = pstate.initial_densities
    dt = pstate.dt
    evolution_time = pstate.evolution_time
    final_system_eval_step = pstate.final_system_eval_step
    hamiltonian = pstate.hamiltonian
    interpolation_policy = pstate.interpolation_policy
    lindblad_data = pstate.lindblad_data
    save_intermediate_densities = pstate.save_intermediate_densities_
    step_costs = pstate.step_costs
    system_eval_count = pstate.system_eval_count
    error = 0
    rhs_lindbladian = _get_rhs_lindbladian(control_eval_times,
                                           controls,
                                           evolution_time,
                                           hamiltonian,
                                           interpolation_policy,
                                           lindblad_data,)

    # Evolve the densities to `evolution_time`.
    # Compute step-costs along the way.
    for system_eval_step in range(system_eval_count):
        # Save the current densities.
        if save_intermediate_densities:
            pstate.save_intermediate_densities(densities, system_eval_step)

        # Determine where we are in the mesh.
        cost_step, cost_step_remainder = divmod(system_eval_step, cost_eval_step)
        is_cost_step = cost_step_remainder == 0
        is_first_system_eval_step = system_eval_step == 0
        is_final_system_eval_step = system_eval_step == final_system_eval_step
        time = system_eval_step * dt

        # Compute step costs every `cost_step`.
        if is_cost_step and not is_first_system_eval_step:
            for i, step_cost in enumerate(step_costs):
                cost_error = step_cost.cost(controls, densities, system_eval_step)
                error = error + cost_error
        
        # Evolve the densities to the next time step.
        if not is_final_system_eval_step:
            densities = integrate_rkdp5(rhs_lindbladian, np.array([time + dt]),
                                        time, densities)
    #ENDFOR

    # Compute non-step-costs.
    for i, cost in enumerate(costs):
        if not cost.requires_step_evaluation:
            cost_error = cost.cost(controls, densities, system_eval_step)
            error = error + cost_error
    
    # Report results.
    reporter.error = error  
    reporter.final_densities = densities

    return error


def _get_rhs_lindbladian(control_eval_times=None,
                         controls=None,
                         evolution_time=None,
                         hamiltonian=None,
                         interpolation_policy=InterpolationPolicy.LINEAR,
                         lindblad_data=None,):
    """
    Produce a function that returns the lindbladian at any point in time.

    Arguments:
    control_eval_times
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
    if controls is not None and control_eval_times is not None:
        if interpolation_policy == InterpolationPolicy.LINEAR:
            interpolate = interpolate_linear_set
        else:
            raise NotImplementedError("This operation does not yet support the interpolation "
                                      "policy {}."
                                      "".format(interpolation_policy))
    else:
        interpolate = lambda x, xs, ys: None

    # Construct dummy functions if the hamiltonian or lindblad functions were not specified.
    if hamiltonian is None:
        hamiltonian = lambda controls, time: None
        
    if lindblad_data is None:
        lindblad_data = lambda time: (None, None)
        
    def rhs(time, densities):
        controls_ = interpolate(time, control_eval_times, controls)
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
    from qutip import mesolve, Qobj
    
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
    system_eval_count = 2
    evolution_time = np.pi / 2
    result = evolve_lindblad_discrete(evolution_time,
                                      initial_densities,
                                      system_eval_count,
                                      hamiltonian=hamiltonian)
    final_densities = result.final_densities
    assert(np.allclose(final_densities, target_densities))
    # Note that qutip only gets this result within 1e-5 error.
    tlist = np.array([0, evolution_time])
    c_ops = list()
    e_ops = list()
    for i, initial_density in enumerate(initial_densities):
        result = mesolve(Qobj(system_hamiltonian),
                         Qobj(initial_density),
                         tlist, c_ops, e_ops,)
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
    sigma_plus = np.array([[0, 1], [0, 0]])
    lindblad_operators = np.stack((sigma_plus,))
    lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
    evolution_time = 1.
    system_eval_count = 2
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
    result = evolve_lindblad_discrete(evolution_time,
                                      initial_densities,
                                      system_eval_count,
                                      lindblad_data=lindblad_data)
    final_density = result.final_densities[0]
    assert(np.allclose(final_density, expected_final_density))

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
        lindblad_dissipators = np.ones((lindblad_operator_count,))
        lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
        density_matrix = _generate_hermitian_matrix(matrix_size)
        initial_densities = np.stack((density_matrix,))
        evolution_time = 5
        system_eval_count = 2
        result = evolve_lindblad_discrete(evolution_time,
                                          initial_densities,
                                          system_eval_count,
                                          hamiltonian=hamiltonian,
                                          lindblad_data=lindblad_data)
        final_density = result.final_densities[0]

        # Evolve under lindbladian with qutip.
        hamiltonian_qutip =  Qobj(hamiltonian_matrix)
        initial_density_qutip = Qobj(density_matrix)
        lindblad_operators_qutip = [Qobj(lindblad_operator)
                                    for lindblad_operator in lindblad_operators]
        e_ops_qutip = list()
        tlist = np.array((0, evolution_time,))
        result_qutip = mesolve(hamiltonian_qutip,
                               initial_density_qutip,
                               tlist,
                               lindblad_operators_qutip,
                               e_ops_qutip,)
        final_density_qutip = result_qutip.states[-1].full()
        assert(np.allclose(final_density, final_density_qutip))
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
    system_eval_count = control_eval_count = 11
    max_norm = 1e-10
    max_control_norms = np.repeat(max_norm, control_count)
    costs = [ForbidDensities(forbidden_densities, system_eval_count)]
    iteration_count = 5
    log_iteration_step = 0
    result = grape_lindblad_discrete(control_count, control_eval_count,
                                     costs, evolution_time,
                                     initial_densities,
                                     system_eval_count,
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
    _test_grape_lindblad_discrete()


if __name__ == "__main__":
    _test()
