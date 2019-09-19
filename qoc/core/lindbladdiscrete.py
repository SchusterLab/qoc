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
from qoc.core.mathmethods import (integrate_rkdp5,
                                  interpolate_linear_set,
                                  get_lindbladian,)
from qoc.models import (Dummy,
                        EvolveLindbladDiscreteState,
                        EvolveLindbladResult,
                        InterpolationPolicy,
                        OperationPolicy,
                        GrapeLindbladDiscreteState,
                        GrapeLindbladResult,)
from qoc.standard import (Adam, ans_jacobian, commutator,
                          conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_lindblad_discrete(evolution_time, initial_densities,
                             system_eval_count,
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
    if controls is not None:
        control_eval_count = controls.shape[0]
    else:
        control_eval_count = 0
    
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
    if pstate.impose_control_conditions is not None:
        controls = pstate.impose_control_conditions(controls)

    # Evaluate the jacobian.
    error, grads = (ans_jacobian(_evaluate_lindblad_discrete, 0)
                    (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = np.conjugate(grads)

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
