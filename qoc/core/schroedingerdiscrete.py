"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

from autograd.extend import Box
import numpy as np

from qoc.core.common import (initialize_controls,
                             slap_controls, strip_controls,
                             clip_control_norms,)
from qoc.core.mathmethods import (interpolate_linear_set,
                                  magnus_m2,
                                  magnus_m4,
                                  magnus_m6,)
from qoc.models import (Dummy, EvolveSchroedingerDiscreteState,
                        EvolveSchroedingerResult,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        InterpolationPolicy,
                        MagnusPolicy)
from qoc.standard import (Adam, ans_jacobian,
                          expm, matmuls)

### MAIN METHODS ###

def evolve_schroedinger_discrete(evolution_time, hamiltonian,
                                 initial_states, system_eval_count,
                                 controls=None,
                                 cost_eval_step=1, costs=list(), 
                                 interpolation_policy=InterpolationPolicy.LINEAR,
                                 magnus_policy=MagnusPolicy.M2,
                                 save_file_path=None,
                                 save_intermediate_states=False,):
    """
    Evolve a set of state vectors under the schroedinger equation
    and compute the optimization error.

    Args:
    evolution_time
    hamiltonian
    initial_states
    system_eval_count

    controls
    cost_eval_step
    costs
    interpolation_policy
    magnus_policy
    save_file_path
    save_intermediate_states

    Returns:
    result :: qoc.models.schroedingermodels.EvolveSchroedingerResult
    """
    if controls is not None:
        control_eval_count = controls.shape[0]
    else:
        control_eval_count = 0
    
    pstate = EvolveSchroedingerDiscreteState(control_eval_count,
                                             cost_eval_step,
                                             costs, evolution_time,
                                             hamiltonian, initial_states,
                                             interpolation_policy,
                                             magnus_policy, save_file_path,
                                             save_intermediate_states,
                                             system_eval_count,)
    pstate.save_initial(controls)
    result = EvolveSchroedingerResult()
    _ = _evaluate_schroedinger_discrete(controls, pstate, result)

    return result


def grape_schroedinger_discrete(control_count, control_eval_count,
                                costs, evolution_time, hamiltonian,
                                initial_states, system_eval_count,
                                complex_controls=False,
                                cost_eval_step=1,
                                impose_control_conditions=None,
                                initial_controls=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                iteration_count=1000, 
                                log_iteration_step=10,
                                magnus_policy=MagnusPolicy.M2,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_file_path=None, save_iteration_step=0,):
    """
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Args:
    control_count
    control_eval_count
    costs
    evolution_time
    hamiltonian
    initial_states

    complex_controls
    cost_eval_step
    impose_control_conditions
    initial_controls
    interpolation_policy
    iteration_count
    log_iteration_step
    magnus_policy
    max_control_norms
    min_error
    optimizer
    save_file_path
    save_iteration_step
    system_eval_count

    Returns:
    result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
    """
    # Initialize the controls.
    initial_controls, max_control_norms = initialize_controls(complex_controls,
                                                              control_count,
                                                              control_eval_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms)
    # Construct the program state.
    pstate = GrapeSchroedingerDiscreteState(complex_controls, control_count,
                                            control_eval_count, cost_eval_step,
                                            costs, evolution_time, hamiltonian,
                                            impose_control_conditions,
                                            initial_controls,
                                            initial_states, interpolation_policy,
                                            iteration_count,
                                            log_iteration_step,
                                            max_control_norms, magnus_policy,
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
    result = GrapeSchroedingerResult()
    # Convert the controls from cost function format to optimizer format.
    initial_controls = strip_controls(pstate.complex_controls, pstate.initial_controls)
    # Run the optimization.
    pstate.optimizer.run(_esd_wrap, pstate.iteration_count, initial_controls,
                         _esdj_wrap, args=(pstate, reporter, result))

    return result


### HELPER METHODS ###

def _esd_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls
    to _evaluate_schroedinger_discrete.

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
    error = _evaluate_schroedinger_discrete(controls, pstate, reporter)

    # Determine if optimization should terminate.
    if error <= pstate.min_error:
        terminate = True
    else:
        terminate = False

    return error, terminate


def _esdj_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls to 
    the jacobian of _evaluate_schroedinger_discrete.

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
    # Rescale the controls to their maximum norm.
    clip_control_norms(controls,
                       pstate.max_control_norms)
    # Impose user boundary conditions.
    if pstate.impose_control_conditions is not None:
        controls = pstate.impose_control_conditions(controls)

    # Evaluate the jacobian.
    error, grads = (ans_jacobian(_evaluate_schroedinger_discrete, 0)
                          (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = np.conjugate(grads)

    # The states need to be unwrapped from their autograd box.
    if isinstance(reporter.final_states, Box):
        final_states = reporter.final_states._value

    # Update best configuration.
    if error < result.best_error:
        result.best_controls = controls
        result.best_error = error
        result.best_final_states = final_states
        result.best_iteration = reporter.iteration
    
    # Save and log optimization progress.
    pstate.log_and_save(controls, error, final_states,
                        grads, reporter.iteration)
    reporter.iteration += 1

    # Convert the gradients from cost function to optimizer format.
    grads = strip_controls(pstate.complex_controls, grads)

    # Determine if optimization should terminate.
    if error <= pstate.min_error:
        terminate = True
    else:
        terminate = False
    
    return grads, terminate


def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Arguments:
    controls :: ndarray (control_eval_count x control_count)
        - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    error :: float - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    control_eval_times = pstate.control_eval_times
    cost_eval_step = pstate.cost_eval_step
    costs = pstate.costs
    dt = pstate.dt
    evolution_time = pstate.evolution_time
    final_system_eval_step = pstate.final_system_eval_step
    hamiltonian = pstate.hamiltonian
    interpolation_policy = pstate.interpolation_policy
    magnus_policy = pstate.magnus_policy
    save_intermediate_states = pstate.save_intermediate_states_
    states = pstate.initial_states
    step_costs = pstate.step_costs
    system_eval_count = pstate.system_eval_count
    error = 0

    # Evolve the states to `evolution_time`.
    # Compute step-costs along the way.
    for system_eval_step in range(system_eval_count):
        # Save the current states.
        if save_intermediate_states:
            pstate.save_intermediate_states(states, system_eval_step)
        
        # Determine where we are in the mesh.
        cost_step, cost_step_remainder = divmod(system_eval_step, cost_eval_step)
        is_cost_step = cost_step_remainder == 0
        is_first_system_eval_step = system_eval_step == 0
        is_final_system_eval_step = system_eval_step == final_system_eval_step
        time = system_eval_step * dt
        
        # Compute step costs every `cost_step`.
        if is_cost_step and not is_first_system_eval_step:
            for i, step_cost in enumerate(step_costs):
                cost_error = step_cost.cost(controls, states, system_eval_step)
                error = error + cost_error
            #ENDFOR

        # Evolve the states to the next time step.
        if not is_final_system_eval_step:
            states = _evolve_step_schroedinger_discrete(dt, hamiltonian,
                                                        states, time,
                                                        control_eval_times=control_eval_times,
                                                        controls=controls,
                                                        interpolation_policy=interpolation_policy,
                                                        magnus_policy=magnus_policy,)
    #ENDFOR

    # Compute non-step-costs.
    for i, cost in enumerate(costs):
        if not cost.requires_step_evaluation:
            cost_error = cost.cost(controls, states, final_system_eval_step)
            error = error + cost_error

    # Report reults.
    reporter.error = error
    reporter.final_states = states
    
    return error


def _evolve_step_schroedinger_discrete(dt, hamiltonian,
                                       states, time,
                                       control_eval_times=None,
                                       controls=None,
                                       interpolation_policy=InterpolationPolicy.LINEAR,
                                       magnus_policy=MagnusPolicy.M2,):
    """
    Use the exponential series method via magnus expansion to evolve the state vectors
    to the next time step under the schroedinger equation for time-discrete controls.
    Magnus expansions are implemented using the methods described in
    https://arxiv.org/abs/1709.06483.

    Arguments:
    dt
    hamiltonian
    states
    time

    control_eval_times
    controls
    interpolation_policy
    magnus_policy
    
    Returns:
    states
    """
    # Choose an interpolator.
    if interpolation_policy == InterpolationPolicy.LINEAR:
        interpolate = interpolate_linear_set
    else:
        raise NotImplementedError("The interpolation policy {} "
                                  "is not yet supported for this method."
                                  "".format(interpolation_policy))

    # Choose a control interpolator.
    if controls is not None and control_eval_times is not None:
        interpolate_controls = interpolate
    else:
        interpolate_controls = lambda x, xs, ys: None

    # Construct a function to interpolate the hamiltonian
    # for all time.
    def get_hamiltonian(time_):
        controls_ = interpolate_controls(time_, control_eval_times, controls)
        hamiltonian_ = hamiltonian(controls_, time_)
        return -1j * hamiltonian_
    
    if magnus_policy == MagnusPolicy.M2:
        magnus = magnus_m2(get_hamiltonian, dt, time)
    elif magnus_policy == MagnusPolicy.M4:
        magnus = magnus_m4(get_hamiltonian, dt, time)
    elif magnus_policy == MagnusPolicy.M6:
        magnus = magnus_m6(get_hamiltonian, dt, time)
    else:
        raise ValueError("Unrecognized magnus policy {}."
                         "".format(magnus_policy))
    #ENDIF

    step_unitary = expm(magnus)
    states = matmuls(step_unitary, states)

    return states
