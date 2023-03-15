"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

from autograd.extend import Box
import numpy as np
import autograd.numpy as anp
from qoc.core.common import (initialize_controls,
                             slap_controls, strip_controls,
                             clip_control_norms)
from qoc.models import (Dummy,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        ProgramType,)
from qoc.standard import (Adam, ans_jacobian,
                          expm_pade)

def grape_schroedinger_discrete(H_s, H_controls, control_eval_count,
                                costs, evolution_time,
                                initial_states,
                                impose_control_conditions=None,
                                initial_controls=None,
                                iteration_count=1000, 
                                log_iteration_step=10,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_file_path=None,
                                save_intermediate_states=False,
                                save_iteration_step=0, gradients_method="AD"):
    """
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Args:
    H_s :: ndarray (hilbert_size x hilbert_size ) - Static system Hamiltonian.
    H_controls :: ndarray (control_num x hilbert_size x hilbert_size) - Control Hamiltonians. control_num is # of control fields.
    control_eval_count :: int >= 2 - This value determines the number of time steps for evolving system
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
        the cost functions that the optimizer should evaluate. This list
        defines the criteria for an "optimal" control set.
    evolution_time :: float - This value specifies the duration of the
        system's evolution.
    initial_states :: ndarray (state_count x hilbert_size )
        - This array specifies the states that should be evolved under the
        specified system. These are the states at the beginning of the evolution.
    impose_control_conditions :: (controls :: (control_count x control_eval_count))
                                 -> (controls :: (control_count x control_eval_count))
        - This function is called after every optimization update. Example uses
        include setting boundary conditions on the control parameters.                             
    initial_controls :: ndarray (control_count x control_eval_count)
        - This array specifies the control parameters at each
        control step. These values will be used to determine the `controls`
        argument passed to the `hamiltonian` function at each time step for
        the first iteration of optimization.
    iteration_count :: int - This value determines how many total system
        evolutions the optimizer will perform to determine the
        optimal control set.
    log_iteration_step :: int - This value determines how often qoc logs
        progress to stdout. This value is specified in units of system steps,
        of which there are `control_step_count` * `system_step_multiplier`.
        Set this value to 0 to disable logging.
    max_control_norms :: ndarray (control_count) - This array
        specifies the element-wise maximum norm that each control is
        allowed to achieve. If, in optimization, the value of a control
        exceeds its maximum norm, the control will be rescaled to
        its maximum norm. Note that for non-complex values, this
        feature acts exactly as absolute value clipping.
    min_error :: float - This value is the threshold below which
        optimization will terminate.
    optimizer :: class instance - This optimizer object defines the
        gradient-based procedure for minimizing the total contribution
        of all cost functions with respect to the control parameters.
    save_file_path :: str - This is the full path to the file where
        information about program execution will be stored.
        E.g. "./out/foo.h5"
    save_intermediate_densities :: bool - If this value is set to True,
        qoc will write the densities to the save file after every
        system_eval step.
    save_intermediate_states :: bool - If this value is set to True,
        qoc will write the states to the save file after every
        system_eval step.
    save_iteration_step :: int - This value determines how often qoc
        saves progress to the save file specified by `save_file_path`.
        This value is specified in units of system steps, of which
        there are `control_step_count` * `system_step_multiplier`.
        Set this value to 0 to disable saving.
    gradients_mode :: string - Either AD or HG. They differ in memory
        usage. When the memory usage is not the bottleneck, use AD, and vice versa.

    Returns:
    result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
    """
    # Initialize the controls.
    control_count = len(H_controls)
    initial_controls, max_control_norms = initialize_controls(
                                                              control_count,
                                                              control_eval_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms)
    # Construct the program state.
    pstate = GrapeSchroedingerDiscreteState(H_s, H_controls,
                                            control_eval_count,
                                            costs, evolution_time,
                                            impose_control_conditions,
                                            initial_controls,
                                            initial_states,
                                            iteration_count,
                                            log_iteration_step,
                                            max_control_norms,
                                            min_error, optimizer,
                                            save_file_path,
                                            save_intermediate_states,
                                            save_iteration_step, gradients_method,
                                            )
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
    control_count = pstate.control_count
    control_eval_count = pstate.control_eval_count
    controls = np.reshape(controls, (control_count, control_eval_count))
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
    control_count = pstate.control_count
    control_eval_count = pstate.control_eval_count
    controls = np.reshape(controls, (control_count, control_eval_count))
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
    else:
        final_states = reporter.final_states

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
    cost_eval_step = 1
    costs = pstate.costs
    program_type = pstate.program_type
    if program_type == ProgramType.GRAPE:
        iteration = reporter.iteration
    else:
        iteration = 0
    save_intermediate_states = pstate.save_intermediate_states_
    states = np.transpose(pstate.initial_states)
    step_costs = pstate.step_costs
    system_eval_count = pstate.control_eval_count
    final_system_eval_step = system_eval_count - 1
    dt = pstate.evolution_time / system_eval_count
    H_s = pstate.H_s
    H_controls = pstate.H_controls
    error = 0

    # Evolve the states to `evolution_time`.
    # Compute step-costs along the way.
    for system_eval_step in range(system_eval_count):
        # If applicable, save the current states.
        if save_intermediate_states:
            if isinstance(states, Box):
                intermediate_states = states._value
            else:
                intermediate_states = states
            pstate.save_intermediate_states(iteration,
                                            intermediate_states,
                                            system_eval_step,)
        
        # Determine where we are in the mesh.
        cost_step, cost_step_remainder = divmod(system_eval_step, cost_eval_step)
        is_cost_step = cost_step_remainder == 0

        # Compute step costs every `cost_step`.
        if is_cost_step:
            for i, step_cost in enumerate(step_costs):
                cost_error = step_cost.cost(controls, states, system_eval_step)
                error = error + cost_error
            #ENDFOR

        # Evolve the states to the next time step.
        H_total = get_H_total(controls, H_controls, H_s, system_eval_step)
        propagator=expm_pade(-1j * dt * H_total)
        states = anp.matmul(propagator, states)
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

def get_H_total(controls: np.ndarray, H_controls: np.ndarray,
                H0: np.ndarray, time_step: float) -> np.ndarray:
    """
    Get the total Hamiltonian in the specific time step
    Parameters
    ----------
    controls:
        All control amplitudes
    H_controls:
        Control Hamiltonian
    H0:
        Static system Hamiltonian
    time_step:
        The specific time step

    Returns
    -------
        The total Hamiltonian in the specific time step
    """
    H_total = H0
    for control, H_control in zip(controls, H_controls):
        H_total = H_total + control[time_step ] * H_control
    return H_total

