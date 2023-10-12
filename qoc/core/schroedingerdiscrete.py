"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

from autograd.extend import Box
from autograd import grad
import autograd.numpy as np
from qoc.core.common import (initialize_controls,
                             strip_controls,
                             clip_control_norms)
from qoc.models import (Dummy,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        ProgramType, )
from qoc.standard import (Adam, expm,ans_jacobian,
                           expmat_der_vec_mul)
def grape_schroedinger_discrete(H_s, H_controls, control_eval_count,
                                costs, evolution_time,
                                initial_states,initial_controls = None,control_func = None,
                                impose_control_conditions=None,
                                iteration_count=1000,
                                log_iteration_step=10,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_file_path=None,
                                save_intermediate_states=False,
                                save_iteration_step=0, gradients_method="AD", expm_method="pade",robust_set=None):
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
    expm_method :: string - the method for evaluating propagator-state product.
    Returns:
    result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
    """
    # Initialize the controls.
    control_count = len(H_controls)
    if control_func == None:
        control_func = []
        for k in range(control_count):
            control_func.append(PWC)
    if initial_controls == None:
        initial_controls, max_control_norms = initialize_controls(
        control_count,
        control_eval_count,
        evolution_time,
        initial_controls,
        max_control_norms)
    if robust_set==None:
        def robust_operator(para):
            return para*np.identity(len(H_s))
        robust_set = [np.array([0]),robust_operator]
    # Construct the program state.
    if max_control_norms is None:
        max_control_norms = np.ones(control_count)
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
                                            expm_method, control_func, robust_set
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

def PWC(controls,time,i):
    return controls[i]

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
    evolution_time = pstate.evolution_time
    control_func = pstate.control_func
    # Convert the controls from optimizer format to cost function format.
    controls = controls.reshape((control_count, int(len(controls) / control_count)))
    # Impose user boundary conditions.
    if pstate.impose_control_conditions is not None:
        controls = pstate.impose_control_conditions(controls)
    # descretize time axis
    times = np.linspace(0, evolution_time, control_eval_count + 1)
    # convert control into piece-wise format
    pwcontrols = np.zeros((control_count, control_eval_count))
    # Evaluate the cost function.
    error = _evaluate_schroedinger_discrete(pwcontrols, pstate, reporter)

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

    control_count = pstate.control_count
    control_eval_count = pstate.control_eval_count
    evolution_time = pstate.evolution_time
    control_func = pstate.control_func
    # Convert the controls from optimizer format to cost function format.
    para_num = int(len(controls)/control_count)
    controls = controls.reshape((control_count,para_num))
    # Impose user boundary conditions.
    if pstate.impose_control_conditions is not None:
        controls = pstate.impose_control_conditions(controls)
    #descretize time axis
    times = np.linspace(0, evolution_time, control_eval_count + 1)
    #convert control into piece-wise format
    pwcontrols = np.zeros((control_count,control_eval_count))
    for k in range(control_count):
        for i in range(control_eval_count):
            pwcontrols[k][i] = control_func[k](controls[k],times[i],i)
    #partial derivative of piece-wise control with respect to control parameters
    grads_control_para = []
    for k in range(control_count):
        grads_control_para.append([])
        for i in range(control_eval_count):
            grads_control_para[k].append(grad(control_func[k])(controls[k], times[i],i))
        grads_control_para[k] = np.array(grads_control_para[k]).transpose()
    grads_control_para = np.array(grads_control_para)

    # Rescale the controls to their maximum norm.
    # clip_control_norms(controls,
    #                    pstate.max_control_norms)



    # Evaluate the jacobian.
    if pstate.gradients_method == "AD":
        consider_error_control = False
        for cost in pstate.costs:
            if cost.type == "control_explicit_related":
                consider_error_control = True
        error, grads = (ans_jacobian(_evaluate_schroedinger_discrete, 0)
                        (pwcontrols, pstate, reporter))
        if consider_error_control:
            error_control, grads_control = (ans_jacobian(control_cost, 0)(pwcontrols, pstate))
            error += error_control
            grads += grads


    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    else:
        #we first calculated hard-coded gradients for fidelity-related cost functions
        #we then calculate control-related cost functions by autograd
        consider_error_control = False
        for cost in pstate.costs:
            if cost.type == "control_explicit_related":
                consider_error_control = True
        if consider_error_control:
            error_control, grads_control = (ans_jacobian(control_cost, 0)(pwcontrols, pstate))
            error = _evaluate_schroedinger_discrete(pwcontrols, pstate, reporter)
            grads_state = H_gradient(pwcontrols, pstate, reporter)
            grads = grads_state + grads_control
            error += error_control
        else:
            error = _evaluate_schroedinger_discrete(pwcontrols, pstate, reporter)
            grads_state = H_gradient(pwcontrols, pstate, reporter)
            grads = grads_state

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
    error_set = []
    for cost in (pstate.costs):
        if isinstance(cost.cost_value, Box):
            error_set.append(cost.cost_value._value)
        else:
            error_set.append(cost.cost_value)

    grads_optimizer = np.zeros((control_count,para_num))
    #calculate overall gradients
    for k in range(control_count):
        grads_optimizer[k] = np.sum(grads[k] * grads_control_para[k], axis=1)
    # Convert the gradients from cost function to optimizer format.
        # Save and log optimization progress.
    grads = grads_optimizer
    pstate.log_and_save(controls, error, final_states,
                            grads, reporter.iteration, error_set)
    reporter.iteration += 1
    grads = strip_controls(pstate.complex_controls, grads)

    # Determine if optimization should terminate.
    if abs(error - pstate.last_error) <= pstate.min_error:
        terminate = True
    else:
        terminate = False
    pstate.last_error = error
    return grads, terminate



def control_cost(controls, pstate, ):
    """
    This function computes control explicitly related cost only.
    Parameters
    ----------
    controls : ndarray (control_count x control_eval_count)
        - the control parameters
    pstate : qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects

    Returns
    -------

    """
    error = 0.
    costs = pstate.costs
    states = None
    for i, cost in enumerate(costs):
        if cost.type == "control_explicit_related":
            # variable "states" is not used in these cost.cost function
            cost_error = cost.cost(controls, states, 0)
            error = error + cost_error
    return error


def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Arguments:
    controls :: ndarray (control_count x control_eval_count)
        - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    error :: float - total error of the evolution
    """

    cost_eval_step = 1
    costs = pstate.costs
    program_type = pstate.program_type
    if program_type == ProgramType.GRAPE:
        iteration = reporter.iteration
    else:
        iteration = 0
    save_intermediate_states = pstate.save_intermediate_states_

    step_costs = pstate.step_costs
    system_eval_count = pstate.control_eval_count
    states = pstate.initial_states
    pstate.forward_states = [pstate.initial_states.transpose()]
    # if pstate.robust_set != None:
    #     fluc_para=[]
    #     for i in range(len(pstate.robust_set[0])):
    #         fluc_para.append(np.random.choice(pstate.robust_set[0][i], 1)[0])
    #     fluc_oper = pstate.robust_set[1]
    #     print(fluc_para[0]/(2*np.pi))
    dt = pstate.evolution_time / system_eval_count
    fluc_oper = pstate.robust_set[1]
    fluc_para = pstate.robust_set[0]
    error = 0
    print_infidelity=[]
    for i in range(len(fluc_para)):
        H_s = pstate.H_s
        H_controls = pstate.H_controls
        states = np.transpose(pstate.initial_states)
        gradients_method = pstate.gradients_method
        # initialize the cost value of each time-dependent cost functions
        for i, step_cost in enumerate(step_costs):
            step_cost.cost_value = 0
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
                                                system_eval_step, )

            # Determine where we are in the mesh.
            cost_step, cost_step_remainder = divmod(system_eval_step, cost_eval_step)
            is_cost_step = cost_step_remainder == 0
            # Evolve the states to the next time step.
            H_total = get_H_total(controls, H_controls, H_s, system_eval_step)
            states = expm(-1j * dt * H_total, states, pstate.expm_method, gradients_method)
            if gradients_method == "SAD":
                pstate.forward_states.append(states)
            # Compute step costs every `cost_step`.
            if is_cost_step:
                for i, step_cost in enumerate(step_costs):
                    cost_error = step_cost.cost(controls, states, gradients_method)
                    step_cost.cost_value += cost_error
                    error += cost_error
                # ENDFOR
        # ENDFOR

        # Compute non-step-costs.
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation == False and cost.type == "control_implicit_related":
                cost_error = cost.cost(controls, states, gradients_method)
                error = error + cost_error
                if isinstance(cost_error, Box):
                    print_infidelity.append(cost_error._value)
                else:
                    print_infidelity.append(cost_error)
    # Report reults.
    # with np.printoptions(formatter={'float_kind': '{:0.1e}'.format}):
    #     print(np.array(print_infidelity))
    reporter.error = error
    reporter.final_states = states
    return error

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
@profile
def H_gradient(controls, pstate, reporter):
    """
    Compute hard-coded gradients of states-related cost contributions.
    Parameters
    ----------
    controls : ndarray (control_count x control_eval_count)
        - the control parameters
    pstate : qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter : any - a reporter for mutable objects
    Returns
    -------
    grads : ndarray gradients.
    """
    H_s = pstate.H_s
    H_controls = pstate.H_controls
    system_eval_count = pstate.control_eval_count
    dt = pstate.evolution_time / system_eval_count
    costs = pstate.costs
    grads = np.zeros_like(controls)
    states = reporter.final_states
    control_count = len(H_controls)
    tol = 1e-8
    gradients_method = pstate.gradients_method
    back_states = 0*states.transpose()
    for l in range(len(costs)):
        if costs[l].type == "control_implicit_related":
            #initialize the backward-propagated states which relate to phi in the paper
            s = costs[l].gradient_initialize()
            back_states = back_states+s
    for system_eval_step in range(system_eval_count):
        # Backward propagation. Consider time step N, N-1, ..., 1 sequentially
        H_total = get_H_total(controls, H_controls, H_s, system_eval_count-system_eval_step-1)
        if pstate.gradients_method=="HG":
            states = expm(1j*dt*H_total, states, pstate.expm_method, gradients_method)
        else:
            states = pstate.forward_states[system_eval_count-system_eval_step-1]
        back_states_der, back_states = expmat_der_vec_mul(1j*dt*H_total, 1j * dt * np.array(H_controls) , tol, back_states, pstate.expm_method, gradients_method)
        for k in range(control_count):
            M = np.matmul(np.conjugate(back_states_der[k]),states)
            grads[k][system_eval_count-system_eval_step-1] = 2 * np.real(np.trace(M))
        for l in range(len((costs))):
            if costs[l].type == "control_implicit_related":
                back_states += costs[l].update_state_back(states)

    return grads

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
        H_total = H_total + control[time_step] * H_control
    return H_total