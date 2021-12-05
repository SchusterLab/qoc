# Define the system.
from qoc import grape_schroedinger_discrete
from qoc.standard import (Adam,TargetStateInfidelity,TargetStateInfidelityTime,
                          conjugate_transpose,matrix_to_column_vector_list,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path,)
from qoc.models.operationpolicy import OperationPolicy
from qoc.models import (Dummy, EvolveSchroedingerDiscreteState,
                        EvolveSchroedingerResult,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        InterpolationPolicy,
                        MagnusPolicy,
                        ProgramType,)
import numpy as np
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
sigmax = ANNIHILATION_OPERATOR + CREATION_OPERATOR
sigmay = -1j * ANNIHILATION_OPERATOR + 1j * CREATION_OPERATOR
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = SIGMA_Z / 2
# Only real control amplitutdes are supported!
hamiltonian = lambda controls, time: (H_SYSTEM_0
                                      + controls[0] * sigmax
                                      + controls[1] * sigmay)

# Define the optimization.
COMPLEX_CONTROLS = False
CONTROL_COUNT = 2
EVOLUTION_TIME = 10  # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = EVOLUTION_TIME + 1
ITERATION_COUNT = 1000

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_PATH = "./out"
SAVE_FILE_NAME = "transmon_pi"
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
INITIAL_STATES = matrix_to_column_vector_list(np.eye(2))
# we could have equivalently done
# initial_state0 = np.array([[1], [0]])
# initial_state1 = np.array([[0], [1]])
# initial_states = np.stack((initial_state0, initial_state1))
target_unitary = np.array([[0, 1], [1, 0]])
target_states = matrix_to_column_vector_list(target_unitary)
# we could have equivalently done
# target_state0 = np.array([[0], [1]])
# target_state1 = np.array([[1], [0]])
# target_states = np.stack((target_state0, target_state1))


# cost_multiplier is hyperprameter of each cost function
# Set neglect_relative_phase as True, relative phase of each column of gate U will be ignored.
# cost_eval_step :: int >= 1- This value determines how often step-costs are evaluated.The units of this value are in system_eval steps. E.g. if this value is 2,step-costs will be computed every 2 system_eval steps.

cost_1 = TargetStateInfidelityTime(system_eval_count=SYSTEM_EVAL_COUNT,
                                   target_states=target_states, cost_multiplier=1,
                                   neglect_relative_phase=False, cost_eval_step=1)
cost_2 = TargetStateInfidelity(target_states=target_states, cost_multiplier=1, neglect_relative_phase=False)
COSTS = [cost_1]
CONTROL_HAMILTONIAN = [sigmax, sigmay]
manual_parameter = {"control_hamiltonian": CONTROL_HAMILTONIAN, "manual_gradient_mode": True, "tol": 1e-16}

optimizer = Adam(beta_1=0.9, beta_2=0.999, clip_grads=None,
                 epsilon=1e-8, learning_rate=1e-3,
                 learning_rate_decay=None, operation_policy=OperationPolicy.CPU,
                 scale_grads=None)


# a class to define the Adam optimizer
#     This implementation follows the original algorithm
#     https://arxiv.org/abs/1412.6980.
#     Fields:
#     apply_clip_grads :: bool - see clip_grads
#     apply_learning_rate_decay :: bool - see learning_rate_decay
#     apply_scale_grads :: bool - see scale_grads
#     beta_1 :: float - gradient decay bias
#     beta_2 :: float - gradient squared decay bias
#     clip_grads :: float - the maximum absolute value at which the gradients
#         should be element-wise clipped, if not set, the gradients will
#         not be clipped
#     epsilon :: float - fuzz factor
#     gradient_moment :: numpy.ndarray - running optimization variable
#     gradient_square_moment :: numpy.ndarray - running optimization variable
#     initial_learning_rate :: float - the initial step size
#     iteration_count :: int - the current count of iterations performed
#     learning_rate :: float - the current step size
#     learning_rate_decay :: float - the number of iterations it takes for
#         the learning rate to decay by 1/e, if not set, no decay is
#         applied
#     operation_policy
#     name :: str - identifier for the optimizer
#     scale_grads :: float - the value to scale the norm of the gradients to,
#         if not set, the gradients will not be scaled
def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
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
                                         optimizer=optimizer,
                                         save_file_path=None,
                                         save_intermediate_states=False,
                                         save_iteration_step=0, manual_parameter=manual_parameter)


#
# Args:
#    control_count :: int - This is the number of control parameters that qoc should
#        optimize over. I.e. it is the length of the `controls` array passed
#        to the hamiltonian.
#    control_eval_count :: int >= 2 - This value determines where definite values
#        of the control parameters are evaluated. This value is used as:
#        `control_eval_times`= numpy.linspace(0, `evolution_time`, `control_eval_count`).
#   costs :: iterable(qoc.models.cost.Cost) - This list specifies all
#       the cost functions that the optimizer should evaluate. This list
#        defines the criteria for an "optimal" control set.
#    evolution_time :: float - This value specifies the duration of the
#        system's evolution.
#    hamiltonian :: (controls :: ndarray (control_count), time :: float)
#                   -> hamiltonian_matrix :: ndarray (hilbert_size x hilbert_size)
#        - This function provides the system's hamiltonian given a set
#        of control parameters and a time value.
#    initial_states :: ndarray (state_count x hilbert_size x 1)
#        - This array specifies the states that should be evolved under the
#        specified system. These are the states at the beginning of the evolution.
#    system_eval_count :: int >= 2 - This value determines how many times
#        during the evolution the system is evaluated, including the
#        initial value of the system. For the schroedinger evolution,
#        this value determines the time step of integration.
#        This value is used as:
#        `system_eval_times` = numpy.linspace(0, `evolution_time`, `system_eval_count`).
#
#    complex_controls :: bool - This value determines if the control parameters
#        are complex-valued. If some controls are real only or imaginary only
#        while others are complex, real only and imaginary only controls
#        can be simulated by taking the real or imaginary part of a complex control.
#    cost_eval_step :: int >= 1- This value determines how often step-costs are evaluated.
#         The units of this value are in system_eval steps. E.g. if this value is 2,
#         step-costs will be computed every 2 system_eval steps.
#    impose_control_conditions :: (controls :: (control_eval_count x control_count))
#                                -> (controls :: (control_eval_count x control_count))
#        - This function is called after every optimization update. Example uses
#        include setting boundary conditions on the control parameters.
#    initial_controls :: ndarray (control_step_count x control_count)
#       - This array specifies the control parameters at each
#        control step. These values will be used to determine the `controls`
#        argument passed to the `hamiltonian` function at each time step for
#        the first iteration of optimization.
#    interpolation_policy :: qoc.models.interpolationpolicy.InterpolationPolicy
#        - This value specifies how control parameters should be
#        interpreted at points where they are not defined.
#    iteration_count :: int - This value determines how many total system
#        evolutions the optimizer will perform to determine the
#       optimal control set.
#    log_iteration_step :: int - This value determines how often qoc logs
#        progress to stdout. This value is specified in units of system steps,
#        of which there are `control_step_count` * `system_step_multiplier`.
#        Set this value to 0 to disable logging.
#    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy - This value
#        specifies what method should be used to perform the magnus expansion
#        of the system matrix for ode integration. Choosing a higher order
#        magnus expansion will yield more accuracy, but it will
#        result in a longer compute time.
#    max_control_norms :: ndarray (control_count) - This array
#       specifies the element-wise maximum norm that each control is
#        allowed to achieve. If, in optimization, the value of a control
#        exceeds its maximum norm, the control will be rescaled to
#        its maximum norm. Note that for non-complex values, this
#        feature acts exactly as absolute value clipping.
#    min_error :: float - This value is the threshold below which
#        optimization will terminate.
#    optimizer :: class instance - This optimizer object defines the
#        gradient-based procedure for minimizing the total contribution
#        of all cost functions with respect to the control parameters.
#    save_file_path :: str - This is the full path to the file where
#        information about program execution will be stored.
#        E.g. "./out/foo.h5"
#    save_intermediate_densities :: bool - If this value is set to True,
#        qoc will write the densities to the save file after every
#        system_eval step.
#    save_intermediate_states :: bool - If this value is set to True,
#        qoc will write the states to the save file after every
#        system_eval step.
#    save_iteration_step :: int - This value determines how often qoc
#        saves progress to the save file specified by `save_file_path`.
#        This value is specified in units of system steps, of which
#       there are `control_step_count` * `system_step_multiplier`.
#        Set this value to 0 to disable saving.

if __name__ == "__main__":
    main()

# %% md
