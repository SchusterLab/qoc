"""
tutorial.py - This module is a walkthrough for some of QOC's functionality.

This tutorial will follow the experimental setup of [1].
All amplitudes are in GHz. All time scales are in ns.

References:
[1] https://arxiv.org/abs/1608.02430
"""

DEBUG = False
EXECUTE = True
PRINT = False

# Every computation that qoc performs has to be differentiable.
# We use autograd (https://github.com/HIPS/autograd)
# to do automatic differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation).
# All operations that you perform on your operands should use autograd's
# numpy wrapper. autograd.numpy wraps the entire numpy namespace, but not all functions
# have derivatives implemented for them. You may view which functionality is supported
# on autograd's github, linked above.
import autograd.numpy as anp

# First, we define our experimental constants as in [1] pp.7.
PI_2 = 2 * anp.pi
W_T = PI_2 * 5.6640 #GHz
W_C = PI_2 * 4.4526
CHI = PI_2 * -2.194
ALPHA_BY_2 = PI_2 * -2.36e-1
KAPPA_BY_2 = PI_2 * -3.7e-6
CHIP_BY_2 = PI_2 * -1.9e-6
T1_T = 1.7e5 #ns
TP_T = 4.3e4
T1_C = 2.7e6

# Second, we define the system.
# qoc.standard is a module that has optimization cost functions, optimizers,
# convenience functions, and other goodies. All of the functions you import
# from qoc.standard use autograd.numpy, so they are OK to use in your operations.
from qoc.standard import (conjugate_transpose,
                          get_creation_operator,
                          get_annihilation_operator,
                          krons, matmuls,)
CAVITY_STATE_COUNT = 2
TRANSMON_STATE_COUNT = 2
HILBERT_SIZE = CAVITY_STATE_COUNT * TRANSMON_STATE_COUNT
A = get_annihilation_operator(CAVITY_STATE_COUNT)
A_DAGGER = get_creation_operator(CAVITY_STATE_COUNT)
A_ID = anp.eye(CAVITY_STATE_COUNT)
# Notice how the state vectors are specified as column vectors.
CAVITY_VACUUM = anp.zeros((CAVITY_STATE_COUNT, 1))
CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
CAVITY_ZERO[0, 0] = 1
CAVITY_ONE = anp.copy(CAVITY_VACUUM)
CAVITY_ONE[1, 0] = 1
B = get_annihilation_operator(TRANSMON_STATE_COUNT)
B_DAGGER = get_creation_operator(TRANSMON_STATE_COUNT)
B_ID = anp.eye(CAVITY_STATE_COUNT)
TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, 1))
TRANSMON_ZERO = anp.copy(TRANSMON_VACUUM)
TRANSMON_ZERO[0, 0] = 1
TRANSMON_ONE = anp.copy(TRANSMON_VACUUM)
TRANSMON_ONE[1, 0] = 1

# Next, we define the system hamiltonian.
# qoc requires you to specify your hamiltonian as a function of control parameters
# and time
# I.e. hamiltonian_function :: (controls :: ndarray (control_count),
#                               time :: float)
#                              -> hamiltonian_matrix :: ndarray (hilbert_size x hilbert_size)
# You will see this notation in the qoc documentation. The symbol `::` is read "as".
# It specifies the object type of the argument. E.g. 1 :: int, True :: bool, 'hello' :: str.
# The parens that follow the `ndarray` type specifies the shape of the array.
# E.g. anp.array([[1, 2], [3, 4]]) :: ndarray (2 x 2)
# Control parameters are values that you will use to vary time-dependent control fields
# that act on your system. Note that qoc supports both complex and real control parameters.
# In this case, we are controlling a charge drive on the cavity, and a charge drive on the transmon.
# Each drive is parameterized by a single, complex control parameter.
SYSTEM_HAMILTONIAN = (W_C * krons(matmuls(A_DAGGER, A), B_ID)
                      + KAPPA_BY_2 * krons(matmuls(A_DAGGER, A_DAGGER, A , A), B_ID)
                      + W_T * krons(A_ID, matmuls(B_DAGGER, B))
                      + ALPHA_BY_2 * krons(A_ID, matmuls(B_DAGGER, B_DAGGER, B, B))
                      + CHI * krons(matmuls(A_DAGGER, A), matmuls(B_DAGGER, B))
                      + CHIP_BY_2 * krons(matmuls(B_DAGGER, B), matmuls(A_DAGGER, A_DAGGER, A, A)))
CONTROL_0 = krons(A, B_ID)
CONTROL_0_DAGGER = krons(A_DAGGER, B_ID)
CONTROL_1 = krons(A_ID, B)
CONTROL_1_DAGGER = krons(A_ID, B_DAGGER)
def hamiltonian(controls, time):
    return (SYSTEM_HAMILTONIAN
            + controls[0] * CONTROL_0
            + anp.conjugate(controls[0]) * CONTROL_0_DAGGER
            + controls[1] * CONTROL_1
            + anp.conjugate(controls[1]) * CONTROL_1_DAGGER)

# Additionally, we need to specify information to qoc about...
# how long our system will evolve for
EVOLUTION_TIME = 15 #ns
# how many controls we have
CONTROL_COUNT = 2
# what domain our controls are in
COMPLEX_CONTROLS = True
# where our controls are positioned in time
# where our system is evaluated in time
CONTROL_EVAL_COUNT = int(1e2)
SYSTEM_EVAL_COUNT = int(1e2)
# Note that `CONTROL_COUNT` is the length of the `controls` array that is passed
# to our `hamiltonian` function.
# CONTROL_EVAL_COUNT is used to determine how many points in time the `controls` are
# evaluated. It is likely this value should be consistent with a physical apparatus,
# such as the sampling rate of an AWG. The points in time where controls are evaluated
# is given by control_eval_times = anp.linspace(0, evolution_time, control_eval_count).
# Note that qoc uses an interpolation method to interpolate the control parameters
# between these time points. You can change this behavior using the
# `interpolation_policy` argument.
# SYSTEM_EVAL_COUNT is used to determine the update step of the evolution.
# Similarly, system_eval_times = anp.linspace(0, evolution_time, system_eval_count).
# Two important things happen at each system_eval step.
# First, cost functions that are computed multiple times throughout
# the evolution (as opposed to those only computed at the end of evolution)
# are evaluated at system_eval steps. You can change this behavior using the
# `cost_eval_step` argument. Second, qoc uses an exponential series method
# to integrate the schroedinger equation. `system_eval_times` specifies the
# time steps used in this integration. Therefore, increasing the `system_eval_count`
# will likely increase the accuracy of the evolution. The accuracy of the evolution
# can also be increased with the `magnus_policy` argument. Increasing the accuracy
# using both of these methods will increase the computational cost.
# Note that qoc does not use an exponential series method to integrate the lindblad
# equation. Therefore, increasing the `system_eval_count` for lindblad methods
# will not increase the accuracy of their evolution.

# Now, we are ready to give qoc a problem.
# Let's try to put a photon in the cavity.
# That is, we desire the fock state transition |0> -> |1>.
INITIAL_STATE_0 = krons(CAVITY_ZERO, TRANSMON_ZERO)
# Notice that when we specify states (or probability density matrices!)
# to qoc, we always give qoc an array of states that we would like it to track,
# even if we only give qoc a single state. The `,` in anp.stack((INITIAL_STATE_0`,`))
# makes a difference.
INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
assert(INITIAL_STATES.ndim == 3)
TARGET_STATE_0 = krons(CAVITY_ONE, TRANSMON_ZERO)
TARGET_STATES = anp.stack((TARGET_STATE_0,))
# Costs are functions that we want qoc to minimize the output of.
# In this example, we want to minimize the infidelity (maximize the fidelity) of
# the initial state and the target state.
# Note that `COSTS` is a list of cost function objects.
from qoc.standard import TargetStateInfidelity
COSTS = [TargetStateInfidelity(TARGET_STATES)]

# Before we move on, it is a good idea to check that everything looks how you would expect it to.
if PRINT:
    print("HILBERT_SIZE:\n{}"
          "".format(HILBERT_SIZE))
    print("SYSTEM_HAMILTONIAN:\n{}"
          "".format(SYSTEM_HAMILTONIAN))
    print("CAVITY_ZERO:\n{}"
          "".format(CAVITY_ZERO))
    print("CAVITY_ONE:\n{}"
          "".format(CAVITY_ONE))
    print("TRANSMON_ZERO:\n{}"
          "".format(TRANSMON_ZERO))
    print("TRANSMON_ONE:\n{}"
          "".format(TRANSMON_ONE))
    print("INITIAL_STATE_0:\n{}"
          "".format(INITIAL_STATE_0))
    print("TARGET_STATE_0:\n{}"
          "".format(TARGET_STATE_0))
    print("CONTROL_EVAL_TIMES:\n{}"
          "".format(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)))
    print("SYSTEM_EVAL_TIMES:\n{}"
          "".format(anp.linspace(0, EVOLUTION_TIME, SYSTEM_EVAL_COUNT)))


# We want to tell qoc how often to store information about the optimization
# and how often to log output. Both `log_iteration_step` and `save_iteration_step`
# are specified in units of optimization iterations.
from qoc.standard import (generate_save_file_path,
                          Adam, LBFGSB,)
LOG_ITERATION_STEP = 1
EXPERIMENT_NAME = "tutorial_cavity_fock1"
SAVE_PATH = "./out"
if not DEBUG:
    OPT_FILE_PATH = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
else:
    OPT_FILE_PATH = "./out/00031_tutorial_cavity_fock1.h5"
SAVE_ITERATION_STEP = 1

# For this problem, the LBFGSB optimizer reaches a reasonable
# answer very quickly.
OPTIMIZER = LBFGSB()
# In practice, we find that using a second order optimizer, such as LBFGSB,
# gives a good initial answer. Then, this answer may be used with a first
# order optimizer, such as Adam, to achieve the desired error.
# You can seed optimizations with a set of controls using the
# `initial_controls` argument.

if EXECUTE:
    # Lastly, we use the GRAPE algorithm to find a set of time-dependent
    # controls that accomplishes the state transfer that we desire.
    from qoc import grape_schroedinger_discrete
    if not DEBUG:
        result = grape_schroedinger_discrete(CONTROL_COUNT,
                                             CONTROL_EVAL_COUNT,
                                             COSTS, EVOLUTION_TIME,
                                             hamiltonian, INITIAL_STATES,
                                             SYSTEM_EVAL_COUNT,
                                             complex_controls=COMPLEX_CONTROLS,
                                             log_iteration_step=LOG_ITERATION_STEP,
                                             optimizer=OPTIMIZER,
                                             save_file_path=OPT_FILE_PATH,
                                             save_iteration_step=SAVE_ITERATION_STEP,)
    
# Next, we want to do some analysis of our results.
import os
CONTROLS_PLOT_FILE = "{}_controls.png".format(EXPERIMENT_NAME)
CONTROLS_PLOT_FILE_PATH = os.path.join(SAVE_PATH, CONTROLS_PLOT_FILE)
# This function will plot the controls, and their fourier transform,
# that achieved the lowest error.
if EXECUTE:
    from qoc.standard import plot_best_controls
    plot_best_controls(OPT_FILE_PATH,
                       save_file_path=CONTROLS_PLOT_FILE_PATH)

# In order to see the population of our state evolve over time,
# we can save the states to the save file every system_eval_step.
# This is not done by default in the grape methods because
# it is expensive.
SAVE_INTERMEDIATE_STATES = True
if not DEBUG:
    EVOL_FILE_PATH = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
else:
    EVOL_FILE_PATH = "./out/00032_tutorial_cavity_fock1.h5"
if EXECUTE:
    import h5py
    from qoc import evolve_schroedinger_discrete
    
    f = h5py.File(OPT_FILE_PATH)
    best_controls_index = anp.argmin(f["error"])
    best_controls = f["controls"][best_controls_index]
    if not DEBUG:
        result = evolve_schroedinger_discrete(EVOLUTION_TIME,
                                              hamiltonian,
                                              INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                              controls=best_controls,
                                              save_file_path=EVOL_FILE_PATH,
                                              save_intermediate_states=SAVE_INTERMEDIATE_STATES)

    # The plot_population method plots the value of the density matrix
    POPULATION_PLOT_FILE = "{}_population.png".format(EXPERIMENT_NAME)
    POPULATION_PLOT_FILE_PATH = os.path.join(SAVE_PATH, POPULATION_PLOT_FILE)
    from qoc.standard import plot_population_states
    plot_population_states(EVOL_FILE_PATH,
                           save_file_path=POPULATION_PLOT_FILE_PATH)










