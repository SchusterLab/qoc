"""
0_transmon_pi.py - This module demonstrates
a simple example of grape on the schroedinger equation
using time discrete control parameters to evolve a transmon qubit
form the ground state to the first excited state.
"""

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,matrix_to_column_vector_list,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path,)

# Define the system.
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
sigmax=ANNIHILATION_OPERATOR +CREATION_OPERATOR
sigmay=-1j*ANNIHILATION_OPERATOR+1j*CREATION_OPERATOR
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = SIGMA_Z / 2
# Only real control amplitutdes are supported!
hamiltonian = lambda controls, time: (H_SYSTEM_0
                                      + controls[0] * sigmax
                                      + controls[1]*sigmay)


# Define the optimization.
COMPLEX_CONTROLS = False
CONTROL_COUNT = 2
EVOLUTION_TIME = 10 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = EVOLUTION_TIME + 1
ITERATION_COUNT = 1000

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_PATH = "./out"
SAVE_FILE_NAME = "transmon_pi"
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
INITIAL_STATES = matrix_to_column_vector_list(anp.eye(2))
# we could have equivalently done
# initial_state0 = anp.array([[1], [0]])
# initial_state1 = anp.array([[0], [1]])
# initial_states = anp.stack((initial_state0, initial_state1))
target_unitary = anp.array([[0, 1], [1, 0]])
target_states = matrix_to_column_vector_list(target_unitary)
# we could have equivalently done
# target_state0 = anp.array([[0], [1]])
# target_state1 = anp.array([[1], [0]])
# target_states = anp.stack((target_state0, target_state1))
COSTS = [TargetStateInfidelity(target_states)]
CONTROL_HAMILTONIAN=[sigmax,sigmay]
manual_parameter={"control_hamiltonian":CONTROL_HAMILTONIAN,"manual_gradient_mode":True,"Hk_approximation":False}

def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                         complex_controls=COMPLEX_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         save_file_path=SAVE_FILE_PATH,
                                         save_iteration_step=SAVE_ITERATION_STEP,
                                         manual_parameter=manual_parameter)


if __name__ == "__main__":
    main()


