"""
0_transmon_pi.py - This module demonstrates
a simple example of grape on the schroedinger equation
using time discrete control parameters to evolve a transmon qubit
form the ground state to the first excited state.
"""

"""
Tips for manual gradient:
1. Set COMPLEX_CONTROLS to False, only real control amplitudes are supported.
2. The sequence of CONTROL_HAMILTONIAN should be consistent with the one in hamiltonia
3. 
"""
import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path,)

# Define the system.
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
sigmax=ANNIHILATION_OPERATOR +CREATION_OPERATOR
sigmay=-1j*ANNIHILATION_OPERATOR+1j*CREATION_OPERATOR
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = SIGMA_Z / 2
# Only real control amplitutdes are supported!
hamiltonian = lambda controls, time: (H_SYSTEM_0
                                      + controls[0] * sigmax
                                      + controls[1]*sigmay)
CONTROL_HAMILTONIAN=[sigmax,sigmay]
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,), axis=0)
TARGET_STATES = anp.stack((TARGET_STATE_0,), axis=0)
COSTS = [TargetStateInfidelity(TARGET_STATES )]

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
CONTROL_HAMILTONIAN=[sigmax,sigmay]

manual_parameter={"control_hamiltonian":CONTROL_HAMILTONIAN,"manual_gradient_mode":True}

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
