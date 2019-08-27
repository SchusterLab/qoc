"""
0_transmon_pi.py - a module for the simplest problem imaginable
"""

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          PAULI_Z,
                          generate_save_file_path)

# Define the system.
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = PAULI_Z / 2
# Use a + a^{dagger} as the drive term to control.
hamiltonian = lambda params, t: (H_SYSTEM_0
                                 + params[0] * ANNIHILATION_OPERATOR
                                 + anp.conjugate(params[0]) * CREATION_OPERATOR)

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,), axis=0)
TARGET_STATES = anp.stack((TARGET_STATE_0,), axis=0)
COSTS = [TargetInfidelity(TARGET_STATES)]

# Define the optimization.
COMPLEX_CONTROLS = True
CONTROL_COUNT = 1
EVOLUTION_TIME = CONTROL_STEP_COUNT = 10
ITERATION_COUNT = 1000

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_PATH = "./out"
SAVE_FILE_NAME = "test"
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)


def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_STEP_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES,
                                         complex_controls=COMPLEX_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         save_file_path=SAVE_FILE_PATH,
                                         save_iteration_step=SAVE_ITERATION_STEP,)


if __name__ == "__main__":
    main()
