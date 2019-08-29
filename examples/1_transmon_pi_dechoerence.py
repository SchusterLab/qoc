"""
1_transmon_pi_decoherence.py - This module demonstrates
an example of grape on the lindblad master equation
using time discrete control parameters to evolve a transmon qubit
form the ground state to the first excited state
when the system is subject to noise.
"""

import autograd.numpy as anp
from qoc import grape_lindblad_discrete
from qoc.standard import (TargetDensityInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_PLUS,
                          generate_save_file_path,)

# Define the system.
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
# E.q. 19 (p. 6) of
# https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = SIGMA_Z / 2
# Use a + a^{dagger} as the drive term to control.
hamiltonian = lambda controls, time: (H_SYSTEM_0
                                      + controls[0] * ANNIHILATION_OPERATOR
                                      + anp.conjugate(controls[0]) * CREATION_OPERATOR)
# Subject the system to T1 type decoherence per fig. 11 of
# https://www.sciencedirect.com/science/article/pii/S0003491617301252.
LINDBLAD_OPERATORS = anp.stack((SIGMA_PLUS,))
T1 = 1e5 # 1e5 nanoseconds = 1e2 microseconds
GAMMA_1 = 1e-5 #GHz
LINDBLAD_DISSIPATORS = anp.stack((GAMMA_1,))
lindblad_data = lambda time: (LINDBLAD_DISSIPATORS, LINDBLAD_OPERATORS)

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,), axis=0)
TARGET_STATES = anp.stack((TARGET_STATE_0,), axis=0)
INITIAL_DENSITIES = anp.matmul(INITIAL_STATES, conjugate_transpose(INITIAL_STATES))
TARGET_DENSITIES = anp.matmul(TARGET_STATES, conjugate_transpose(TARGET_STATES))
# Note that the TargetDensityInfidelity function uses the frobenius inner product,
# so even if our matrices are identical, the total optimization error may
# not reach zero. In this case, the minimum error that we could reach
# should be 0.75.
COSTS = [TargetDensityInfidelity(TARGET_DENSITIES)]

# Define the optimization.
COMPLEX_CONTROLS = True
CONTROL_COUNT = 1
EVOLUTION_TIME = CONTROL_STEP_COUNT = 10 # nanoseconds
ITERATION_COUNT = int(1e4)

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_PATH = "./out"
SAVE_FILE_NAME = "transmon_pi_decoherence"
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)


def main():
    result = grape_lindblad_discrete(CONTROL_COUNT, CONTROL_STEP_COUNT,
                                     COSTS, EVOLUTION_TIME,
                                     INITIAL_DENSITIES,
                                     complex_controls=COMPLEX_CONTROLS,
                                     hamiltonian=hamiltonian,
                                     iteration_count=ITERATION_COUNT,
                                     lindblad_data=lindblad_data,
                                     log_iteration_step=LOG_ITERATION_STEP,
                                     save_file_path=SAVE_FILE_PATH,
                                     save_iteration_step=SAVE_ITERATION_STEP,)


if __name__ == "__main__":
    main()
