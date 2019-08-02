"""
0_transmon_pi.py - a module for the simplest problem imaginable
"""

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (Adam, SGD, LBFGSB, TargetInfidelity)
from qoc.util import (conjugate_transpose,
                      get_annihilation_operator,
                      get_creation_operator,
                      PAULI_Z,)

# Define the system.
HILBERT_SIZE = 2
ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
# E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
H_SYSTEM_0 = PAULI_Z / 2
H_CONTROL_0 = ANNIHILATION_OPERATOR + CREATION_OPERATOR
H_CONTROL_0_DAGGER = conjugate_transpose(H_CONTROL_0)
hamiltonian = lambda params, t: (H_SYSTEM_0
                                 + params[0] * H_CONTROL_0
                                 + (anp.conjugate(params[0])
                                    * H_CONTROL_0_DAGGER))

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,), axis=0)
TARGET_STATES = anp.stack((TARGET_STATE_0,), axis=0)
COSTS = [TargetInfidelity(TARGET_STATES)]

# Define the optimization.
PARAM_COUNT = 1
PULSE_TIME = 100
PULSE_STEP_COUNT = 100
ITERATION_COUNT = 100
INITIAL_PARAMS = anp.ones((PULSE_STEP_COUNT, PARAM_COUNT))
MAX_PARAM_AMPLITUDES = anp.ones(PARAM_COUNT) * 10000
OPTIMIZER = LBFGSB()

# Define output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_PATH = "."
SAVE_FILE_NAME = "test"


def main():
    result = grape_schroedinger_discrete(COSTS, hamiltonian, INITIAL_STATES,
                                         ITERATION_COUNT, PARAM_COUNT,
                                         PULSE_STEP_COUNT, PULSE_TIME,
                                         initial_params=INITIAL_PARAMS,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_param_amplitudes=MAX_PARAM_AMPLITUDES,
                                         optimizer=OPTIMIZER,
                                         save_file_name=SAVE_FILE_NAME,
                                         save_iteration_step=SAVE_ITERATION_STEP,
                                         save_path=SAVE_PATH,)


if __name__ == "__main__":
    main()
