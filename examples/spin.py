"""
spin.py
"""

from copy import copy

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z, SIGMA_X,
                          generate_save_file_path,)

MAX_AMP_0 = 2 * anp.pi * 3e-1
OMEGA = 2 * anp.pi * 1e-2

# Define the system.
HILBERT_SIZE = 2
H_SYSTEM_0 = OMEGA * SIGMA_Z / 2
H_CONTROL_0 = SIGMA_X / 2
hamiltonian = lambda controls, time: (
    H_SYSTEM_0
    + controls[0] * H_CONTROL_0
)
MAX_CONTROL_NORMS = anp.array([MAX_AMP_0])

# Define the problem.
INITIAL_STATE_0 = anp.array([[1], [0]])
TARGET_STATE_0 = anp.array([[0], [1]])
INITIAL_STATES = anp.stack((INITIAL_STATE_0,),)
TARGET_STATES = anp.stack((TARGET_STATE_0,),)
COSTS = [
    TargetStateInfidelity(TARGET_STATES)
]

# Define the optimization.
COMPLEX_CONTROLS = False
CONTROL_COUNT = 1
EVOLUTION_TIME = 100 # nanoseconds
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = 2 * int(EVOLUTION_TIME) + 1
ITERATION_COUNT = 1000

# Define output.
SAVE_PATH = "./out"
SAVE_FILE_NAME = "spin"
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES_GRAPE = True


GRAPE_CONFIG = {
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "complex_controls": COMPLEX_CONTROLS,
    "max_control_norms": MAX_CONTROL_NORMS,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "save_iteration_step": SAVE_ITERATION_STEP,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_GRAPE,
}

def do_grape():
    save_file_path = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)



def main():
    do_grape()


if __name__ == "__main__":
    main()
