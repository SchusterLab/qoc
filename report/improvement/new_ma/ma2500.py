
import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path, )
from memory_profiler import profile
import numpy as np

from qutip import (qsave,qload)
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix

def get_memory_manually(N,l):
    """
    0_transmon_pi.py - This module demonstrates
    a simple example of grape on the schroedinger equation
    using time discrete control parameters to evolve a transmon qubit
    form the ground state to the first excited state.
    """


    HILBERT_SIZE = l
    # E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
    # E.q. 19 (p. 6) of https://arxiv.org/abs/1904.06560.
    diagnol=np.arange(HILBERT_SIZE)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, HILBERT_SIZE + 1))
    state = (1 / np.sqrt(HILBERT_SIZE)) * np.ones(HILBERT_SIZE)
    data = [low_diagnol, diagnol, up_diagnol]
    offsets = [-1, 0, 1]

    H_SYSTEM_0= dia_matrix((data, offsets), shape=(HILBERT_SIZE, HILBERT_SIZE)).todense()
    sigmax = dia_matrix(([low_diagnol, up_diagnol], [-1, 1]), shape=(HILBERT_SIZE, HILBERT_SIZE)).todense()
    ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
    CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
    sigmax = ANNIHILATION_OPERATOR + CREATION_OPERATOR
    H_SYSTEM_0=np.matmul(CREATION_OPERATOR,ANNIHILATION_OPERATOR)
    # Only real control amplitutdes are supported!
    hamiltonian = lambda controls, time: (H_SYSTEM_0
                                          + controls[0] * sigmax)

    # Define the problem.
    CAVITY_VACUUM = anp.zeros((l, 1))
    CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
    CAVITY_ZERO[0, 0] = 1
    CAVITY_ONE = anp.copy(CAVITY_VACUUM)
    CAVITY_ONE[1, 0] = 1
    CAVITY_VACUUM= anp.zeros((l, 1))

    INITIAL_STATE_0=anp.copy(CAVITY_VACUUM)
    INITIAL_STATE_0[0,0]=1
    TARGET_STATE_0 = anp.copy(CAVITY_VACUUM)
    TARGET_STATE_0[1,0]=1
    INITIAL_STATES = anp.stack((INITIAL_STATE_0,), axis=0)
    TARGET_STATES = anp.stack((TARGET_STATE_0,), axis=0)
    COSTS = [TargetStateInfidelity(TARGET_STATES)]

    # Define the optimization.
    COMPLEX_CONTROLS = False
    CONTROL_COUNT = 1
    EVOLUTION_TIME = N  # nanoseconds
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = N + 1
    ITERATION_COUNT = 1

    # Define output.
    LOG_ITERATION_STEP = 1
    SAVE_ITERATION_STEP = 1
    SAVE_PATH = "./out"
    SAVE_FILE_NAME = "transmon_pi"
    SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
    CONTROL_HAMILTONIAN = [sigmax]

    manual_parameter = {"control_hamiltonian": CONTROL_HAMILTONIAN, "manual_gradient_mode": True}

    def main():
        result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, EVOLUTION_TIME, hamiltonian,
                                             INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                             complex_controls=COMPLEX_CONTROLS,
                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP,
                                             manual_parameter=manual_parameter
                                             )
    if __name__ == "__main__":
        main()
get_memory_manually(1,2500)