"""
This notebook shows memory measurement of Fock state generation
"""

import os

import autograd.numpy as anp

from qoc import grape_schroedinger_discrete
from qoc.standard import (krons, TargetStateInfidelity,)

from filprofiler.api import profile


def get_memory(N, l):
    """
    This function contains a simple example of grape
    on the schroedinger equation using time discrete
    control parameters to evolve a cavity from the
    Fock state 0 to the Fock state 1 with a transmon
    as an ancilla

    Args:
    N :: int - Numbers of time steps
    l :: int - Hilbert space dimension
    """
    PI_2 = 2 * anp.pi
    W_T = 3.5  # GHz
    W_C = 3.9
    G_TC = PI_2 * 0.1
    ALPHA = 0.225

    # define the system
    CAVITY_STATE_COUNT = l
    TRANSMON_STATE_COUNT = 2

    ens = anp.array([PI_2*ii*(W_T - 0.5*(ii-1)*ALPHA) for ii in anp.arange(TRANSMON_STATE_COUNT)])
    Q_x = anp.diag(anp.sqrt(anp.arange(1, TRANSMON_STATE_COUNT)), 1)+anp.diag(anp.sqrt(anp.arange(1, TRANSMON_STATE_COUNT)), -1)
    Q_z = anp.diag(anp.arange(0, TRANSMON_STATE_COUNT))
    I_q = anp.identity(TRANSMON_STATE_COUNT)
    H_q = anp.diag(ens)

    mode_ens = anp.array([ PI_2*ii*(W_C) for ii in anp.arange(CAVITY_STATE_COUNT)])
    M_x = anp.diag(anp.sqrt(anp.arange(1, CAVITY_STATE_COUNT)), 1)+anp.diag(anp.sqrt(anp.arange(1, CAVITY_STATE_COUNT)), -1)
    H_m = anp.diag(mode_ens)
    I_m = anp.identity(CAVITY_STATE_COUNT)
    # Notice how the state vectors are specified as column vectors.
    CAVITY_VACUUM = anp.zeros((CAVITY_STATE_COUNT, 1))
    CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
    CAVITY_ZERO[0, 0] = 1.
    CAVITY_ONE = anp.copy(CAVITY_VACUUM)
    CAVITY_ONE[1, 0] = 1.

    TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, 1))
    TRANSMON_ZERO = anp.copy(TRANSMON_VACUUM)
    TRANSMON_ZERO[0, 0] = 1.
    TRANSMON_ONE = anp.copy(TRANSMON_VACUUM)
    TRANSMON_ONE[1, 0] = 1.

    SYSTEM_HAMILTONIAN = krons(H_q, I_m) + krons(I_q, H_m) + G_TC*krons(Q_x, M_x)
    CONTROL_0 = krons(Q_x, I_m)
    CONTROL_1 = krons(Q_z, I_m)

    def hamiltonian(controls, time):
        return (SYSTEM_HAMILTONIAN
                + controls[0] * CONTROL_0
                + controls[1] * CONTROL_1)

    # Additionally, we need to specify information to qoc about...
    # how long our system will evolve for
    EVOLUTION_TIME = 40. #ns
    # how many controls we have
    CONTROL_COUNT = 2
    # what domain our controls are in
    COMPLEX_CONTROLS = False
    # where our controls are positioned in time
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = N + 1
    # initial controls
    INITIAL_CONTROLS = 0.1*anp.ones((CONTROL_EVAL_COUNT, CONTROL_COUNT))


    # fock state transition |0, g> -> |1, g>.
    INITIAL_STATE_0 = krons(TRANSMON_ZERO, CAVITY_ZERO)
    INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
    assert(INITIAL_STATES.ndim == 3)
    TARGET_STATE_0 = krons(TRANSMON_ZERO, CAVITY_ONE)
    TARGET_STATES = anp.stack((TARGET_STATE_0,))

    COSTS = [TargetStateInfidelity(TARGET_STATES)]

    ITERATION_COUNT = 1
    LOG_ITERATION_STEP = 1

    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                         complex_controls=COMPLEX_CONTROLS,
                                         initial_controls=INITIAL_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,)

if __name__ == "__main__":
    # total number of time step
    N_ = 40
    h_dims = [2*int(10**i) for i in range(0, 4, 1)]

    # get current directory
    current_dir = os.getcwd()

    # using fil
    for dim in h_dims:
        final_dir = os.path.join(current_dir, f'fil-result/ad_mem/dim_{dim}-N_{N_}')
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        fil_result = profile(lambda: get_memory(N_, dim), final_dir)