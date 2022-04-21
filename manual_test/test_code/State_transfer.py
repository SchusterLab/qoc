from qoc.standard.constants import harmonic, transmon,  Identity
from scipy.sparse import kron
import os
os.environ['OMP_NUM_THREADS'] = '1' # set number of OpenMP threads to run in parallel
from scipy import signal
import matplotlib.pyplot as plt
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                           )
import numpy as np

from qutip import (qsave, qload)
from scipy.sparse import dia_matrix


def simulation(fock, dim_c, dim_trans, w_c, w_t, anharmonicity, g, evolution_time, step, initial):
    asd, b_dag, b = transmon(w_01=w_t, anharmonicity=anharmonicity, H_size=dim_trans)
    delta = w_t-w_c
    a_dag, a = harmonic(dim_c)
    I_t = Identity(dim_trans)
    I_c = Identity(dim_c)
    H_trans = delta*np.dot(b_dag, b)+ 1 / 2 * anharmonicity * np.dot(np.dot(b_dag, b), np.dot(b_dag, b) - I_t)
    H_trans = kron(H_trans, I_c, format="csc")

    H_0 = g * kron(b, a_dag, format="csc") + g * kron(b_dag, a, format="csc") + H_trans
    H_control = [(kron(b, I_c) + kron(b_dag, I_c)), kron(np.dot(b_dag, b), I_c)]
    hamiltonian = lambda controls, time: (H_0
                                          + controls[0] * H_control[0]
                                          + controls[1] * H_control[1]
                                          )
    T_initial = np.zeros(dim_trans)
    T_initial[0] = 1
    C_initial = np.zeros(dim_c)
    C_initial[0] = 1
    C_final = np.zeros(dim_c)
    C_final[fock] = 1
    #    C_final=coherent_state(N,alpha)
    Initial_state = np.kron(T_initial, C_initial)
    Initial_state = Initial_state.reshape(1, Initial_state.shape[0], 1)
    Target = np.kron(T_initial, C_final)
    Target = Target.reshape((1, Target.shape[0], 1))
    CONTROL_COUNT = 2
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = step + 1
    ITERATION_COUNT = 1000
    COSTS = [TargetStateInfidelity(Target, cost_multiplier=1)]

    # Define output.
    LOG_ITERATION_STEP = 1

    manual_parameter = {"control_hamiltonian": H_control, "manual_gradient_mode": True, "tol": 1e-8}
    def Impose(control):
        control[0] = control[control.shape[0] - 1] = 0
        return control
    if fock < 2:
        H_0 = H_0.toarray()
        for i in range(len(H_control)):
            H_control[i] = H_control[i].toarray()
        result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, evolution_time, hamiltonian,
                                             Initial_state, SYSTEM_EVAL_COUNT,
                                             complex_controls=False,
                                             initial_controls=initial,
                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP, min_error=0.1,

                                             impose_control_conditions=Impose,

                                             save_iteration_step=1,

                                             )
    else:
        result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, evolution_time, hamiltonian,
                                             Initial_state, SYSTEM_EVAL_COUNT,
                                             complex_controls=False,
                                             initial_controls=initial,
                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP, min_error=0.05,


                                             manual_parameter=manual_parameter, impose_control_conditions=Impose,
                                             save_iteration_step=1,
                                             )
    return result
def initial_pulse():
    t = np.linspace(-500, 500, 4001, endpoint=False)
    i = 0.1*signal.gausspulse(t, fc=0.03,bw=0.05,)
    controls = np.zeros((4001,2))
    for j in range(4001):
        controls[j][0]=i[j]
        controls[j][1]=i[j]
    return controls
pre=2*np.pi
for i in range(1,100):
    if i==1:
        pulse = initial_pulse()
    else:
        pulse = result.best_controls
    if i>=10:
        dim =3*i
    dim = 2 * i
    result = simulation(dim - 1, dim, 6, 6 * pre, 3 * pre, -0.225 * pre, 0.01 * pre, 1000, 4000, pulse)
