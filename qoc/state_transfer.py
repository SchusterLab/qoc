from qoc.standard.constants import harmonic, transmon,  Identity
from scipy.sparse import kron
from scipy.sparse import csr_matrix
import os
os.environ['OMP_NUM_THREADS'] = '1' # set number of OpenMP threads to run in parallel
from scipy import signal
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,ForbidStatesprojector)
import numpy as np
def projector_tran(dim_trans,dim_c,i):
    I_c=np.identity(dim_c)
    tran0 = np.zeros((dim_trans,dim_trans))
    tran0[i][i]=1
    tran0=np.kron(tran0,I_c)
    return tran0
def projector_tran_set(dim_trans,dim_c):
    a=[]
    a.append(projector_tran(dim_trans,dim_c,dim_trans-1))
    a.append(projector_tran(dim_trans,dim_c,dim_trans-2))
    a.append(projector_tran(dim_trans,dim_c,dim_trans-3))
    return np.array(a)
def simulation(fock, dim_c , dim_trans, w_c, w_t, anharmonicity, g, evolution_time, step, mode):
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

    COSTS = [TargetStateInfidelity(target_states=Target,cost_multiplier=0.99),
             ForbidStatesprojector(projector_tran_set(dim_trans,dim_c),system_eval_count=SYSTEM_EVAL_COUNT,cost_multiplier=0.01),]
    max_control_norms=np.array([6., 6.])
    # Define output.
    LOG_ITERATION_STEP = 1
    def Impose(control):
        control[0] = control[control.shape[0] - 1] = 0
        return control
    if mode == "AD":
        H_0 = H_0.toarray()
        for i in range(len(H_control)):
            H_control[i] = H_control[i].toarray()
        manual_parameter = {"control_hamiltonian": H_control, "manual_gradient_mode": False, "tol": 1e-8}
    else:

        manual_parameter = {"control_hamiltonian": H_control, "manual_gradient_mode": True, "tol": 1e-8}
    name=["transmon0","transmon1","transmon2"]
    states_plot=[name,projector_tran_set(dim_trans,dim_c)]
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, evolution_time, hamiltonian,
                                             Initial_state, SYSTEM_EVAL_COUNT,
                                             complex_controls=False,
                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP, min_error=0.001,
                                              max_control_norms= max_control_norms,
                                             manual_parameter=manual_parameter, impose_control_conditions=Impose,
                                             )
    return result
def initial_pulse():
    t = np.linspace(-500, 500, 4001, endpoint=False)
    i = 1*signal.gausspulse(t, fc=0.03,bw=0.05,)
    k=1*np.ones(4001)
    controls = np.zeros((4001,2))
    for j in range(4001):
        controls[j][0]=i[j]
        controls[j][1]=i[j]
    return controls
