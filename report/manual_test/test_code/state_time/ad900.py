from qoc.standard.constants import harmonic, transmon, coherent_state, Identity
from scipy.sparse import kron
import numpy as np
import sys
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity, ControlNorm, ControlVariation, generate_save_file_path,
                          ForbidStates)
from qoc.standard import Adam
from scipy import signal
from qutip import coherent
from qoc.models.operationpolicy import OperationPolicy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
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

from qutip import (qsave, qload)
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix


def simulation(fock, dim_c, dim_trans, w_c, w_t, anharmonicity, g, evolution_time, step, initial, max_con):
    asd, b_dag, b = transmon(w_01=w_t, anharmonicity=anharmonicity, H_size=dim_trans)

    a_dag, a = harmonic(dim_c)
    I_t = Identity(dim_trans)
    I_c = Identity(dim_c)
    H_trans = 1 / 2 * anharmonicity * np.dot(np.dot(b_dag, b), np.dot(b_dag, b) - I_t)
    H_trans = kron(H_trans, I_c, format="csc")

    H_0 = g * kron(b, a_dag, format="csc") + g * kron(b_dag, a, format="csc") + H_trans
    H_control = [(kron(b, I_c) + kron(b_dag, I_c)), kron(1j * (b - b_dag), I_c), kron(np.dot(b_dag, b), I_c)]
    hamiltonian = lambda controls, time: (H_0
                                          + controls[0] * H_control[0]
                                          + controls[1] * H_control[1]
                                          + controls[2] * H_control[2])
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
    CONTROL_COUNT = 4
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = step + 1
    ITERATION_COUNT = 1

    max_control_norms = max_con * np.ones(CONTROL_COUNT)
    cost_first = ControlVariation(control_count=CONTROL_COUNT,
                                  control_eval_count=CONTROL_EVAL_COUNT,
                                  cost_multiplier=0.01,
                                  max_control_norms=max_control_norms * 0.2,
                                  order=1)
    cost_second = ControlVariation(control_count=CONTROL_COUNT,
                                   control_eval_count=CONTROL_EVAL_COUNT,
                                   cost_multiplier=0.001,
                                   max_control_norms=max_control_norms * 0.05,
                                   order=1)

    COSTS = [TargetStateInfidelity(Target, cost_multiplier=1)]

    # Define output.
    LOG_ITERATION_STEP = 1

    manual_parameter = {"control_hamiltonian": H_control, "manual_gradient_mode": True, "tol": 1e-8}
    optimizer = Adam(beta_1=0.9, beta_2=0.999, clip_grads=None,
                     epsilon=1e-8, learning_rate=1e-3,
                     learning_rate_decay=None, operation_policy=OperationPolicy.CPU,
                     scale_grads=None)

    def Impose(control):
        control[0] = control[control.shape[0] - 1] = 0
        return control

    SAVE_PATH = "./out"
    SAVE_FILE_NAME = "fock"
    SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)

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
                                             max_control_norms=max_control_norms,
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
                                             max_control_norms=max_control_norms,
                                             impose_control_conditions=Impose,
                                             save_iteration_step=1,
                                             )
    return result
pre=2*np.pi
dim=350
simulation(1,dim,6,3.9*pre,3.5*pre,-0.225*pre,0.1*pre,2.5,10,None,5)
