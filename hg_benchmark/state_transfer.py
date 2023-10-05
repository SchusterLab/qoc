from scipy.sparse import kron
import os
os.environ['OMP_NUM_THREADS'] = '1' # set number of OpenMP threads to run in parallel
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity, OperatorAverage)
from scipy.sparse import dia_matrix,identity
import numpy as np

def Identity(H_size):
    return identity(H_size)

def harmonic(H_size):
    diagnol = np.arange(H_size)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, H_size + 1))
    a= dia_matrix(([ up_diagnol], [ 1]), shape=(H_size, H_size)).tocsc()
    a_dag=dia_matrix(([ low_diagnol], [ -1]), shape=(H_size, H_size)).tocsc()
    return a_dag,a

def transmon(w_01,anharmonicity,H_size):
    b_dag,b=harmonic(H_size=H_size)
    H0=b_dag.dot(b)
    diagnol=np.ones(H_size)
    I= dia_matrix(([ diagnol], [ 0]), shape=(H_size, H_size)).tocsc()
    H0=w_01*H0+anharmonicity/2*H0*(H0-I)
    return H0,b_dag,b

def projector_tran(dim_trans,dim_c,i,mode):
    I_c=Identity(dim_c)
    tran0 = np.zeros(dim_trans)
    tran0[i]=1
    tran0=dia_matrix(([tran0],[0]),shape=(dim_trans, dim_trans)).tocsc()
    tran0=kron(tran0,I_c)
    return tran0.toarray()
def total_cost(dim_trans,dim_c,mode,costs):
    costs.append(OperatorAverage(projector_tran(dim_trans,dim_c,dim_trans-1,mode), cost_multiplier=1/3))
    costs.append(OperatorAverage(projector_tran(dim_trans,dim_c,dim_trans-2,mode), cost_multiplier=1/3))
    costs.append(OperatorAverage(projector_tran(dim_trans,dim_c,dim_trans-3,mode), cost_multiplier=1/3))
    return costs
def simulation(fock, dim_c , dim_trans, w_c, w_t, anharmonicity, g, evolution_time, step, mode):
    asd, b_dag, b = transmon(w_01=w_t, anharmonicity=anharmonicity, H_size=dim_trans)
    delta = w_t-w_c
    a_dag, a = harmonic(dim_c)
    I_t = Identity(dim_trans)
    I_c = Identity(dim_c)
    H_trans = delta*np.dot(b_dag, b)+ 1 / 2 * anharmonicity * np.dot(np.dot(b_dag, b), np.dot(b_dag, b) - I_t)
    H_trans = kron(H_trans, I_c, format="csc")
    H_0 = (g * kron(b, a_dag, format="csc") + g * kron(b_dag, a, format="csc") + H_trans)*1j*(-1j)
    H_control = [(kron(b, I_c) + kron(b_dag, I_c))*1j*(-1j), kron(np.dot(b_dag, b), I_c)*1j*(-1j)]

    T_initial = np.zeros(dim_trans)
    T_initial[0] = 1
    C_initial = np.zeros(dim_c)
    C_initial[0] = 1
    C_final = np.zeros(dim_c)
    C_final[fock] = 1
    #    C_final=coherent_state(N,alpha)
    Initial_state = np.array([np.kron(T_initial, C_initial)])
    # Initial_state = Initial_state.reshape(1, Initial_state.shape[0],1)
    Target = np.array([np.kron(T_initial, C_final)])
    # Target = Target.reshape(1, Target.shape[0],1)
    CONTROL_EVAL_COUNT = step
    ITERATION_COUNT = 1

    COSTS = [TargetStateInfidelity(target_states=Target,cost_multiplier=0.99),]
    COSTS = total_cost(dim_trans,dim_c,mode,COSTS)

    H_0 = H_0.toarray()
    for i in range(len(H_control)):
        H_control[i] = H_control[i].toarray()
    result = grape_schroedinger_discrete(H_0, H_control, CONTROL_EVAL_COUNT,
                                         COSTS, evolution_time,
                                         initial_states=Initial_state,
                                         iteration_count=ITERATION_COUNT, gradients_method=mode, expm_method="taylor")
    return result